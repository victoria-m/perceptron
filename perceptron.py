from __future__ import division
from numpy import random
from matplotlib import pyplot as plot
import csv

class Perceptron:
    def __init__(self, step_size, number_of_weights):
        # the step size represents the learning rate
        self.step_size = step_size

        # used to weigh the inputs, same size as a single input vector
        # generate a list of weights, each weight is between 0 and 1
        self.weights = random.rand(number_of_weights)
        self.weight_updates = 0
        # display the weight values since they are randomly generated
        print("Initial weights: " + str(self.weights))

        # begin with 100% error rate
        self.training_error_rate = 100
        self.training_errors = 0

    def learn(self, input_vector, answer):
        # if predicted correctly, error = 0, otherwise there is an error
        error = answer - self.predict(input_vector)
        if error != 0:
            self.training_errors += 1

        # in the case of an error, adjust the weights
        self.adjust_weights(input_vector, error)

    def adjust_weights(self, input_vector, error):
        for input, weight in zip(input_vector, self.weights):
            weight += error * self.step_size * input
        self.weight_updates += 1

    # used to classify the input vector
    def predict(self, input_vector):
        result = 0
        threshold = 0

        # multiply the inputs in a vector by their corresponding weights and sum them
        for input, weight in zip(input_vector, self.weights):
            result += input * weight

        # classify the input into its true class (1 or -1)
        return 1 if result >= threshold else -1


class Train:
    def __init__(self, step_size, training_data):
        self.perceptron = Perceptron(step_size, len(training_data[0]))
        self.training_data = training_data
        self.iterations = 0


    def train(self):
        MAX_ITERATIONS = 50
        MIN_ERROR_RATE = 10
        error_rate_decrease = 100
        previous_error_rate = 100000

        # run through the training data until one of these conditions is met:
            # a correct classification (with zero error) is reached
            # a pre-determined maximum number of iterations is reached
            # the error rate ceases to decrease significantly (stagnation)
        while self.iterations < MAX_ITERATIONS or error_rate_decrease < MIN_ERROR_RATE:
            for data in training_data:
                # allows the decision line to move left and right
                bias = 1

                x = data[0]
                y = data[1]
                answer = data[2]

                self.perceptron.learn([x, y, bias], answer)

            # update the error rate when finished running through the data
            self.perceptron.training_error_rate = self.perceptron.training_errors / len(training_data)

            # end training if error rate reaches 0
            if self.perceptron.training_error_rate is 0:
                return

            # if this is our first run through the data
            if previous_error_rate is 100000:
                previous_error_rate = self.perceptron.training_error_rate

            error_rate_decrease = (previous_error_rate - self.perceptron.training_error_rate) + previous_error_rate * 100
            self.iterations += 1


class Test:
    def __init__(self, test_data, trained_perceptron):
        self.test_data = test_data
        self.perceptron = trained_perceptron

        # start at 100% error
        self.error_rate = 100
        self.errors = 0

    def test(self):
        for data in test_data:
            # allows the decision line to move left and right
            bias = 1

            x = data[0]
            y = data[1]
            answer = data[2]

            error = answer - self.perceptron.predict([x, y, bias])
            if error != 0:
                self.errors += 1

            # update the testing error rate
            self.error_rate = self.errors / len(self.test_data)


# Step 1: read in the testing and training data

training_data = []
# create a list of training data vectors
with open("data/training_data.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        training_data.append([float(col) for col in row])

test_data = []
# create a list of test data vectors
with open("data/testing_data.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_data.append([float(col) for col in row])


# Step 2: train and test the perceptron

# create the trainer object, used for training the perceptron
trainer = Train(0.2, training_data)
trainer.train()

# create the tester object, used for testing the perceptron
tester = Test(test_data, trainer.perceptron)
tester.test()

print("Final decision boundary: %.2fx + %.2fy + %.2f = 0" % (trainer.perceptron.weights[0], trainer.perceptron.weights[1], trainer.perceptron.weights[2]))
print("Number of iterations made over training set: " + str(trainer.iterations))
print("Number of weight vector updates: " + str(trainer.perceptron.weight_updates))
print("Final misclassification error for training data: " + str(trainer.perceptron.training_error_rate) + "%")
print("Final misclassification error for test data: " + str(tester.error_rate) + "%")

# Step 3: plot the training and testing data

figure = plot.figure()
axis = figure.add_subplot(111)

# Step 3.1: plot training data graph
Y = []
X = []
answers = []

# populate the testing coordinate and answer lists
with open("data/training_data.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X.append(row[0])
        Y.append(row[1])
        answers.append(row[2])

# coordinates classified as 1 are blue, -1 are red
colors = {"1": 'blue', "-1": 'red'}

# plot the points
for i, j in enumerate(X):
    plot.scatter(X[i], Y[i], color = colors.get(answers[i]))

plot.title("Training Data Points")
plot.grid(True)
plot.show()

# Step 3.2: plot testing data graph
del Y[:]
del X[:]
del answers[:]

with open("data/testing_data.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X.append(row[0])
        Y.append(row[1])
        answers.append(row[2])

# plot the points
for i, j in enumerate(X):
    plot.scatter(X[i], Y[i], color = colors.get(answers[i]))

plot.title("Testing Data Points")
plot.show()
