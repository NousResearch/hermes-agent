#!/usr/bin/env python3
"""       turtle-example-suite:

        xtx_lindenmayer_indian.py

Each morning women in Tamil Nadu, in southern
India, place designs, created by using rice
flour and known as kolam on the thresholds of
their homes.

These can be described by Lindenmayer systems,
which can easily be implemented with turtle
graphics and Python.

Two examples are shown here:
(1) the snake kolam
(2) anklets of Krishna

Taken from Marcia Ascher: Mathematics
Elsewhere, An Exploration of Ideas Across
Cultures

"""
################################
# Mini Lindenmayer tool
###############################

from turtle import *

def replace( seq, replacementRules, n ):
    for i in range(n):
        newseq = ""
        for element in seq:
            newseq = newseq + replacementRules.get(element,element)
        seq = newseq
    return seq

def draw( commands, rules ):
    for b in commands:
        try:
            rules[b]()
        except TypeError:
            try:
                draw(rules[b], rules)
            except:
                pass


def main():
    ################################
    # Example 1: Snake kolam
    ################################


    def r():
        right(45)

    def l():
        left(45)

    def f():
        forward(7.5)

    snake_rules = {"-":r, "+":l, "f":f, "b":"f+f+f--f--f+f+f"}
    snake_replacementRules = {"b": "b+f+b--f--b+f+b"}
    snake_start = "b--f--b--f"

    drawing = replace(snake_start, snake_replacementRules, 3)

    reset()
    speed(3)
    tracer(1,0)
    ht()
    up()
    backward(195)
    down()
    draw(drawing, snake_rules)

    from time import sleep
    sleep(3)

    ################################
    # Example 2: Anklets of Krishna
    ################################

    def A():
        color("red")
        circle(10,90)

    def B():
        from math import sqrt
        color("black")
        l = 5/sqrt(2)
        forward(l)
        circle(l, 270)
        forward(l)

    def F():
        color("green")
        forward(10)

    krishna_rules = {"a":A, "b":B, "f":F}
    krishna_replacementRules = {"a" : "afbfa", "b" : "afbfbfbfa" }
    krishna_start = "fbfbfbfb"

    reset()
    speed(0)
    tracer(3,0)
    ht()
    left(45)
    drawing = replace(krishna_start, krishna_replacementRules, 3)
    draw(drawing, krishna_rules)
    tracer(1)
    return "Done!"

if __name__=='__main__':
    msg = main()
    print(msg)
    mainloop()
