# File: tdemo_chaos.py
# Author: Gregor Lingl
# Date: 2009-06-24

# A demonstration of chaos

from turtle import *

N = 80

def f(x):
    return 3.9*x*(1-x)

def g(x):
    return 3.9*(x-x**2)

def h(x):
    return 3.9*x-3.9*x*x

def jumpto(x, y):
    penup(); goto(x,y)

def line(x1, y1, x2, y2):
    jumpto(x1, y1)
    pendown()
    goto(x2, y2)

def coosys():
    line(-1, 0, N+1, 0)
    line(0, -0.1, 0, 1.1)

def plot(fun, start, color):
    pencolor(color)
    x = start
    jumpto(0, x)
    pendown()
    dot(5)
    for i in range(N):
        x=fun(x)
        goto(i+1,x)
        dot(5)

def main():
    reset()
    setworldcoordinates(-1.0,-0.1, N+1, 1.1)
    speed(0)
    hideturtle()
    coosys()
    plot(f, 0.35, "blue")
    plot(g, 0.35, "green")
    plot(h, 0.35, "red")
    # Now zoom in:
    for s in range(100):
        setworldcoordinates(0.5*s,-0.1, N+1, 1.1)
    return "Done!"

if __name__ == "__main__":
    main()
    mainloop()
