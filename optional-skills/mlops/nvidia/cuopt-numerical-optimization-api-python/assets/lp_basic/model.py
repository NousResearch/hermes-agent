# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal LP: variables, constraints, objective, solve.

Problem:
    Maximize: x + y
    Subject to: x + y <= 10, x - y >= 0, x, y >= 0
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings


def main():
    problem = Problem("Simple LP")
    x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")
    problem.addConstraint(x + y <= 10, name="c1")
    problem.addConstraint(x - y >= 0, name="c2")
    problem.setObjective(x + y, sense=MAXIMIZE)

    settings = SolverSettings()
    settings.set_parameter("time_limit", 60)
    problem.solve(settings)

    if problem.Status.name in ["Optimal", "PrimalFeasible"]:
        print(f"Objective: {problem.ObjValue}")
        print(f"x = {x.getValue()}, y = {y.getValue()}")
    else:
        print(f"Status: {problem.Status.name}")


if __name__ == "__main__":
    main()
