# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal MILP: integer variables with bounds, linear constraints.

Problem:
    Maximize: 5x + 3y
    Subject to: 2x + 4y >= 230, 3x + 2y <= 190, 10 <= y <= 50, x, y integer
"""

from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings


def main():
    problem = Problem("Simple MIP")
    x = problem.addVariable(vtype=INTEGER, name="V_x")
    y = problem.addVariable(lb=10, ub=50, vtype=INTEGER, name="V_y")
    problem.addConstraint(2 * x + 4 * y >= 230, name="C1")
    problem.addConstraint(3 * x + 2 * y <= 190, name="C2")
    problem.setObjective(5 * x + 3 * y, sense=MAXIMIZE)

    settings = SolverSettings()
    settings.set_parameter("time_limit", 60)
    problem.solve(settings)

    if problem.Status.name in ["Optimal", "FeasibleFound"]:
        print(f"Objective: {problem.ObjValue}")
        print(f"x = {x.getValue()}, y = {y.getValue()}")
    else:
        print(f"Status: {problem.Status.name}")


if __name__ == "__main__":
    main()
