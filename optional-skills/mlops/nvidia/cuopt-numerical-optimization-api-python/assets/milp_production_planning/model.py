# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Production planning: two products, resource limits (machine, labor, material), maximize profit.
"""

from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings


def main():
    problem = Problem("Production Planning")
    x1 = problem.addVariable(lb=10, vtype=INTEGER, name="Product_A")
    x2 = problem.addVariable(lb=15, vtype=INTEGER, name="Product_B")
    problem.addConstraint(2 * x1 + x2 <= 100, name="Machine_Time")
    problem.addConstraint(x1 + 3 * x2 <= 120, name="Labor_Hours")
    problem.addConstraint(4 * x1 + 2 * x2 <= 200, name="Material")
    problem.setObjective(50 * x1 + 30 * x2, sense=MAXIMIZE)

    settings = SolverSettings()
    settings.set_parameter("time_limit", 30)
    problem.solve(settings)

    if problem.Status.name in ["Optimal", "FeasibleFound"]:
        print(f"Product A: {x1.getValue()}, Product B: {x2.getValue()}")
        print(f"Total profit: {problem.ObjValue}")
    else:
        print(f"Status: {problem.Status.name}")


if __name__ == "__main__":
    main()
