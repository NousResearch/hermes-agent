# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LP with dual values and reduced costs.

Problem:
    Minimize: 3x + 2y + 5z
    Subject to: x + y + z = 4, 2x + y + z = 5, x, y, z >= 0
"""

from cuopt.linear_programming.problem import Problem, MINIMIZE


def main():
    problem = Problem("min_dual_rc")
    x = problem.addVariable(lb=0.0, name="x")
    y = problem.addVariable(lb=0.0, name="y")
    z = problem.addVariable(lb=0.0, name="z")
    problem.addConstraint(x + y + z == 4.0, name="c1")
    problem.addConstraint(2.0 * x + y + z == 5.0, name="c2")
    problem.setObjective(3.0 * x + 2.0 * y + 5.0 * z, sense=MINIMIZE)
    problem.solve()

    if problem.Status.name in ["Optimal", "PrimalFeasible"]:
        print(f"Objective: {problem.ObjValue}")
        for v in problem.getVariables():
            print(
                f"{v.VariableName} = {v.Value}, ReducedCost = {v.ReducedCost}"
            )
        for c in problem.getConstraints():
            print(f"{c.ConstraintName} DualValue = {c.DualValue}")
    else:
        print(f"Status: {problem.Status.name}")


if __name__ == "__main__":
    main()
