# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PDLP warmstart: solve a similar LP faster by reusing solution context.

Warmstart is for LP only, not MILP.
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_METHOD,
    CUOPT_PDLP_SOLVER_MODE,
)
from cuopt.linear_programming.solver_settings import (
    SolverSettings,
    SolverMethod,
    PDLPSolverMode,
)


def main():
    print("=== Problem 1 ===")
    problem = Problem("LP1")
    x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")
    problem.addConstraint(4 * x + 10 * y <= 130, name="c1")
    problem.addConstraint(8 * x - 3 * y >= 40, name="c2")
    problem.setObjective(2 * x + y, sense=MAXIMIZE)

    settings = SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Stable2)
    problem.solve(settings)
    print(f"Objective: {problem.ObjValue}")

    warmstart_data = problem.getWarmstartData()
    print("\n=== Problem 2 (with warmstart) ===")
    new_problem = Problem("LP2")
    x = new_problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = new_problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")
    new_problem.addConstraint(4 * x + 10 * y <= 100, name="c1")
    new_problem.addConstraint(8 * x - 3 * y >= 50, name="c2")
    new_problem.setObjective(2 * x + y, sense=MAXIMIZE)
    settings.set_pdlp_warm_start_data(warmstart_data)
    new_problem.solve(settings)
    if new_problem.Status.name in ["Optimal", "PrimalFeasible"]:
        print(f"Objective: {new_problem.ObjValue}")


if __name__ == "__main__":
    main()
