# TaskPlanner — Automatic Task Decomposition

Rule-based planner that decomposes high-level goals into parallel execution graphs of `SwarmTask` objects.

## Usage

```python
from swarm.planner import TaskPlanner

planner = TaskPlanner()

# From free-form text
tasks = planner.decompose("""
1. Research the existing API surface
2. Implement the new endpoint
3. Write tests after step 2
4. Review the implementation after step 2
5. Merge everything once steps 3 and 4 pass
""")

# From explicit steps + dependency graph
tasks = planner.plan_from_steps(
    steps=["Research API", "Implement endpoint", "Write tests", "Review code", "Merge"],
    deps={2: [1], 3: [1], 4: [2, 3]},
)

# Inspect
print(planner.format_plan(tasks))
print("Max parallelism:", planner.estimate_parallelism(tasks))
print("Errors:", planner.validate_plan(tasks))
```

## Role Assignment

Roles are inferred from step text via keyword matching:

| Role       | Keywords                                      |
|------------|-----------------------------------------------|
| executor   | implement, build, create, write, code, fix    |
| critic     | review, critique, check quality, feedback     |
| verifier   | test, verify, validate, assert, confirm       |
| researcher | research, search, find, investigate, explore  |
| planner    | plan, design, architect, outline, decompose   |
| merger     | merge, combine, integrate, consolidate        |

## Dependency Detection

In `decompose()`, dependencies are parsed from natural language:

- **Explicit**: "after step 2", "once step 1", "depends on step 3" — references a specific prior step index.
- **Sequential**: "then", "after that", "next", "finally" — depends on the immediately preceding step.
- **Parallel**: steps with no dependency keywords run concurrently.

## Validation

`validate_plan()` checks:
- References to non-existent task IDs
- Self-dependencies
- Cycles (Kahn's algorithm / topological sort)
- Duplicate task IDs
