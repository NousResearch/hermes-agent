---
name: constraint-checked-planning
description: Plan resource allocation and verify hard constraints.
version: 0.1.0
author: Ahmed Hassan (@AAH20)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Multi-Agent, Graph, Planning, Evaluation]
    category: Autonomous AI Agents
    related_skills: [hermes-agent]
---

# Constraint-Checked Planning Skill

Use a small, role-scoped agent team to produce a graph or capacity plan, then check its hard constraints with a deterministic local evaluator. This workflow improves traceability; it does not prove optimality or authorize real-world actions.

## When to Use

- A planning task has separable graph, domain, and review workstreams.
- The input and proposed assignments can be represented as JSON.
- A human can review the result before any external action.

Do not use this workflow to directly control data-center equipment, IoT devices, or physical fleets.

## Prerequisites

- Hermes with the `terminal`, `file`, and `kanban` toolsets.
- A checkout or workspace where the user has provided synthetic or authorized planning data.
- Python 3.10+ for the bundled constraint checker; it uses only the standard library.

## How to Run

1. Ask the user to define the objective, hard constraints, soft preferences, and human approval point. Record unresolved assumptions; do not fill in missing capacity or cost data.
2. Create a coordinator, a planner, a domain reviewer, and an independent verifier as Hermes profiles. Give each the minimum tools and data needed for its task.
3. Use a Kanban task tree for explicit dependencies. Keep agent decisions and handoffs in task comments so the run can be inspected.
4. Have the planner produce `problem.json` and `plan.json` following the Quick Reference contract. Do not let the planner mark its own output as verified.
5. Run the bundled evaluator from the installed skill directory (`~/.hermes/skills/autonomous-ai-agents/constraint-checked-planning` for the default Hermes home). Pass the problem and plan paths explicitly:

   ```text
   python3 scripts/evaluate_plan.py /path/to/problem.json /path/to/plan.json
   ```

   Resolve infeasible constraints by revising the plan, then rerun the same evaluator. Preserve both the original and revised result in the task evidence.
6. Ask the independent verifier to compare the plan and evaluator output against the original objective. Report feasibility, assumptions, unresolved risks, and token usage separately.
7. Stop at a human review gate. Do not issue infrastructure commands or claim the plan is globally optimal.

## Quick Reference

`problem.json` contains:

```json
{
  "budget_limit": 12000,
  "zone_capacity_kw": {"zone-a": 10, "zone-b": 10},
  "workloads": [
    {"id": "api", "required_kw": 8, "min_zones": 2}
  ]
}
```

`plan.json` contains one placement per workload and zone:

```json
{
  "placements": [
    {"workload_id": "api", "zone": "zone-a", "kw": 4, "monthly_cost": 5000},
    {"workload_id": "api", "zone": "zone-b", "kw": 4, "monthly_cost": 5000}
  ]
}
```

The evaluator reports JSON with `feasible`, `errors`, and aggregate totals. Feasibility means only that the declared capacity, budget, and zone-count checks pass; it is not an optimization or safety certification.

## Procedure

### 1. Establish the decision contract

Ask for the target outcome, data source, units, hard constraints, soft objectives, and who approves the result. Record the task in Kanban with a definition of done.

Done when the input files and approval point are explicit, and each numeric field has a unit.

### 2. Delegate bounded work

Assign graph decomposition, domain assumptions, and independent review to separate profiles. Give each task a named input and output file. The coordinator resolves disagreements in Kanban and keeps assumptions visible.

Done when each task has an owner, dependency, and inspectable deliverable.

### 3. Validate independently

Run `scripts/evaluate_plan.py` through `terminal` on the planner's files. Treat the tool output as authoritative only for the implemented checks; compare its declared assumptions with the original request.

Done when the evaluator returns a result, the independent reviewer has checked it, and violations or unsupported claims are listed.

### 4. Report for human review

Summarize the proposed plan, feasibility result, trade-offs, token usage, limits, and any decision needed from the user. Make no external changes until the user approves them.

Done when a human can reproduce the validation and make the next decision from the recorded artifacts.

## Pitfalls

- A feasible plan can still be expensive, fragile, or suboptimal.
- Missing or inaccurate source data invalidates the conclusion; show provenance and units.
- A model's cost estimate is not an invoice. Report measured token usage and label price assumptions.
- Multiple agents can repeat the same mistaken assumption. The verifier must check against the original input, not only the planner's summary.
- Do not include secrets, private telemetry, or operational credentials in prompts or fixtures.

## Verification

- From the skill directory, run `python3 scripts/evaluate_plan.py examples/problem.json examples/plan-feasible.json`; expect `feasible: true`.
- From the skill directory, run `python3 scripts/evaluate_plan.py examples/problem.json examples/plan-infeasible.json`; expect `feasible: false` with at least one explicit violation.
- Run the skill tests with the Hermes repository test runner.
- Report the Hermes version, Python version, evaluator result, and model token usage. Do not describe a synthetic fixture as a production benchmark.
