# Hermes OS v3 Work Graph Roadmap

## Purpose

Hermes OS v3 becomes a control plane, governance layer, and execution orchestrator. The next central engine is the Work Graph Compiler.

The system path is:

```text
Architecture
  -> Work Graph
  -> Execution
  -> Validation
  -> Dashboard
  -> Continuous Improvement
```

## Work Graph Compiler

`hermes plan` reads architecture artifacts and emits `workgraph.json`.

Inputs:

- `PROJECT.md`
- `DOMAIN.md`
- `WORKFLOWS.md`
- `DASHBOARD.md`
- `METRICS.md`
- `APPROVALS.md`
- `AGENTS.md`

Output:

- Project.
- Epics.
- Workflows.
- Tasks.
- Subtasks.
- Dependencies.
- Approvals.
- Artifacts.
- Metrics.
- Agent assignments.
- Execution results.
- Validation results.

## Execution Model

Execution becomes graph-aware:

```text
Work Graph
  -> Dependency Resolution
  -> Execution Queue
  -> Agent Assignment
  -> Validation
  -> Artifacts
  -> Dashboard Update
```

## Dashboards

Hermes OS v3 adds:

- Architecture dashboard.
- Work graph dashboard.
- Agent dashboard.
- Workspace dashboard.

## Control Plane

`hermes workspace` summarizes architecture score, task health, execution status, blockers, approvals, and runtime usage across workspace projects.

## Continuous Improvement

Autonomous review loops should periodically review projects, identify missing docs and workflows, generate roadmap updates, persist score history, and require approval for high-risk changes.
