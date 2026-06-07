# Delivery Report — Factory Runtime Remediation

## Executive summary

The Factory runtime remediation has now completed two technical layers:

1. Operational autonomy repair: blockers are classified, watchdog alerts exist, and blocked projects can be repaired/continued safely.
2. Repo-first Runtime Contract v1: required docs must be listed in `DOCUMENTATION_INDEX.md`, and project-local Factory artifacts must have a git checkpoint before high/critical readiness/delivery closure.

This directly implements Jean's correction: Factory DB remains the operational truth, but the project repository and commits are the professional source of truth for project documentation and implementation history.

## Delivered code/runtime changes

| Area | Evidence |
|---|---|
| Blocker classification/action records | `hermes_cli/factory_pg.py`, `~/.hermes/scripts/factory_blocker_detector.py` |
| Watchdog alerts/notification path | `factory_watchdog_alerts.py`, cron `factory-watchdog-alerts`, status payload alerts |
| Blocked project dispatch repair | Factory dispatcher claim predicates and `force_tick` repair/reopen path |
| Repo-first document index enforcement | `docs_not_indexed` anomaly and critical readiness blocker |
| Git commit checkpoint enforcement | `uncommitted_project_artifacts` anomaly and critical readiness blocker |

## Delivered PM/artifact changes

- Required docs under `factory/projects/factory-runtime-remediation/`.
- `PROJECT_GLOBAL_VISION.md` as compact project memory for Zeus/builders.
- `FACTORY_RUNTIME_CONTRACT_V1.md` documenting the new runtime contract.
- `BUILDER_CONTEXT_TEMPLATE.md` documenting how Claude Code/Codex/OpenHands should receive repo context.
- Updated `DOCUMENTATION_INDEX.md`, `PRD.md`, `SPRINT_PLAN.md`, `TASK_GRAPH.md`, and `QA_REPORT.md`.

## Verification summary

Repo-first Runtime Contract v1 was implemented with TDD:

```text
RED: 3 expected failures for missing docs_not_indexed/uncommitted artifact behavior.
GREEN: 20 passed in test_factory_canonical_runtime.py.
GREEN: 22 passed, 1 warning in canonical runtime + factory tools tests.
```

## Residual risks

| Risk | Mitigation |
|---|---|
| Builders still receive prose prompts rather than structured `run_result.json` | Next increment should implement builder context generation and machine-validated run-result output. |
| Direct Zeus recursion can self-approve | Keep tests deterministic and route review to independent worker/profile when the runtime is stable enough. |

## Final close evidence

Final closure was validated after:

1. Project-local artifacts were committed in the repo-first remediation worktree.
2. `hermes factory project reconcile factory-runtime-remediation --json` returned `anomalies=[]`.
3. Final `critical_readiness` and `delivery` gates passed after the commit checkpoint.
4. The project-specific Notion PM page was linked and verified as human reporting projection.
