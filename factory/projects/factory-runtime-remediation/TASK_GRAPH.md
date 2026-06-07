# Task Graph — Factory Runtime Remediation

```mermaid
graph TD
  F0[F0 Intake + methodology + Notion/docs]
  F1[F1 Blocker detector classification + action records]
  F2[F2 Cron/dashboard watchdog alerts + notifications]
  F3[F3 Dispatcher can repair/continue blocked projects]
  F4[F4 QA + live smoke + delivery report]
  F5[F5 Methodology debt correction: required docs + Notion + no waivers]
  F6[F6 Repo-first runtime contract: doc index + git checkpoint]

  F0 --> F1
  F0 --> F2
  F1 --> F3
  F2 --> F4
  F3 --> F4
  F4 --> F5
  F5 --> F6
```

## Factory DB tasks

| Task ID | Title | Status | Evidence |
|---|---|---|---|
| `factory-runtime-remediation-f0-intake-task-graph-and-remediation-arc` | F0 — Intake, task graph, remediation architecture | done | Project created, lane/task graph, docs started |
| `factory-runtime-remediation-f1-blocker-detector-classifications-and-` | F1 — Blocker detector classifications and actions | done | `factory_pg.py`, `factory_blocker_detector.py`, tests |
| `factory-runtime-remediation-f2-cron-dashboard-watchdog-alerts-and-te` | F2 — Watchdog alerts and notification path | done | alerts in status, cron `factory-watchdog-alerts` |
| `factory-runtime-remediation-f3-dispatcher-acts-on-blocked-projects-s` | F3 — Dispatcher acts on blocked projects | done | claim predicates and force_tick repair/reopen |
| `factory-runtime-remediation-f4-qa-live-smoke-and-delivery-report` | F4 — QA, smoke, report | done | focused runtime tests and script smokes |
| `factory-runtime-remediation-reconcile-missing-notion-project` | R0 — Notion PM tracker reconciliation | todo/hold | Side-effect/documentation projection; not technical runtime blocker |
| `factory-runtime-remediation-f5-repo-first-runtime-contract-v1` | F5/F6 — Repo-first runtime contract v1 | implemented in this worktree | Tests for `docs_not_indexed`, `uncommitted_project_artifacts`, critical readiness |

## Dependency rules

- F1 and F2 require F0.
- F3 requires F1.
- F4 requires F2 and F3.
- F5 is a correction dependency before final accepted/completed state.
