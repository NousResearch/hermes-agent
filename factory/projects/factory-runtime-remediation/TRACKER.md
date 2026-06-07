# Tracker — Factory Runtime Remediation

## Executive status

| Field | Value |
|---|---|
| Project ID | `factory-runtime-remediation` |
| Methodology | Hybrid |
| Source of truth | Agent Core Postgres `factory.*` |
| Repo artifacts | `factory/projects/factory-runtime-remediation/` |
| Notion | Required; created/linked during correction |
| Current state | Completed; repo-first Runtime Contract v1 delivered and R0 Notion reconciliation closed |

## Task tracker

| Task | Status | Owner | Reviewer | Evidence |
|---|---|---|---|---|
| F0 Intake/methodology | done | Zeus/factory-reporter | factory-orchestrator | Factory DB project/lane/task graph + docs |
| F1 Blocker classification | done | Zeus/claude-builder | quality-reviewer | `factory_pg.py`, detector script, tests |
| F2 Watchdog alerts | done | Zeus/claude-builder | devops-release | `factory_watchdog_alerts.py`, cron job |
| F3 Blocked dispatch repair | done | Zeus/claude-builder | security-reviewer | claim predicates, force_tick behavior |
| F4 QA/live smoke | done | Zeus/qa-verifier | factory-orchestrator | 23/23 tests + script smoke |
| F5 Notion/docs correction | done | Zeus/factory-reporter | factory-orchestrator | full docs + Notion metadata |
| R0 Notion reconciliation | done | factory-reconciler | factory-reconciler | Notion page verified; reconcile anomalies=0 |

## Open methodology debt

(none — all resolved)

## R0 Closure evidence

- Notion PM page: `Factory Runtime Remediation — Factory PM`
- Notion page ID: `37737b39-cad6-8198-a63e-faf0920031d4`
- Notion URL: `https://app.notion.com/p/Factory-Runtime-Remediation-Factory-PM-37737b39cad68198a63efaf0920031d4`
- `hermes factory project reconcile factory-runtime-remediation` → `anomalies: []`
- Acceptance criteria: all 3 met (Notion exists, metadata linked, DB stays source of truth)

## Close criteria

- `hermes factory status factory-runtime-remediation --json` has no `reconciliation_anomalies`.
- Notion metadata present.
- Required docs inventory complete.
- Delivery/reconciliation gate recorded after correction.
