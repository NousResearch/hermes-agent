# Hermes Operating System Build Plan V15-V21

This document is the compact continuation plan for the next Hermes control-plane layer. It starts after the V8-V14 dashboard/design-system work and focuses on making Hermes act like the coordinating layer for TLC Capital Group OS without giving it unsafe autonomous authority too early.

## Status Summary

| Version | Capability | Status | Built Now | Remaining Runtime Hook |
| --- | --- | --- | --- | --- |
| V15 | Live project signal integration | 100% trackable infrastructure | Signal registry page, data contract, route, metadata, validation | Project-owned `/dashboard-snapshot` endpoints for Media Engine, Khashi VC, and future dashboards |
| V16 | Agent task routing | 100% trackable infrastructure | Work intake queue, task source/owner/priority model, route, validation | Real task executor and notification queue |
| V17 | Memory and decision ledger | 100% trackable infrastructure | Decision record model, decision dashboard, route, validation | Durable database-backed memory with review cadence |
| V18 | Model and cost routing | 100% trackable infrastructure | Provider policy dashboard, approval gates, route, validation | Live provider adapter metrics and approval workflow integration |
| V19 | Autonomous operating loops | 100% trackable infrastructure | Loop registry, cadence/output model, route, validation | Scheduled loop runner after V20 permission enforcement |
| V20 | Secure tool and permission layer | 100% trackable infrastructure | Permission policy model, approval/audit states, route, validation | Runtime enforcement middleware and audit store |
| V21 | TLC business OS scorecards | 100% trackable infrastructure | Business scorecard dashboard, revenue/cost maturity states, route, validation | Finance, analytics, project signal feeds |

## V15 Live Project Signal Integration

- [x] Create a standard signal registry for Hermes, Media Engine, Khashi VC, and future dashboards.
- [x] Show endpoint readiness, expected signals, and next steps.
- [x] Route exposed at `/live-signals`.
- [x] Metadata contract registered in `dashboard-page-metadata.ts`.
- [x] Validation script registered as `dashboard:v15:validate`.

The purpose is to make it obvious which projects can feed Hermes with health, capacity, cost, queue, and action-needed signals.

## V16 Agent Task Routing

- [x] Create a task intake model that captures title, source, owner, priority, status, and next step.
- [x] Show queued, assigned, blocked, and done task states.
- [x] Route exposed at `/task-routing`.
- [x] Metadata contract registered.
- [x] Validation script registered as `dashboard:v16:validate`.

The purpose is to turn dashboard signals into work Hermes can triage instead of leaving every dashboard as a passive report.

## V17 Memory And Decision Ledger

- [x] Create a decision ledger model with decision, reason, owner, status, and review timestamp.
- [x] Show active, superseded, and needs-review states.
- [x] Route exposed at `/decision-ledger`.
- [x] Metadata contract registered.
- [x] Validation script registered as `dashboard:v17:validate`.

The purpose is to keep Hermes from repeatedly rediscovering context, especially around why migrations, caps, provider choices, or permission choices were made.

## V18 Model And Cost Routing

- [x] Create model-routing policy records for local-first, cheap API first, and premium approval workflows.
- [x] Show approval-required states before expensive fallback.
- [x] Route exposed at `/model-routing`.
- [x] Metadata contract registered.
- [x] Validation script registered as `dashboard:v18:validate`.

The purpose is to make model usage governable instead of letting expensive providers get called silently.

## V19 Autonomous Operating Loops

- [x] Create recurring loop records with cadence, owner, status, and output.
- [x] Separate ready loops from draft loops.
- [x] Route exposed at `/operating-loops`.
- [x] Metadata contract registered.
- [x] Validation script registered as `dashboard:v19:validate`.

The purpose is to define the work Hermes can eventually run on a schedule, while making sure scheduled autonomy waits for permission enforcement.

## V20 Secure Tool And Permission Layer

- [x] Create permission policy records for viewer, operator, and admin actions.
- [x] Mark confirm and explicit approval requirements.
- [x] Mark which actions require audit logging.
- [x] Route exposed at `/permission-security`.
- [x] Metadata contract registered.
- [x] Validation script registered as `dashboard:v20:validate`.

The purpose is to make the future Hermes CEO/control-plane layer useful without letting it deploy, change secrets, or run risky automations without the right approval trail.

## V21 TLC Business Operating System

- [x] Create business-unit scorecards for TLC Capital Group OS, Media Business, and Research/Investing.
- [x] Show business health, revenue signal maturity, cost signal maturity, and operating focus.
- [x] Route exposed at `/business-os`.
- [x] Metadata contract registered.
- [x] Validation script registered as `dashboard:v21:validate`.

The purpose is to move from project dashboards to a business operating system view: what is healthy, what is creating evidence, what costs attention, and what needs focus.

## Next Practical Build After V21

- [ ] Add project-owned `/dashboard-snapshot` endpoints to Media Engine and Khashi VC after their migrations resume.
- [ ] Persist decision records and task records in a real store.
- [ ] Add provider execution metrics before enabling automatic model fallback.
- [ ] Add permission middleware before any autonomous operating loop can trigger production-affecting commands.
- [ ] Connect business scorecards to real revenue, cost, publishing, research, and infrastructure signals.
