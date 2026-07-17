# Hermes Autonomy Readiness Build Plan V22-V30

This document extends the V15-V21 operating-system layer into the next practical control-plane stretch: live truth, durable memory, enforcement, model-cost governance, loop execution readiness, cross-business command, workbench supervision, quality gates, and autonomy launch readiness.

Status legend:

- `[x]` Trackable infrastructure complete in Hermes.
- `[~]` Runtime integration pending.
- `[!]` Must stay gated until permissions, audits, and rollback controls are live.

## Status Summary

| Version | Capability | Status | Built Now | Remaining Runtime Hook |
| --- | --- | --- | --- | --- |
| V22 | Live project snapshot contracts | `[x]` | `/project-snapshots`, snapshot contract model, route, metadata, validation | Project-owned production endpoints |
| V23 | Durable memory and decision store | `[x]` | `/durable-memory`, memory readiness model, route, metadata, validation | DB-backed task/decision/context storage |
| V24 | Permission enforcement runtime | `[x]` + `[!]` | `/permission-runtime`, enforcement gates, approval/audit model | Runtime middleware and audit persistence |
| V25 | Model router and cost governor | `[x]` | `/cost-governor`, provider policy, cost gate model | Provider adapters, outcome scoring, live budget enforcement |
| V26 | Operating loop runner | `[x]` + `[!]` | `/loop-runner`, loop runner readiness, dry-run contract | Scheduler after permission runtime and audit store |
| V27 | Cross-business command center | `[x]` | `/business-command`, business rollup model | Live revenue/cost/project signal feeds |
| V28 | Agent workbench | `[x]` + `[!]` | `/agent-workbench`, plan/approval/artifact/report model | Live execution bridge after V24/V30 gates |
| V29 | Evaluation and quality gates | `[x]` | `/evaluation-gates`, gate family model | Provider outcome history and promotion enforcement |
| V30 | Production autonomy readiness | `[x]` + `[!]` | `/autonomy-readiness`, autonomy levels, kill-switch/budget-breaker model | Runtime kill switches, incident store, autonomy policy enforcement |

## V22 Live Project Snapshot Contracts

- [x] Define the six signal families Hermes needs from projects: health, cost, capacity, queue, actions, and risk.
- [x] Add `/project-snapshots` route.
- [x] Add metadata and validation script.
- [x] Mark project-owned endpoint implementation as gated.

Acceptance criteria:

- [x] Hermes can name exactly what every project must expose before it becomes a live operating source.

## V23 Durable Memory And Decision Store

- [x] Define task, decision, review, and context-link readiness.
- [x] Add `/durable-memory` route.
- [x] Add metadata and validation script.
- [x] Mark database-backed persistence as gated.

Acceptance criteria:

- [x] Hermes has a trackable memory model before any runtime store is introduced.

## V24 Permission Enforcement Runtime

- [x] Define permission checks, approval gates, and audit requirements.
- [x] Add `/permission-runtime` route.
- [x] Add metadata and validation script.
- [x] Keep deploy, secret, and autonomous actions explicitly gated.

Acceptance criteria:

- [x] Hermes can show which commands are safe, which need confirmation, and which require explicit approval.

## V25 Model Router And Cost Governor

- [x] Define local-first, cheap-API-first, and premium-approval routing modes.
- [x] Add `/cost-governor` route.
- [x] Add metadata and validation script.
- [x] Mark live provider quality/cost scoring as gated.

Acceptance criteria:

- [x] Hermes can explain why it would use local Codex, a cheaper provider, or a premium fallback.

## V26 Operating Loop Runner

- [x] Define manual loop run, dry-run output, and scheduler readiness.
- [x] Add `/loop-runner` route.
- [x] Add metadata and validation script.
- [x] Keep production scheduling locked until V24 and V30 runtime gates are live.

Acceptance criteria:

- [x] Hermes has a safe loop-runner model before recurring autonomy is enabled.

## V27 Cross-Business Command Center

- [x] Define project-to-business mapping, attention rollups, and finance feed readiness.
- [x] Add `/business-command` route.
- [x] Add metadata and validation script.
- [x] Mark live revenue and cost feeds as gated.

Acceptance criteria:

- [x] Hermes can present TLC Capital Group OS as a business control plane, not just a project dashboard list.

## V28 Agent Workbench

- [x] Define plan, approve, execute, evidence, and report workflow.
- [x] Add `/agent-workbench` route.
- [x] Add metadata and validation script.
- [x] Keep live execution gated behind permissions and audit.

Acceptance criteria:

- [x] Hermes has a supervised workbench pattern before it gets broader action authority.

## V29 Evaluation And Quality Gates

- [x] Define code, design, dashboard, model, and production gate families.
- [x] Add `/evaluation-gates` route.
- [x] Add metadata and validation script.
- [x] Mark provider outcome scoring and promotion blocking as gated.

Acceptance criteria:

- [x] Hermes has a measurable quality gate model before cheaper models are trusted with production work.

## V30 Production Autonomy Readiness

- [x] Define autonomy levels, kill switches, budget breakers, and incident review expectations.
- [x] Add `/autonomy-readiness` route.
- [x] Add metadata and validation script.
- [x] Keep real autonomy progression gated behind runtime enforcement, audit, rollback, and budget breakers.

Acceptance criteria:

- [x] Hermes can describe what must be true before any project moves from assisted operation to limited autonomy.

## Post-V30 Build Boundary

- [ ] Implement production project snapshot endpoints.
- [ ] Choose and migrate durable memory storage.
- [ ] Enforce permissions in the runtime command path.
- [ ] Add provider adapters and task outcome scoring.
- [ ] Enable loop dry-runs before scheduled runs.
- [ ] Add kill-switch and budget-breaker runtime controls.
