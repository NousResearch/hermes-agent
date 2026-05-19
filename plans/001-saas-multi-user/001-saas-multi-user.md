# Multi-User SaaS Hermes — Master Plan

## Goal

Evolve Hermes from a single-user local agent into a multi-tenant SaaS platform supporting teams on Slack (and other platforms). Multiple users per workspace get isolated personal memory and skills, shared team memory and skills, and conflict-safe agent self-modification — all backed by cloud-native WAL storage.

## Context

Hermes already knows everything it needs to be multi-tenant:
- **`platform`** — already passed in every gateway turn (e.g. "slack")
- **`team_id`** — Slack workspace ID is in every Slack event
- **`user_id`** — Slack user ID is in every Slack event
- **`channel_id`** / **`thread_id`** — already tracked for conversation continuity

The gap is: none of these are formalized into an identity model, and all state (skills, memory, sessions) is stored locally on a single machine.

## The Three-Layer Scoping Model

Every resource (skill, Atlas knowledge, conversation) lives in exactly one scope and is **readable upward** but **writable only at its own level**:

```
┌──────────────────────────────────────────────┐
│  GLOBAL  (platform defaults)                 │
│  read-only to all agents                     │
├──────────────────────────────────────────────┤
│  TEAM  (Slack workspace)                     │
│  writable by team members; shared            │
├──────────────────────────────────────────────┤
│  PERSONAL  (individual user)                 │  ← most specific, wins
│  private; writable only by that user's agent │
└──────────────────────────────────────────────┘
```

Resolution order: personal overrides team overrides global (CSS specificity model).

## Phases

| # | Phase | Type | Dependencies |
|---|-------|------|-------------|
| 0 | Identity & Tenant Model | Foundation | None |
| A | Scoped Skills | Feature | Phase 0 | **Superseded by Plan 003** — Plan 003 is the full implementation |
| B | Scoped Atlas Memory | Feature | Phase 0 |
| C | Conflict-Safe Self-Modification | Safety | Phase A |
| D | Cloud Storage Backend | Infrastructure | Phase 0 |
| E | Stateless Deployment | Infrastructure | Phase D |

## Dependency Graph

```
Phase 0 (Identity)
  ├─> Phase A (Skills) ──> Phase C (Locks)
  ├─> Phase B (Memory)
  └─> Phase D (Storage) ──> Phase E (Deployment)
```

## Execution Rules

- Complete each phase fully before the next (phases within D/E can partially parallelize)
- Each phase produces a commit + push
- `HERMES_MODE=saas` env flag gates all cloud paths — local dev stays unchanged
- No agent turn may write to `global` scope — ever

## Subplan Files

- `plans/saas-multi-user/phase-0-identity.md`
- `plans/saas-multi-user/phase-A-scoped-skills.md`
- `plans/saas-multi-user/phase-B-scoped-memory.md`
- `plans/saas-multi-user/phase-C-conflict-locks.md`
- `plans/saas-multi-user/phase-D-cloud-storage.md`
- `plans/saas-multi-user/phase-E-deployment.md`
- `plans/saas-multi-user/PROGRESS.md`

## Budget Estimate

| Component | MVP (~5 tenants) | Scale (~100 teams) |
|---|---|---|
| Neon PostgreSQL | $0 (free tier) | ~$19/mo |
| S3 (skills) | ~$0.50/mo | ~$5/mo |
| ECS Fargate (2 tasks) | ~$30/mo | ~$150/mo |
| Upstash Redis | $0 (free tier) | ~$10/mo |
| DynamoDB (skill locks) | ~$0 | ~$1/mo |
| **Total** | **~$31/mo** | **~$185/mo** |
