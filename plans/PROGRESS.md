# Plans — Hermes Agent Global Progress

| Plan | Title | Status | Run after |
|------|-------|--------|-----------|
| 001 | Multi-User SaaS Hermes — Identity, Scoped Skills & Memory, Cloud Storage | **IN PROGRESS** (Phases 0/A/B Complete 2026-05-20; Neon live + RLS verified). 6 phases (0, A, B, C, D, E). C/D/E pending AWS Fargate + S3 follow-ups. | — |
| 002 | Hermes Self-Organization — Directory Structure, MCP Pool, Runtime Isolation | DRAFT (2026-05-18) — 4 phases (A, B, C, D). Defines the canonical directory layout, pooled MCP gateway, runtime workspace isolation, credential injection. Cloud-routable via env vars. | 001-0 |
| 003 | Skills Service — Scoped Registry, MCP Surface, Git-Backed Collaboration | APPROVED (2026-05-18) — 6 phases (A–F). Standalone repo (`hermes-skills-service/`), port 8001, `promote_requires_pr: false`, `blake-cowork-plugins` as team scope. Extracts skills into an external service (mirroring Atlas) with three-scope CSS resolution (personal → team → global), Git-backed registries, promotion CLI, advisory write locks, S3 backend for SaaS mode. Supersedes Plan 001-A. | 001-0 |
| 004 | Self-Improvement Service — Hermes↔Skills↔Atlas Feedback Loop | STUB (2026-05-18) — Stub plan only. No phases yet. Defines the boundary between Hermes agent behavior scoring, Skills Service promotion, and Atlas memory. Port 8002 (TBD). | 003 + Atlas 012 |

## Active plans

- **001 — Multi-User SaaS** (IN PROGRESS 2026-05-20): Phases 0/A/B all Complete. Neon DB live (project `hermes-saas`, us-east-1); 4 tables + 3 RLS policies verified end-to-end (cross-tenant isolation confirmed; unset GUC raises). `hermes_app` role + DSN in AWS Secrets Manager. Plans 002, 003 fully unblocked. Phases C/D/E pending the AWS Fargate + S3 follow-ups.
- **002 — Hermes Self-Organization** (2026-05-18): Spec complete at `~/.hermes/STRUCTURE.md` (v2.0). Plan lives at `plans/002-hermes-self-organization/`. Not started — awaiting Plan 001-0 (HermesIdentity dataclass).
- **003 — Skills Service** (APPROVED 2026-05-18): All decisions locked. Awaiting Plan 001-0 (for phases B–F); phase A (config block) and phase E (blake-cowork-plugins migration) can start independently.
- **004 — Self-Improvement Service** (STUB 2026-05-18): Placeholder only. Real spec TBD after Skills Service (003) + Atlas 012 (Hermes↔Atlas connector) complete.

## Cross-repo consistency pass

Completed + corrected 2026-05-18. See `plans/CONSISTENCY-PASS.md` for full inventory and findings across all four repos (hermes-agent, army-of-one, rooben-pro, agentic-hub). Key findings:
- agentic-hub Plans 001–008 are ALL COMPLETE. Stack is healthy (1 container: atlas only).
- agentic-hub/rooben-planning/ IS the Rooben planning core, vendored in-process (Plan 005-F). The `/refine` + Specification + LLMPlanner pipeline IS available to Hermes via `_dispatch_to_rooben()`.
- Atlas Plan 014 is IN PROGRESS (A–F done; G–I remain). Does not block Hermes execution.
- Hermes 001-0 (HermesIdentity) is the single critical-path gating item for all Hermes SaaS plans.
- Port assignments confirmed: Atlas=8000, Skills=8001, Self-Improvement=8002 (TBD).
- Format fixes applied: Plan 001 renamed to `001-saas-multi-user/phases/`; Atlas 010/011/012 `phases/` dirs created.

## Completed plans (archived)

_(none yet)_
