# UA Flywheel Integration — Plan Index

> **Purpose:** Canonical entry point for the Understand-Anything → Hermes Flywheel integration.
> All details live in `.plans/` and `.beads/`. This file is a navigation pointer.

## Key Documents

| Doc | Path | Purpose |
|---|---|---|
| Incorporation Strategy | `.plans/ua-incorporation-strategy.md` | What to adopt from UA, what to skip, phased roadmap |
| Phase 2 Execution Plan | `.plans/phase-2-flywheel-ua-integration.md` | Full Phase 2 scope, approval gates, verification matrix |
| Project State | `.plans/project-state-ua-flywheel.md` | Current status of every deliverable |
| Review Doc | `understand-anything-to-flywheel-review.md` | Full risk analysis and artifact inventory |

## Phase 1 (Foundation) — ✅ COMPLETE

All deliverables committed in `24356edcd`. 80 tests pass.

## Phase 2 (Orchestration) — ✅ D1-D3 Complete / Reviewed / Pushed

| Bead | File | Status |
|---|---|---|
| D1 — extract_imports.py | `.beads/phase2-d1-extract-imports.md` | Complete; committed `5a39c7fc7`; pushed to `jc-fork` |
| D2 — code-scan SKILL.md | `.beads/phase2-d2-code-scan-skill.md` | Complete; committed `5a39c7fc7`; pushed to `jc-fork` |
| D3 — validation-gate SKILL.md | `.beads/phase2-d3-validation-gate-skill.md` | Complete; reviewer PASS; committed `5a39c7fc7` |
| D4 — code-review integration | `.beads/phase2-d4-review-integration-deferred.md` | Deferred |

> JC approved **D1-D3 only** for autonomous execution; checkpoint committed and pushed as `5a39c7fc7`.
> **D4** remains deferred — requires explicit JC approval. Merge/deploy remain unapproved.

## Execution Model

Beads under `.beads/` are the authoritative execution units. Each bead contains:
- Exact function signatures and schemas
- Test contracts and FIXTURE expectations
- Verification commands
- Allowed/forbidden file lists
- RED/GREEN/FULL evidence requirements
- No commit/push authority for subagents

## Quick Links

- Branch: `docs/ua-flywheel-phase1-phase2-plan`
- Parent commit: `24356edcd` (Phase 1)
- Latest Phase 2 checkpoint: `5a39c7fc7` pushed to `jc-fork/docs/ua-flywheel-phase1-phase2-plan`
- Test bed repos: `cass_memory_system/`, `mission-control/`, hermes-agent itself
