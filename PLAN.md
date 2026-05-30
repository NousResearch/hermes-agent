# UA Flywheel Integration — Plan Index

> **Purpose:** Canonical entry point for the Understand-Anything → Hermes Flywheel integration.
> All details live in `.plans/` and `.beads/`. This file is a navigation pointer.
> **Last updated:** 2026-05-30 (Phase 3 D1-D3 merged to local main at `0133a0a4b` via PR #6; CI green)

## Key Documents

| Doc | Path | Purpose |
|---|---|---|
| Incorporation Strategy | `.plans/ua-incorporation-strategy.md` | What to adopt from UA, what to skip, phased roadmap |
| Phase 2 Execution Plan | `.plans/phase-2-flywheel-ua-integration.md` | Full Phase 2 scope, approval gates, verification matrix |
| Phase 3 Execution Plan | `.plans/phase-3-incremental-analysis.md` | Full Phase 3 approved D1-D3 scope, gates, verification matrix |
| Project State | `.plans/project-state-ua-flywheel.md` | Current status of every deliverable |
| Review Doc | `understand-anything-to-flywheel-review.md` | Full risk analysis and artifact inventory |

## Phase 1 (Foundation) — ✅ COMPLETE

All deliverables committed in `24356edcd`. 80 tests pass.

## Phase 2 (Orchestration) — ✅ D1-D3 Evaluated, Merged to `jc-fork/main`

| Bead | File | Status |
|---|---|---|
| D1 — extract_imports.py | `.beads/phase2-d1-extract-imports.md` | Complete; merged to `jc-fork/main` at HEAD `24e9fe65a` |
| D2 — code-scan SKILL.md | `.beads/phase2-d2-code-scan-skill.md` | Complete; merged to `jc-fork/main` at HEAD `24e9fe65a` |
| D3 — validation-gate SKILL.md | `.beads/phase2-d3-validation-gate-skill.md` | Complete; merged to `jc-fork/main` at HEAD `24e9fe65a` |
| D4 — code-review integration | `.beads/phase2-d4-review-integration-deferred.md` | Deferred |

> JC approved **D1-D3 only** for autonomous execution. Phase 2 D1-D3 evaluation: **11/11 PASS**.
> Merged to `jc-fork/main` — final HEAD `24e9fe65a` (`test(run-agent): isolate proxy tests from lazy dependency installs`).
> Post-push CI: Tests ✅, Lint ✅, Nix ✅.
> **D4** remains deferred — requires explicit JC approval. Phase 3 D1-D3 approved for autonomous execution.

## Phase 3 (Incremental Analysis) — ✅ D1-D3 MERGED TO LOCAL MAIN

| Bead | File | Status |
|---|---|---|
| D1 — fingerprint model | `.beads/phase3-d1-fingerprint-model.md` | ✅ Merged at `0133a0a4b` via PR #6; 61 tests pass; reviewer PASS |
| D2 — incremental scan | `.beads/phase3-d2-incremental-scan.md` | ✅ Merged at `0133a0a4b` via PR #6; 40 scan tests + 61 D1 regression tests pass; reviewer PASS |
| D3 — graph assembly | `.beads/phase3-d3-assemble-graph.md` | ✅ Merged at `0133a0a4b` via PR #6; 64 D3 tests + 132 regression tests pass; reviewer PASS |
| D4 — skill integration | `.beads/phase3-d4-skill-integration-deferred.md` | Deferred by default |

> **Phase 3 D1-D3 merged to local main at `0133a0a4b`** (PR #6 squash merge). CI on main: Tests ✅, Lint ✅, Nix ✅. D4 remains deferred by default.
> Approval doc: `.plans/phase-3-incremental-analysis.md`.

## Execution Model

Beads under `.beads/` are the authoritative execution units. Each bead contains:
- Exact function signatures and schemas
- Test contracts and FIXTURE expectations
- Verification commands
- Allowed/forbidden file lists
- RED/GREEN/FULL evidence requirements
- No commit/push authority for subagents

## Quick Links

- Branch: `main` (local main, 13 commits ahead of origin/main)
- HEAD (local main): `0133a0a4b` (`feat(code-scan): add Phase 3 incremental analysis (#6)`)
- origin/main: `5f84c9144`
- Phase 2 merge commit: `7d7785dc4`; Phase 3 merge: `0133a0a4b` (PR #6 squash)
- Phase 1 commit: `24356edcd`
- Evaluation evidence: `/tmp/phase2-d1-d3-eval-corrected-latest.log`
- Test bed repos: `cass_memory_system/`, `mission-control/`, hermes-agent itself
