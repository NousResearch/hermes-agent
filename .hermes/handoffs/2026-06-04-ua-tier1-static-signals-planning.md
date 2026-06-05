# Handoff: UA Tier 1 Static Signals Planning Package

## Context

- User approved a planning package only for the UA Tier 1 static-signals layer.
- Approved scope quote:

```text
[JC] Approve planning package for UA Tier 1 static-signals layer only:
create/update the Tier 1 plan package and beads under .hermes/plans, .beads, and .hermes/handoffs;
do not execute implementation beads yet;
do not modify run_ua.py or production code in this approval;
do not commit or push without a separate explicit approval.
```

- Expected artifacts:
  - `.hermes/plans/ua-tier1-static-signals-plan.md`
  - `.beads/ua-tier1-001-static-signals-schema.md`
  - `.beads/ua-tier1-002-supabase-migration-markers.md`
  - `.beads/ua-tier1-003-edge-package-config-markers.md`
  - `.beads/ua-tier1-004-entrypoint-hotspot-refinement.md`
  - `.beads/ua-tier1-005-run-ua-report-integration.md`
  - `.hermes/handoffs/2026-06-04-ua-tier1-static-signals-planning.md`

## Work Completed

- Created a Tier 1 plan package describing architecture, boundaries, inclusion priorities, bead order, verification, and future execution routing.
- Created five planned-not-approved implementation beads.
- Preserved the approval quote verbatim in the plan and each bead.
- No implementation bead was executed.
- No source, test, dependency, production, runtime, `run_ua.py`, or report code was intentionally modified.
- No commit or push was performed.

## Verification

Hermes-owned planning verification completed:

```text
PLANNING_VERIFIER_PASS expected_files=7 beads=5 quote_preserved=true headings=true boundaries=true
DIFF_CHECK_PASS
```

Scoped changed files observed after verification:

```text
?? .beads/ua-tier1-001-static-signals-schema.md
?? .beads/ua-tier1-002-supabase-migration-markers.md
?? .beads/ua-tier1-003-edge-package-config-markers.md
?? .beads/ua-tier1-004-entrypoint-hotspot-refinement.md
?? .beads/ua-tier1-005-run-ua-report-integration.md
```

Hidden `.hermes/plans` and `.hermes/handoffs` files are ignored by git in this repo, so Hermes must verify them explicitly by path before any future checkpoint.

## Subagent Reliability

- Exit/failure class: completed reviewer readiness pass.
- Expected vs actual artifacts: match after Hermes verification.
- Reviewer verdict: PASS; no blockers. Optional notes only: fixtures/module absent by design for planning-only package, `.hermes/PROJECT_STATE.md` and root `PLAN.md` intentionally untouched, and T1-004 uses acceptable `Modify likely` phrasing.
- Recovery path: patch planning docs only if future execution approval review finds a defect.

## Issues / Caveats

- `.hermes/PROJECT_STATE.md` was not updated in this package to preserve the exact approved path set unless JC separately approves ledger sync.
- Root `PLAN.md` was not updated for the same reason.
- This package is intentionally not an implementation approval.

## Next Recommended Action

Run reviewer readiness on the planning package, patch any planning defects, then ask JC for separate execution approval for Tier 1 beads if desired.
