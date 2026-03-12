# New Features — Subagent Execution Plan (2026-03-12)

## Initiative
Finalize and ship the current gateway/Discord/runtime-control feature tranche with manager-mode process enforcement.

## Scope
- Runtime override and routing controls in `gateway/run.py`
- Discord attachment + slash command parity updates in `gateway/platforms/discord.py`
- Model parser hardening in `hermes_cli/models.py`
- Coverage in gateway/model validation tests
- Manager-mode docs and runbooks for repeatable subagent delivery

## Constraints
- Keep branch hygiene on `feat/gateway-phase5-hardening`
- No scope creep beyond touched modules and operator docs
- Preserve existing passing behavior in gateway suite

## Acceptance Criteria
1. A taskized plan and packets exist under `docs/plans/` and are executable.
2. Feature code is reviewed by subagents in two stages:
   - Spec compliance: PASS
   - Code quality: APPROVED (or fixed then approved)
3. Verification evidence is fresh and attached:
   - targeted tests for touched modules
   - syntax/compile check for touched Python files
4. Hygiene policy is validated:
   - branch naming conforms
   - no backup/temp/secrets files
   - commit/PR guidance documented for merge
5. Final output includes merge recommendation + risk/rollback/follow-ups.

## Task Breakdown
- TASK-1: Plan + packet finalization (manager prep)
- TASK-2: Spec compliance review of current feature set
- TASK-3: Code quality review and remediation
- TASK-4: Verification gate (tests + compile + hygiene)
- TASK-5: Merge-ready handoff summary

## Execution Notes
- Sequential execution for overlapping files (`gateway/run.py` intersects routing and command surfaces).
- Parallelization only allowed for independent docs-only checks.

## Done Definition
All five tasks are completed, all manager review gates pass, and delivery can proceed without additional blocking engineering work.
