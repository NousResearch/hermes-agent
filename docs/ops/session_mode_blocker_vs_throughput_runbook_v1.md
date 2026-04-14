# Session Mode Runbook: BLOCKER_BURNDOWN vs THROUGHPUT (v1)

Date: 2026-03-19  
Status: active  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose
Define deterministic entry/exit criteria for session mode so throughput does not start on stale or false-green state.

## Modes
- `BLOCKER_BURNDOWN`: blocker/readiness recovery mode (default when blockers exist).
- `THROUGHPUT`: controlled parallel execution mode when truth is fresh and blockers are clear.

## Entry criteria
### Enter BLOCKER_BURNDOWN if any are true
- Active blocker class exists.
- Truth/verification freshness is stale or unknown.
- Critical dependency health is degraded/unknown.
- Recent wave crossed stale-risk without closure proof.

### Enter THROUGHPUT only if all are true
- Verify-before-resume gate result is `allowed`.
- No active blocker class forbids execution.
- Current wave state is coherent and evidence-backed.

## Operating constraints
### BLOCKER_BURNDOWN constraints
- Max active subagents: 2 (unless explicitly justified).
- Use control-loop timings from `docs/ops/blocker_burndown_control_loop_v1.md`.
- Prefer serial lane ordering: diagnose → implement → verify.

### THROUGHPUT constraints
- Parallelism allowed only for non-overlapping scopes.
- Each slice must declare objective, boundary, artifacts, verification, and kill condition.
- Keep validator independence for higher-risk changes.

## Mode switch procedure
1. Record current mode + rationale.
2. Run `docs/ops/verify_before_resume_gate_checklist_v1.md`.
3. Decide target mode from gate result:
   - `forbidden` -> remain/return to `BLOCKER_BURNDOWN`.
   - `caution` -> remain in `BLOCKER_BURNDOWN` or run tightly constrained mini-throughput.
   - `allowed` -> `THROUGHPUT` eligible.
4. Announce mode decision with evidence refs and next control point.

## Required mode decision fields
- `current_mode`
- `target_mode`
- `decision`
- `rationale`
- `gate_result`
- `evidence_refs[]`
- `effective_at`
- `next_review_at`
