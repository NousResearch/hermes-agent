# Verify-Before-Resume Gate Checklist (v1)

Date: 2026-03-19  
Status: active  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose
Prevent false-green resume decisions by requiring fresh truth, verification, and dependency health checks before mutation-heavy or roadmap work continues.

## When to run
Run before:
- switching from blocker handling back to throughput work,
- resuming after stale handover/continuity conditions,
- approving nontrivial mutation waves after incident/degraded periods.

## Required evidence inputs
- Latest continuity/ground-truth snapshot path + timestamp.
- Latest gate/verification outputs path + timestamp.
- Active blocker registry state.
- Dependency health signals (connectors/watchdogs/scheduler integrity where relevant).

## Checklist
Mark each item pass/fail with evidence pointer.

1. **Fresh truth check**
   - [ ] Ground truth snapshot is present and fresh for current operating window.
2. **Verification check**
   - [ ] Required validation gates are executed and not stale.
3. **Dependency health check**
   - [ ] No critical dependency is in unknown/failed state.
4. **Continuity coherence check**
   - [ ] Continuity artifacts are coherent (not contradicted by newer runtime truth).
5. **Blocker state check**
   - [ ] No unresolved blocker class forbids mutation resume.
6. **Evidence quality check**
   - [ ] Claims are evidence-backed (no narrative-only “healthy” assertion).

## Gate outcome
Set one explicit result:
- `allowed`:
  - All checks pass with fresh evidence.
- `caution`:
  - Minor staleness/risk exists; proceed only with reduced scope, tighter review, and explicit risk callout.
- `forbidden`:
  - Any critical check fails or evidence is stale/missing; no mutation until corrected.

## Required output record (minimum)
- `gate_result`: `allowed | caution | forbidden`
- `evaluated_at`
- `evidence_refs[]`
- `failed_checks[]`
- `constraints_if_caution[]`
- `next_recheck_at`
