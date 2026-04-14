# Blocker Burndown Control Loop (v1)

Date: 2026-03-14  
Status: active (subordinate doctrine module)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose
This runbook exists to prevent stale-wave drift, radio silence, and false progress framing during readiness/blocker-clearing work.

## Applicability
Use whenever session mode is `BLOCKER_BURNDOWN`.

## Mandatory wave fields
Every wave must record:
- `wave_id`
- `objective`
- `spawned_at`
- `review_at = spawned_at + 25m`
- `stale_at = spawned_at + 45m`
- `lanes`
- `expected_artifacts`
- `kill_conditions`

## Trust model
Never use `running` as a progress signal by itself.

Classify each active lane by **last meaningful child signal age**:
- `green`: <10m
- `amber`: 10–30m
- `red`: >30m
- `stale`: >45m without meaningful completion

## Mandatory actions
### At review_at (~25m)
- inspect fresh evidence
- decide: continue / narrow / kill
- report risk state if blocker work remains active

### At stale_at (~45m)
Default action:
1. report stale status,
2. kill stale lane(s),
3. salvage useful edits/findings,
4. relaunch narrower only if next slice is clearly better-bounded.

## Silence rule
- Max silent interval during blocker work: **30m**.
- If stale-risk or stale is reached, update proactively; do not wait for the operator to ask.

## Status update schema
Every blocker-work update should include:
- `claim`
- `evidence_ref`
- `evidence_timestamp`
- `last_signal_age`
- `risk_state`
- `next_control_point`
- `what_changed`
- `what_remains`

## Anti-patterns
- saying “still running” without fresh evidence
- treating absence of failure as proof of progress
- letting documentation changes substitute for live supervision
- using broad subagent waves inside degraded continuity state without tight manual oversight

## Recommended operating pattern
1. Pin canonical blocker state.
2. Launch at most 2 bounded lanes.
3. Track review/stale times explicitly.
4. Verify claimed fixes independently.
5. Update canonical blocker registry only after proof.
6. Resume throughput mode only after blocker class changes.
