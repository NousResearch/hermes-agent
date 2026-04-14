# Invalid Output Retry/Relaunch Contract (v1)

Date: 2026-03-28  
Status: active (canonical A2/A6 execution-supervisor contract)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`  
Companion docs: `docs/ops/execution_meaningful_event_reporting_doctrine_v1.md`, `docs/ops/subagent_slot_fill_protocol_v1.md`

## Purpose

Canonicalize deterministic handling for invalid/junk worker output so the system never depends on manual relaunch judgment.

This contract encodes the required action path for `INVALID_OUTPUT` outcomes in queue/executor supervision.

## Contract scope

Applies when a child/executor branch produces unusable output and the supervisor can classify outcome as:
- `INVALID_OUTPUT` (junk/unusable output), or
- `FAILED` with retryable posture that follows the same relaunch path.

## Deterministic transition contract

When invalid output is detected:

1. Emit `SUBAGENT_FINISHED` with `outcome_class=INVALID_OUTPUT`.
2. Emit `WORKER_FAILED_OR_JUNK` in the same control turn.
3. Decision is fail-closed and explicit:
   - `decision_taken=relaunch` when dispatch is launchable,
   - otherwise `decision_taken=blocked` with explicit blocker reason.
4. If relaunch is applied, emit `QUEUE_RELAUNCHED` (and `EXECUTOR_IDLE_TO_RELAUNCHED` when applicable).

Hard rule: explanation cannot replace the action decision for invalid-output transitions.

Fail-closed extension (EXB-02 broader closure): if queue transition reason text is inconclusive,
execution-supervisor failure classification (`FAILED_NO_ARTIFACT`, empty-content `FAILED_NO_ARTIFACT`, or `FAILED_PROVIDER_QUOTA`) must still drive
`SUBAGENT_FINISHED.outcome_class=INVALID_OUTPUT` + `WORKER_FAILED_OR_JUNK` so empty/junk/quota-bound completions cannot silently pass as success.

Landed-row cap guard: when execution-supervisor landed-task sampling is capped, the latest queue transition task
must be prioritized into sampled landed candidates (when it has expected artifacts or terminal failure hints), so
fresh empty/invalid completions are not dropped from failure accounting by historical DONE-row ordering.

## Required packet fields (invalid-output path)

In addition to the general meaningful-event packet minimums:
- `outcome_class` on `SUBAGENT_FINISHED` (`INVALID_OUTPUT`)
- `decision_taken` (`relaunch` or `blocked`)
- `next_action` (`relaunch_branch` or `mark_blocked`)
- `decision_reason` with failure class (`failure_class:invalid_output` or equivalent)
- `queue_truth_projection_parity.status` + `next_candidate` on queue relaunch transitions

## Operator-surface contract

Operator truth must expose this transition family through canonical surfaces:
- `state/continuity/latest/execution_meaningful_event_reporting_latest.json`
- `state/continuity/latest/execution_meaningful_event_reporting_status_latest.json`
- `state/continuity/latest/operator_mission_control.json`
  - `meaningful_event_reporting`
  - `meaningful_event_reporting_contract`

Fail-close condition:
- required invalid-output event packets expected but missing/stale/mismatched status projection => mission-control contract must not report silent `ok`.

## XR-006 closeout evidence packet (2026-03-28)

- State-transition test output:
  - `state/continuity/latest/xr_006_invalid_output_retry_relaunch_state_transition_test_output_2026-03-28.json`
- Supervisor harness evidence:
  - `state/continuity/latest/xr_006_invalid_output_retry_relaunch_supervisor_harness_evidence_2026-03-28.json`
- Incident artifact sample:
  - `state/continuity/latest/xr_006_invalid_output_retry_relaunch_incident_artifact_sample_2026-03-28.json`

Validation command:

```bash
pytest -q tests/test_execution_meaningful_event_reporting_surface.py -k "invalid_output_relaunch_contract"
```

Pass criteria (fail-closed):
- `SUBAGENT_FINISHED` includes `outcome_class=INVALID_OUTPUT`.
- `WORKER_FAILED_OR_JUNK` carries deterministic `relaunch` or explicit `blocked`.
- Relaunch path emits `QUEUE_RELAUNCHED` (plus idle->relaunched event when applicable).

## Canonical references

- `docs/ops/execution_meaningful_event_reporting_doctrine_v1.md`
- `docs/ops/execution_meaningful_event_reporting_checklist_v1.md`
- `docs/ops/subagent_slot_fill_protocol_v1.md`
- `docs/ops/core_roadmap_queue_layer_doctrine_v1.md`
