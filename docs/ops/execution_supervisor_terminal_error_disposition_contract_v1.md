# Execution Supervisor Terminal-Error Disposition Contract (v1)

Date: 2026-04-04  
Status: active (canonical A2/A6 fail-closed disposition contract)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`  
Companion docs: `docs/ops/execution_supervisor_artifact_check_contract_v1.md`, `docs/ops/invalid_output_retry_relaunch_contract_v1.md`

## Purpose

Create an explicit, bounded operator disposition path for `FAILED_OTHER` reroute blocks that are currently surfaced as:

- `failure_classification.code = FAILED_OTHER`
- `failure_classification.reason = terminal_error_without_artifact_failure`
- `reroute.status = BLOCKED`
- `reroute.reason = terminal_error_requires_operator_review`

This contract is fail-closed by default: without a valid disposition packet, these tasks remain blocked.

## Canonical disposition artifact

- Path: `state/continuity/latest/execution_supervisor_terminal_error_dispositions_latest.json`
- Schema: `ops/openclaw/architecture/schemas/execution_supervisor_terminal_error_dispositions.schema.json`
- Template: `docs/ops/templates/execution_supervisor_terminal_error_dispositions.template.json`

## Allowed disposition action (v1)

### `acknowledge_and_park`

Use only when operator explicitly decides the current terminal-error blocker should be parked (not relaunched yet).

Required fields per row:
- `task_id`
- `action=acknowledge_and_park`
- `approved_by`
- `approved_at`

Optional guards (recommended):
- `classification` (default `FAILED_OTHER`)
- `failure_reason` (default `terminal_error_without_artifact_failure`)
- `reroute_reason` (default `terminal_error_requires_operator_review`)
- `expires_at` (recommended for bounded parking)

## Runtime application semantics

`continuity_current.sh` consumes this packet fail-closed and only applies disposition when all checks pass:

1. Disposition artifact schema id matches v1.
2. Row action is `acknowledge_and_park`.
3. Approval metadata is present (`approved_by`, parseable `approved_at`).
4. If `expires_at` is present, it is parseable and still in the future.
5. Runtime task row still matches the guarded blocker class/reasons.

When applied:
- task `status` remains `FAILED_OTHER` (failure truth is preserved),
- reroute is converted to non-dispatch posture:
  - `reroute.required=false`
  - `reroute.status=none`
  - `reroute.reason=operator_disposition_acknowledge_and_park`
- applied disposition metadata is attached to the task row (`operator_disposition`),
- reroute pressure counters no longer treat that task as active reroute-required work.

## Hard fail-closed rules

- Missing/invalid schema: no disposition is applied.
- Invalid/expired rows: skipped.
- Guard mismatch against current blocker semantics: skipped.
- No automatic retry/dispatch mutation is introduced by this contract.

## Validation

```bash
pytest -q tests/test_execution_supervisor_task_ledger_runtime.py -k "terminal_error_disposition"
```

Pass criteria:
- valid disposition row suppresses reroute-required pressure for guarded terminal-error blocker,
- expired/invalid disposition rows do not suppress blocker posture.
