# P0 State Machine

This is the frozen transition table for phase 0.

| From | To | Rule ID | Condition | Action | Auto |
|---|---|---|---|---|---:|
| healthy | suspect | `HEALTHY_TO_SUSPECT_2X_INTERVAL_V1` | `now - last_expected_or_terminal_at > 2 x expected_interval` | none | yes |
| healthy | systemic_failure | `HEALTHY_TO_SYSTEMIC_FAILURE_LEDGER_V1` | ledger/runtime/state-store produces a clear infrastructure fault | none | yes |
| suspect | healthy | `SUSPECT_TO_HEALTHY_EVIDENCE_RECOVERED_V1` | required evidence becomes complete and conflict-free | none | yes |
| suspect | transient_failure | `SUSPECT_TO_TRANSIENT_FAILURE_PROVIDER_V1` | terminal failed execution is classified as provider/auth/quota transient | none | yes |
| suspect | stale_running | `SUSPECT_TO_STALE_RUNNING_MAX_RUNTIME_V1` | running or claimed execution exceeds max_runtime and has no terminal evidence | `reset_job` | no |
| suspect | systemic_failure | `SUSPECT_TO_SYSTEMIC_FAILURE_LEDGER_V1` | scheduler, WAL, schema, or ledger evidence is broken | `repair_state_store` | no |
| suspect | quarantined | `SUSPECT_TO_QUARANTINED_PARTIAL_V1` | evidence is partial, conflicting, or otherwise incomplete | `escalate_to_human` | no |
| transient_failure | recoverable | `TRANSIENT_TO_RECOVERABLE_FALLBACK_OK_V1` | fallback exists, policy permits it, and route budget remains | `switch_provider` | no |
| transient_failure | human_required | `TRANSIENT_TO_HUMAN_REQUIRED_POLICY_BLOCK_V1` | policy, provider, auth, or output policy blocks any safe fallback or rerun | `escalate_to_human` | no |
| transient_failure | quarantined | `TRANSIENT_TO_QUARANTINED_BUDGET_EXHAUSTED_V1` | retry or route budget is exhausted or fallback is incompatible | `escalate_to_human` | no |
| stale_running | recoverable | `STALE_RUNNING_TO_RECOVERABLE_RESET_READY_V1` | reset preconditions pass and job is safe to reset | `reset_job` | no |
| stale_running | human_required | `STALE_RUNNING_TO_HUMAN_REQUIRED_NONIDEMPOTENT_V1` | job is non-idempotent or side effects are unknown | `escalate_to_human` | no |
| systemic_failure | repair_in_progress | `SYSTEMIC_TO_REPAIR_IN_PROGRESS_LEASE_V1` | repair action is explicitly approved and lease is acquired | `repair_state_store` | no |
| recoverable | repair_in_progress | `RECOVERABLE_TO_REPAIR_IN_PROGRESS_LEASE_V1` | recovery lease is acquired for the incident/job pair | `switch_provider` | no |
| repair_in_progress | recovered | `REPAIR_IN_PROGRESS_TO_RECOVERED_READBACK_V1` | post-action evidence is complete and read-back matches | none | no |
| repair_in_progress | quarantined | `REPAIR_IN_PROGRESS_TO_QUARANTINED_FAILED_V1` | `action_failed` or `verification_failed` | `escalate_to_human` | no |
| recovered | healthy | `RECOVERED_TO_HEALTHY_NEXT_TERMINAL_V1` | next terminal execution evidence is normal | none | yes |
| recovered | suspect | `RECOVERED_TO_SUSPECT_OBSERVATION_WINDOW_V1` | observation window sees renewed freshness drift | none | yes |
| quarantined | suspect | `QUARANTINED_TO_SUSPECT_EVIDENCE_COMPLETE_V1` | missing or conflicting evidence is resolved | none | yes |
| quarantined | human_required | `QUARANTINED_TO_HUMAN_REQUIRED_ESCALATION_V1` | escalation SLA expires | `escalate_to_human` | yes |
| human_required | suspect | `HUMAN_REQUIRED_TO_SUSPECT_OVERRIDE_V1` | approved override lands with fresh evidence | none | no |
| human_required | quarantined | `HUMAN_REQUIRED_TO_QUARANTINED_DENY_V1` | manual deny or override expiry | none | yes |

Frozen rule IDs are also captured in `state-machine.json`.
