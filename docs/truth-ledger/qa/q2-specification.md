# Q2 — Specification/compliance review (ledger core)

Date: 2026-07-19
Task: t_0974915d
Reviewer: automation-operator (fresh worker)
Plan: `/Users/hermes/.hermes/hermes-agent/.hermes/plans/2026-07-17_143520-truth-ledger-option-2.md`
Workspace: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`
Commit under review: `48ee34c2bf448a480ef499bb8d24d184d0fcc198`

## Scope

Independent Q2 compliance review for T5-T8 against Q1/plan contracts. No fixes in this task.

Checked domains:
- T5 schema rejection and semantic contract coverage
- T6 deterministic identity/idempotency/reconciliation behavior
- T7 admission/redaction identity gating behavior
- T8 append-only persistence, concurrency, replay/projection, and latency

## Independent rerun evidence

### Canonical focused suite
Command:
- `scripts/run_tests.sh tests/plugins/truth_ledger/test_contracts_and_schemas.py tests/plugins/truth_ledger/test_spool.py tests/plugins/truth_ledger/test_ledger.py tests/plugins/truth_ledger/test_projection.py tests/plugins/truth_ledger/test_concurrency.py tests/plugins/test_truth_ledger_reconciliation.py tests/plugins/test_truth_ledger_admission_redaction.py tests/agent/test_turn_finalizer_post_llm_call_metadata.py tests/agent/test_turn_finalizer_interrupt_alternation.py tests/agent/test_turn_finalizer_final_response_persistence.py tests/agent/test_turn_finalizer_cleanup_guard.py -q`

Result:
- 11 files discovered
- 61 passed, 0 failed

### Independent malformed/idempotency/concurrency/latency probes
Ad-hoc probe runner executed from shared worktree (`python .tmp_q2_probe.py`) produced:

```json
{
  "schema_rejection": {"pass": true, "error_type": "ValueError"},
  "idempotency": {
    "first_decision": "append",
    "second_decision": "duplicate",
    "same_event_id": true,
    "reason": "idempotent_replay"
  },
  "concurrency": {
    "all_exit_zero": true,
    "race_indexed_count": 1,
    "line_count": 161,
    "expected_line_count": 161
  },
  "latency_ms": {"n": 250, "p50": 0.46, "p95": 0.76, "p99": 1.495, "max": 1.755}
}
```

Interpretation:
- malformed schema payloads fail-closed
- duplicate callback replay is idempotent
- multi-process race on shared event key indexes exactly once
- enqueue path stays well below 10ms p95 target

## Compliance findings

### Summary decision
PASS.

No critical or important compliance gaps were reproduced in T5-T8 for this review scope.

### Findings table

| ID | Severity | Component | Evidence location | Reproduction | Expected behavior | Actual behavior | Owner |
|---|---|---|---|---|---|---|---|
| Q2-F1 | info | Schema validation rejects malformed operations | `plugins/truth-ledger/schemas.py:53-60`; `tests/plugins/truth_ledger/test_contracts_and_schemas.py:146-159` | Submit `fact-candidates.v1` with `operation=delete` | Validation error (fail-closed) | `ValueError` raised with schema path context | truth-ledger plugin |
| Q2-F2 | info | Reconciliation idempotency for duplicate callbacks | `plugins/truth-ledger/reconciliation.py:142-153`; `tests/plugins/test_truth_ledger_reconciliation.py:70-90` | Replay same observation/turn with same fact value | No duplicate append; return existing event identity | `decision=duplicate`, same `event_id`, reason `idempotent_replay` | truth-ledger plugin |
| Q2-F3 | info | Append concurrency and race-key dedupe | `plugins/truth-ledger/ledger.py:108-143`; `tests/plugins/truth_ledger/test_concurrency.py:66-90` | 4-process append fanout + shared race key | Non-corrupt ledger and exactly one winner for shared key | `race_indexed_count=1`, expected line count matched | truth-ledger plugin |
| Q2-F4 | info | Projection compatibility for canonical operation/supersedes fields | `plugins/truth-ledger/projection.py:46-66`; `tests/plugins/truth_ledger/test_projection.py:59-111` | Build current view from `operation`-field events and retract via `supersedes` | Active state updated correctly, retract removes target | Applied/active counts matched expected cases | truth-ledger plugin |
| Q2-F5 | info | Hook-path latency headroom | `plugins/truth-ledger/spool.py:62-79` | 250 enqueue operations under local probe | p95 < 10ms | p95 = 0.76ms (PASS) | truth-ledger plugin |

## T5-T8 acceptance mapping

- T5 (schemas/contracts): PASS
- T6 (identity/idempotency/reconciliation): PASS
- T7 (admission/redaction/identity tests in focused suite): PASS
- T8 (spool/ledger/projection/concurrency/latency): PASS

## Residual risk notes (non-blocking)

1) Redaction efficacy remains environment-dependent and should continue to be asserted via hermetic `scripts/run_tests.sh` coverage before release decisions.
2) Latency numbers above are local probe values and should be re-sampled in production-like load if rollout criteria tighten.

## Final QA decision

Q2 specification/compliance review: PASS for ledger-core scope (T5-T8), with no reproduced critical/important gaps.