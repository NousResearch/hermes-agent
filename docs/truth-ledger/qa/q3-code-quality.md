# Q3 — Code-quality review (extraction/projection/integration)

Date: 2026-07-19
Task: t_0ddaaddf
Reviewer: automation-operator (fresh worker)
Plan: `/Users/hermes/.hermes/hermes-agent/.hermes/plans/2026-07-17_143520-truth-ledger-option-2.md`
Workspace: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`
Commit reviewed: `48ee34c2bf448a480ef499bb8d24d184d0fcc198`

> Historical point-in-time review. Subsequent remediation bounded and lock-protected
> `_SEEN_ENVELOPES` at 1,024 entries, fixed nested dead-letter reason rendering,
> added bounded-dedupe regression coverage, and preserved downstream idempotency.
> The missing automatic pending-queue consumer remains intentionally deferred in
> the reduced MVP and is still documented in the operator runbook.

## Scope

Fresh code-quality review for T9-T12 with focus on:
- async/shutdown behavior
- leaks and races
- provider coupling
- error handling and observability
- operator UX and test adequacy

No fixes in this task.

## Independent evidence run

1) Focused Q3 test suite
- Command:
  - `scripts/run_tests.sh tests/plugins/truth_ledger/test_lifecycle_integration.py tests/plugins/truth_ledger/test_commands.py tests/plugins/truth_ledger/test_extractor.py tests/plugins/truth_ledger/test_projection.py tests/plugins/truth_ledger/test_spool.py tests/agent/test_turn_finalizer_post_llm_call_metadata.py -q`
- Result: 6 files, 48 passed, 0 failed.

2) Syntax sanity
- Command:
  - `python -m py_compile plugins/truth-ledger/__init__.py plugins/truth-ledger/extractor.py plugins/truth-ledger/commands.py plugins/truth-ledger/spool.py plugins/truth-ledger/projection.py tests/plugins/truth_ledger/test_lifecycle_integration.py tests/plugins/truth_ledger/test_commands.py tests/plugins/truth_ledger/test_extractor.py tests/plugins/truth_ledger/test_projection.py`
- Result: pass.

3) Runtime probe — in-memory dedupe growth and no reset
- Probe result:
  - `{'seen_size_after_posts': 2000, 'seen_size_after_session_start': 2000}`
- Evidence: `plugins/truth-ledger/__init__.py:27,126-137,140-145`

4) Runtime probe — queue does not drain on session start
- Probe result:
  - `{'pending_before_session_start': 3, 'processing_before_session_start': 0, 'pending_after_session_start': 3, 'processing_after_session_start': 0}`
- Evidence: `plugins/truth-ledger/__init__.py:140-145`; no extraction/reconciliation/append call path from hooks.

5) Runtime probe — dead-letter review reason blind spot
- Probe result (`review_report` after dead-lettering with reason=`schema_mismatch`):
  - `dead_letter_preview[0].reason == "unknown"`
- Evidence: `plugins/truth-ledger/commands.py:157-163` reads only top-level `dead_letter_reason`/`reason`; actual reason is nested under `flow.dead_letter_reason` from spool (`plugins/truth-ledger/spool.py:347-355,408-416`).

## Findings

| ID | Severity | Area | Evidence | Finding |
|---|---|---|---|---|
| Q3Q-F1 | important | Leak / long-running stability | `plugins/truth-ledger/__init__.py:27,126-137,140-145` + probe #3 | `_SEEN_ENVELOPES` is process-global, grows per eligible turn, and is never bounded/compacted/reset by lifecycle hooks. This is an unbounded memory-growth surface for long-lived gateway sessions. |
| Q3Q-F2 | important | Async/shutdown lifecycle robustness | `plugins/truth-ledger/__init__.py:116-145`; usage search shows `extract_candidates` only in extractor tests and not in hook path | Runtime hook wiring enqueues source envelopes and recovers stale `processing`, but does not run a bounded consumer loop that drains `pending` into extraction/reconciliation/ledger/projection. This creates an accumulating backlog until queue limits trigger shed/dead-letter behavior. |
| Q3Q-F3 | moderate | Observability / operator UX | `plugins/truth-ledger/commands.py:157-163`; `plugins/truth-ledger/spool.py:347-355,408-416` + probe #5 | `review_report` surfaces dead-letter reason as `unknown` for normal spool-produced records because it ignores `flow.dead_letter_reason`. Operators lose actionable triage signal. |
| Q3Q-F4 | moderate | Race / duplicate-pressure guardrail | `plugins/truth-ledger/__init__.py:126-133`; `plugins/truth-ledger/spool.py:257-275` | Dedupe check/add is in-memory and non-atomic across concurrent hook invocations (`if in set` then `add`), so duplicate enqueues remain possible under contention; downstream idempotency is deferred to later stages. |
| Q3Q-F5 | moderate | Test adequacy | `tests/plugins/truth_ledger/test_extractor.py`, `test_lifecycle_integration.py`, `test_commands.py` | Tests are strong per module but do not cover end-to-end hook->spool->extract->reconcile->ledger->projection flow, do not assert bounded dedupe memory behavior, and do not verify dead-letter reason rendering in `review_report`. |

## Positive quality notes

- Hook eligibility contract is explicit and readable (`completed/failed/interrupted/turn_exit_reason/delegation/kanban` gates).
- Fail-open behavior on hook errors is consistent and tested.
- Spool and projection maintain defensive file-permission and corrupt-tail handling patterns.
- Structured extraction tests cover retries, schema mismatch, and redaction leakage paths well.

## Verdict

FAIL for Q3 code-quality gate at this time.

Blocking reasons: Q3Q-F1 and Q3Q-F2 (unbounded process-state growth and missing bounded runtime drain path), with additional observability/test gaps in Q3Q-F3/Q3Q-F5.

## Recommended follow-up (no fixes applied here)

1) Replace unbounded `_SEEN_ENVELOPES` with bounded/expiring dedupe or persisted idempotency keyed at spool/ledger boundary.
2) Add explicit bounded lifecycle worker/drain path from pending spool to extraction/reconciliation/append/projection with shutdown-safe limits.
3) Fix `review_report` to read nested dead-letter reason (`flow.dead_letter_reason`) and add regression test.
4) Add an integration test that drives real hook-enqueued envelopes through end-to-end processing and asserts queue drains and projection updates.
