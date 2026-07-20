# Q2 — Code-quality review (ledger core)

Date: 2026-07-19
Task: t_a180b5b5
Reviewer: automation-operator (fresh worker)
Plan: `/Users/hermes/.hermes/hermes-agent/.hermes/plans/2026-07-17_143520-truth-ledger-option-2.md`
Workspace: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`
Commit reviewed: `48ee34c2bf448a480ef499bb8d24d184d0fcc198`

## Scope

Fresh code-quality review of T5-T8 for:
- API boundaries and module coupling
- readability and maintainability
- test adequacy and coverage shape
- races/error-path handling
- privacy/permission posture

No code fixes in this task.

## Independent evidence run

1) Focused regression suite
- Command:
  - `scripts/run_tests.sh tests/plugins/truth_ledger/test_contracts_and_schemas.py tests/plugins/truth_ledger/test_spool.py tests/plugins/truth_ledger/test_ledger.py tests/plugins/truth_ledger/test_projection.py tests/plugins/truth_ledger/test_concurrency.py tests/plugins/test_truth_ledger_reconciliation.py tests/plugins/test_truth_ledger_admission_redaction.py tests/agent/test_turn_finalizer_post_llm_call_metadata.py tests/agent/test_turn_finalizer_interrupt_alternation.py tests/agent/test_turn_finalizer_final_response_persistence.py tests/agent/test_turn_finalizer_cleanup_guard.py -q`
- Result: 11 files, 61 passed, 0 failed.

2) Static syntax sanity
- Command:
  - `python -m py_compile plugins/truth-ledger/*.py tests/plugins/truth_ledger/*.py tests/plugins/test_truth_ledger_reconciliation.py tests/plugins/test_truth_ledger_admission_redaction.py tests/agent/test_turn_finalizer_post_llm_call_metadata.py`
- Result: pass.

3) Targeted API-shape probe for reconciliation→projection
- Probe A result:
  - `{'decision': 'append', 'op': 'assert', 'has_top_level_scope': False, 'active': 1, 'applied': 1}`
- Probe B result (two distinct reconciled facts):
  - `{'events': 2, 'active': 1, 'view_lines': 1}`

## Findings

| ID | Severity | Area | Evidence | Finding |
|---|---|---|---|---|
| Q2Q-F1 | important | API boundary / coupling | `plugins/truth-ledger/reconciliation.py:189-210`, `plugins/truth-ledger/projection.py:47-49`, probe B above | Reconciliation emits canonical event objects with nested `fact` (`fact.scope/subject/key/value`). Projection computes logical keys from top-level `scope/subject/key`. When projection consumes reconciliation-shaped events directly, multiple distinct facts collapse to the same logical key (`"||"`), causing silent state overwrite/data loss in current view. |
| Q2Q-F2 | moderate | Test adequacy / integration safety | `tests/plugins/test_truth_ledger_reconciliation.py`, `tests/plugins/truth_ledger/test_projection.py`, `tests/plugins/truth_ledger/test_ledger.py` | Tests are strong at unit boundaries but do not include end-to-end reconciliation→ledger append→projection rebuild invariants. This gap allowed Q2Q-F1 to persist while all focused tests remain green. |
| Q2Q-F3 | moderate | Portability / maintainability | `plugins/truth-ledger/ledger.py:29-43` | `_FileLock` depends on `fcntl` (POSIX-only). No platform guard/fallback is present. On Windows execution paths, this will fail at runtime if this component is activated there. |
| Q2Q-F4 | info | Privacy boundary awareness | `agent/turn_finalizer.py:386`, `plugins/truth-ledger/redaction.py:31-45`, `plugins/truth-ledger/admission.py:61-63,93-99` | `post_llm_call` hook receives full `conversation_history`; downstream truth-ledger redaction/admission layers do enforce stripping/redaction, but the boundary is currently policy-by-convention (plugin discipline) rather than upstream minimization. |

## Race/error-path notes

- Positive:
  - Append path uses lock + fsync + checksum/index update (`plugins/truth-ledger/ledger.py:121-141`).
  - Corrupt tail quarantine exists (`plugins/truth-ledger/ledger.py:146-181`).
  - Spool hard-cap fails closed with explicit reason, and soft overflow sheds oldest to dead-letter (`plugins/truth-ledger/spool.py:62-79`, `147-163`).

- Risk concentration:
  - Data-shape coupling between reconciliation and projection remains the highest correctness risk.

## Verdict

FAIL for Q2 code-quality gate at this time due to Q2Q-F1 (important API-boundary/coupling defect with reproducible state-loss behavior when canonical reconciliation events are projected).

## Recommended follow-up (no fixes applied here)

1) Unify event contract between reconciliation output and projection input (single canonical shape).
2) Add an integration test that starts from reconciliation events and asserts projection cardinality/value correctness across >=2 distinct logical keys.
3) Add portability guard/explicit non-support path for `fcntl` lock behavior.
4) Consider upstream minimization for `post_llm_call` payloads before plugin hooks where feasible.
