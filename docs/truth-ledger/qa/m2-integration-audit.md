# M2 integration audit (R5): T5-T8 unpause gate

Date: 2026-07-17
Task: t_d961089e
Workspace: /Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2
Branch: feat/truth-ledger-option-2

## 1) Scope and source-control safety

- Verified shared worktree HEAD before commit work: `6481a6aaaa51c5190f31788f0a4e53fd4ae1e664`.
- Verified dirty source checkout remained untouched and separate:
  - `/Users/hermes/.hermes/hermes-agent` HEAD: `3a1a3c7e6727a31df89b61b27bad313430bdac45`
  - status count/classes during this run: `696`, all `??`.
- No reset/clean/stash/delete operations used.
- `git diff --check` run after changes: clean (no whitespace/check failures).

## 2) Ownership map (project-owned artifacts)

T5 — schemas/contracts
- plugins/truth-ledger/contracts.py
- plugins/truth-ledger/schemas.py
- plugins/truth-ledger/schemas/source-envelope-v1.schema.json
- plugins/truth-ledger/schemas/fact-candidates-v1.schema.json
- plugins/truth-ledger/schemas/ledger-event-v1.schema.json
- plugins/truth-ledger/schemas/spool-record-v1.schema.json
- plugins/truth-ledger/schemas/dead-letter-v1.schema.json
- plugins/truth-ledger/schemas/current-projection-v1.schema.json
- tests/plugins/truth_ledger/test_contracts_and_schemas.py
- docs/truth-ledger/design/schema-contracts.md

T6 — deterministic reconciliation
- plugins/truth-ledger/reconciliation.py
- tests/plugins/test_truth_ledger_reconciliation.py

T7 — admission/redaction/identity
- plugins/truth-ledger/admission.py
- plugins/truth-ledger/redaction.py
- plugins/truth-ledger/identity.py
- tests/plugins/test_truth_ledger_admission_redaction.py

T8 — spool/ledger/projection
- plugins/truth-ledger/spool.py
- plugins/truth-ledger/ledger.py
- plugins/truth-ledger/projection.py
- tests/plugins/truth_ledger/conftest.py
- tests/plugins/truth_ledger/test_spool.py
- tests/plugins/truth_ledger/test_ledger.py
- tests/plugins/truth_ledger/test_projection.py
- tests/plugins/truth_ledger/test_concurrency.py

Shared integration
- plugins/truth-ledger/__init__.py

Not in this commit boundary (pre-existing/unrelated to T5-T8 integration commit)
- docs/truth-ledger/qa/q1-architecture.md (tracked modification from prior Q1 task)
- docs/truth-ledger/discovery/gate-0.md (pre-existing untracked discovery artifact)
- docs/truth-ledger/discovery/repository-conventions.md (pre-existing untracked discovery artifact)

## 3) Fresh audit finding and fix applied in this run

Finding: projection consumer expected legacy event keys (`event`, `retracts`) while reconciliation/schema path emits canonical keys (`operation`, `supersedes`).

Impact:
- Operation-only ledger records were ignored by projection.
- Retract events keyed by `supersedes` were not removing active facts.

TDD evidence for this run:
- RED:
  - `scripts/run_tests.sh tests/plugins/truth_ledger/test_projection.py -q`
  - Failure: `test_rebuild_current_view_accepts_operation_field_for_assert` (`active` expected 1, got 0).
- GREEN:
  - Updated `plugins/truth-ledger/projection.py` to read `operation` with `event` fallback, and `supersedes` with `retracts` fallback.
  - Added regression coverage in `tests/plugins/truth_ledger/test_projection.py`.
  - Re-ran same test file: 4/4 passed.

## 4) Enumerated acceptance-critical test files and reruns

Enumerated Truth Ledger test files in scope:
- tests/plugins/truth_ledger/test_contracts_and_schemas.py
- tests/plugins/truth_ledger/test_spool.py
- tests/plugins/truth_ledger/test_ledger.py
- tests/plugins/truth_ledger/test_projection.py
- tests/plugins/truth_ledger/test_concurrency.py
- tests/plugins/test_truth_ledger_reconciliation.py
- tests/plugins/test_truth_ledger_admission_redaction.py

Enumerated canonical turn_finalizer metadata/regression files:
- tests/agent/test_turn_finalizer_post_llm_call_metadata.py
- tests/agent/test_turn_finalizer_interrupt_alternation.py
- tests/agent/test_turn_finalizer_final_response_persistence.py
- tests/agent/test_turn_finalizer_cleanup_guard.py

Canonical rerun command:
- `scripts/run_tests.sh tests/plugins/truth_ledger/test_contracts_and_schemas.py tests/plugins/truth_ledger/test_spool.py tests/plugins/truth_ledger/test_ledger.py tests/plugins/truth_ledger/test_projection.py tests/plugins/truth_ledger/test_concurrency.py tests/plugins/test_truth_ledger_reconciliation.py tests/plugins/test_truth_ledger_admission_redaction.py tests/agent/test_turn_finalizer_post_llm_call_metadata.py tests/agent/test_turn_finalizer_interrupt_alternation.py tests/agent/test_turn_finalizer_final_response_persistence.py tests/agent/test_turn_finalizer_cleanup_guard.py -q`
- Result: discovered 11 files, 61 tests passed, 0 failed.

## 5) Concurrency and latency evidence (reproduced)

- Concurrency evidence: `tests/plugins/truth_ledger/test_concurrency.py` passed; validates multiprocess append integrity and one-indexed winner for shared race key.
- Hook-path latency benchmark (250 enqueues):
  - p50: 0.468ms
  - p95: 0.721ms
  - p99: 0.790ms
  - target p95 < 10ms: PASS
- Partial-tail quarantine simulation:
  - parsed valid records: 2
  - quarantine files created: 1

## 6) Per-card integration verdicts

- T5 (t_713edf97): PASS for merge-boundary integration. Contracts/schemas validate under Draft 2020-12 and fail closed.
- T6 (t_3775886b): PASS for merge-boundary integration. Deterministic reconciliation semantics validated; idempotent replay behavior covered.
- T7 (t_babe73e3): PASS for merge-boundary integration. Admission/redaction/identity gates enforce fail-closed behavior and sensitive-data exclusion.
- T8 (t_a5e67ca1): PASS after R5 fix. Projection now consumes canonical operation/supersession fields with compatibility fallback; storage tests + concurrency + latency evidence pass.

## 7) Commit boundary rationale

Single milestone commit used for T5-T8 because modules are cross-coupled (schema/reconciliation/admission/storage/projection) and projection compatibility fix spans shared interfaces across cards. Splitting per-card commits would create unsafe intermediate states where ledger/projection semantics diverge.

Commit SHA: recorded in Kanban evidence comments for t_713edf97/t_3775886b/t_babe73e3/t_a5e67ca1.

## 8) Residual risks

- Current projection writer still emits event-shaped records (derived format) and does not yet include full schema-shaped projection entries with explicit `logical_key/state` envelope; current tests validate existing behavior and replay consistency, but this remains a follow-up hardening target.
- Queue-byte caps and configurable threshold plumbing are documented/frozen but only partially represented in code defaults; production rollout should confirm config wiring before enabling beyond controlled evaluation.
