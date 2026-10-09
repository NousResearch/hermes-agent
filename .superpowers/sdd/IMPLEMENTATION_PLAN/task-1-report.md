# Task 1 report — route-scoped token budgets

## Status

DONE_WITH_CONCERNS

The Task 1 contract is implemented and its focal and adjacent verification is green. The repository-wide suite was also launched with the official runner; its final checkpoint is recorded below because it exposes unrelated baseline/environment failures outside this task.

## Changes

- Materialized the preserved token-budget policy and its offline/adversarial regression tests.
- Added `TokenBudgetRuntimeMixin` as a narrow integration boundary instead of expanding `run_agent.py`.
- Applied route-scoped token budgets only after provider, client, compressor, and prompt-cache initialization.
- Made model switch, fallback activation, and primary restore transactional: failed policy application restores the complete runtime route graph.
- Added current 0.21.5 route-side state to the rollback snapshot, including request overrides, capabilities, custom providers, credential/fallback state, and compression notices.
- Enforced request payload caps while preserving explicit one-shot caps and provider-default cleanup when the policy is disabled.
- Kept helper compatibility by omitting optional `capabilities` and `reset_at` keywords when their value is `None`.
- Made the preserved adversarial test file self-contained because the official per-file runner does not make sibling test modules importable.

## TDD evidence

1. **RED (preserved contract):** official runner collected 133 tests and produced exactly **103 passed / 30 failed**; all 30 failures were the expected missing runtime integration cases in `tests/run_agent/test_token_budget_runtime.py`.
2. **GREEN (preserved contract):** after implementation, the same four focal files produced **133 passed / 0 failed**.
3. **Regression added during self-review:** `test_policy_failure_restores_current_switch_side_effects` first failed because new 0.21.5 runtime fields leaked after rollback; after extending the transactional snapshot, the focal set produced **134 passed / 0 failed**.

## Verification

- Focal plus adjacent selection: **593 passed / 0 failed / 4 skipped** (the skips are Windows-only).
- Ruff over changed implementation and tests: **all checks passed**.
- Compatibility-pointer check: **passed** (`no in-tree dependency on the 2085 plugin-compat pointers`).
- `git diff --check`: **passed**.
- Repository-wide official suite: **48,180 passed / 239 failed / 772 skipped** across 4,716 files, 100% complete in 970.2 s with 20 workers (exit 1). Five flaky files failed once and passed on retry.

## Rulings

- Kept the policy engine separate from the runtime adapter so policy rules remain deterministic and independently testable.
- Used a mixin to own only lifecycle boundaries; the 0.21.5 helpers remain the source of truth for provider switching/fallback behavior.
- Treated route mutation as a transaction and restored every current mutable route field on explicit failure or exception.
- Did not install optional dependencies or alter unrelated baseline behavior to make the global suite green.

## Concerns

- `GLOBAL_SUITE_BASELINE_RED`: the repository-wide suite completed but is not a clean task-specific signal in this checkout. Its 239 failures include missing optional `acp` and `anthropic` packages plus unrelated existing gateway/compression/provider failures; five additional files were flaky and passed on retry. None touched the Task 1 files, and the focused/adjacent suite is clean.
- No live configuration, runtime service, network, credential, or launchd state was changed.

## Fix round 1 — reviewer findings

- **Account/route request boundary:** route baselines now include a non-secret account identity (pool entry ID or credential fingerprint). A request detects account/base-url identity drift and transactionally reapplies policy before building payloads. Same physical-route accounts inherit the original pre-policy compressor baseline, never another account's promoted state.
- **Transactional initialization:** policy validation now runs before `init_agent`; post-init policy/compressor failures restore the initialized baseline and close allocated clients.
- **Policy-off fast path:** switch/fallback/restore call the 0.21.5 helper directly when policy is absent or disabled, preserving legitimate upstream bookkeeping on `False`.
- **Prepare/commit side effects:** billing-route writes and fallback/restore notifications are staged while an enabled-policy transition is prepared, committed only after policy success, and discarded on rollback.
- **Request one-shots:** failed request construction restores `_ephemeral_max_output_tokens`, `_ephemeral_reasoning_off`, and `_wire_reasoning_config`, including mutable identity/state.

### Fix-round TDD evidence

1. Added eight reviewer-regression cases and observed **0 passed / 8 failed** before production changes.
2. Applied the five fixes and observed **8 passed / 0 failed** on the same selection.
3. Self-review added the cross-account baseline-removal invariant; it failed with a leaked 500K promoted baseline, then passed after physical-route baseline inheritance was implemented.
4. Final combined focal+adjacent suite: **601 passed / 0 failed / 4 skipped** (Windows-only).
5. Ruff, compatibility-pointer validation, and `git diff --check`: **passed**.
