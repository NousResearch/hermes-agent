# HZ-009.1 credential corrective report

## Scope and result

- Branch: `fix/hermes-zeca-v0215-regressions-20261009`
- Corrective base: `7a31e1100aff2470a88518c4c88a4fa6a009f8fb`
- Implementation commit: `b2ef1de73c505b97f3e2e6c456f47b945a3daa1c`
- Adjudicated source: `final-fix-rereview.md`
- Scope: only the two accepted credential findings. Compressor code/tests were not changed.
- Result: the two lock cycles and tokenless `claude_code` loser gap are closed by one shared refresh transaction hierarchy.
- No live runtime, real credentials, network endpoint, config, gateway, dependency, merge, or push action occurred.

## RED evidence

The three requested deterministic contracts and the corrected pre-existing order contract were added before production changes. Tests used the parent checkout development Python, a fresh hermetic `HERMES_HOME`, disabled user config/lazy installs, and no real network.

```bash
$PY -m pytest -o addopts= -p pytest_asyncio.plugin -p no:cacheprovider --tb=short -q \
  tests/agent/test_pool_revert_after_cooldown.py \
  -k 'owner_store_before_source or deferred_refresh_have_one_pool_first_winner or root_anthropic_refresh_and_borrowed_reclaim or two_profile_claude_code'
```

Observed before implementation: `4 failed, 20 deselected`.

- Anthropic order contract: observed `source -> owner` instead of `owner -> source`.
- Borrowed reclaim versus deferred refresh: deferred refresh failed with `TimeoutError('deferred refresh waited for the live pool')`.
- Root Anthropic refresh versus profile reclaim: root refresh failed with `TimeoutError('lock cycle at .credentials.json')`.
- Two-profile `claude_code`: one caller failed with `RuntimeError('stale credential reclaim ticket: durable row changed')`.

## Corrective design

### One hierarchy for ordinary refresh and reclaim

`CredentialPool._single_use_refresh_transaction()` is now the single component-owned boundary for both paths:

```text
local pool -> active profile auth -> exact owner/root store -> external source singleton
```

- Ordinary single-use refresh enters the local pool boundary before any auth/source lock, so it never upgrades `auth -> pool` during `_replace_entry()` or `_persist()`.
- Borrowed reclaim delegates to the same transaction instead of maintaining a second lock implementation.
- Codex/xAI retain the canonical active-to-provider-source transaction; the exact row owner is nested in that direction.
- Anthropic acquires the exact owner/root row store before `claude_code` or `hermes_pkce` source locks, eliminating the root/source cycle.
- Reentrant mutation/persistence remains inside the same component transaction, and the live entry is re-read after entering the boundary.

### Tokenless Claude loser adoption

- A newer healthy durable row with a higher reclaim revision is still the CAS winner even when `claude_code` sanitization removed its raw tokens.
- While the authoritative Claude source lock is held, the loser re-reads the rotated source pair.
- The source access token must match the durable row's `secret_fingerprint` before it is merged with the durable revision/status metadata.
- The losing live pool adopts that hydrated winner without claiming the durable write; the root pool row remains tokenless.

## GREEN and regression evidence

Focused progression:

- Four corrective contracts: `4 passed, 20 deselected in 0.80s`.
- Determinism repeat A: `4 passed, 20 deselected in 0.85s`.
- Determinism repeat B: `4 passed, 20 deselected in 0.84s`.
- Full modified focal file: `24 passed in 2.43s`.
- Credential/profile/lifecycle focal selection: `87 passed in 7.32s`.
- Additional deferred-refresh, Anthropic race/stress, borrowed authority, persistence-failure, plugin and terminal adjacency: `40 passed, 1 skipped in 2.33s`.

Exact 21-file selection from the accepted HZ-009 plan:

```bash
$PY -m pytest -o addopts= -p pytest_asyncio.plugin -p no:cacheprovider --tb=short -q \
  tests/agent/test_token_budget_policy.py \
  tests/run_agent/test_token_budget_runtime.py \
  tests/hermes_cli/test_config.py \
  tests/agent/test_model_metadata.py \
  tests/agent/test_switch_model_context.py \
  tests/agent/test_primary_runtime_restore.py \
  tests/agent/test_context_compressor.py \
  tests/agent/test_context_compressor_route.py \
  tests/agent/test_context_engine.py \
  tests/agent/test_context_engine_host_contract.py \
  tests/agent/test_plugin_context_engine_init.py \
  tests/agent/test_compression_anti_thrash_persistence.py \
  tests/agent/test_proactive_prune_restart_safety.py \
  tests/agent/test_compression_rotation_state.py \
  tests/agent/test_pool_revert_after_cooldown.py \
  tests/agent/test_credential_rotation_route_settings.py \
  tests/agent/test_fallback_credential_isolation.py \
  tests/agent/test_pool_rotation_endpoint_veto.py \
  tests/agent/test_codex_responses_adapter.py \
  tests/agent/test_run_agent_codex_responses.py \
  tests/agent/transports/test_codex_transport.py
```

Result: `933 passed, 4 skipped in 58.55s`.

Static verification:

- Ruff on the accepted production/test file set: `All checks passed!`
- `python -m py_compile` on all modified Python files: passed.
- `python scripts/check_compat_pointers.py`: no in-tree dependency on the 2,085 compatibility pointers.
- Worktree and staged `git diff --check`: passed.

The direct hermetic pytest route follows the runner ruling already accepted in the HZ-009 ledger; no blocked wrapper result was counted as a pass.

## Self-review

- Critical: `0` open.
- Important: `0` open within the two adjudicated credential findings.
- Scope: exactly `agent/credential_pool.py`, `agent/credential_pool_reclaim.py`, and `tests/agent/test_pool_revert_after_cooldown.py`; compressor remains closed and untouched.
- Durable secrecy: the final Claude root row contains no `access_token` or `refresh_token`; its fingerprint, revision, and status identify the winner.
- Trade-off: a single-use refresh holds the local pool boundary across its source transaction. This is the explicit hierarchy selected by the re-review to remove lock upgrades; provider-specific timeout bounds remain in force.
- Previously deferred target-only read-only Minor remains deferred and was not reassessed.

## Verdict

HZ-009.1 is ready for final scoped re-review. The branch and worktree remain local and preserved; promotion remains out of scope.
