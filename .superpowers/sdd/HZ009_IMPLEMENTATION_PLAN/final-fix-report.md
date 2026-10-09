# HZ-009 final review fix report

## Scope and result

- Branch: `fix/hermes-zeca-v0215-regressions-20261009`
- Fix-pass base: `66357f76f6a312929a6689c34610b2b0b8f29a05`
- Implementation commit: `73d18348631d5ba495e59e74613c909c501be00b`
- Review source: `final-review.md` (three Important findings, no Critical findings)
- Result: all three Important findings are fixed with deterministic RED-to-GREEN contracts.
- Live Hermes runtime, real credentials, network refresh endpoints, config, gateway, and dependencies were not touched.

## RED evidence

All pytest commands used the parent checkout's existing development Python with a fresh hermetic `HERMES_HOME`, `HERMES_IGNORE_USER_CONFIG=1`, `HERMES_DISABLE_LAZY_INSTALLS=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and only the checked-out worktree source.

```bash
$PY -m pytest -o addopts= -p pytest_asyncio.plugin -p no:cacheprovider --tb=short -q \
  tests/agent/test_context_compressor_route.py \
  -k 'concurrent_generation_zero or abort_after_concurrent_winner'
```

Observed before implementation: `2 failed, 9 deselected`. Both generation-zero tickets reached the durable reset, and a losing durable abort restored stale live state.

```bash
$PY -m pytest -o addopts= -p pytest_asyncio.plugin -p no:cacheprovider --tb=short -q \
  tests/agent/test_pool_revert_after_cooldown.py \
  -k 'reclaim_and_normal_persist or borrowed_root_single_use_reclaim'
```

Observed before implementation: `2 failed, 18 deselected`. Normal `_persist()` timed out on the reversed auth/pool order, and the second profile failed stale instead of adopting the single source-serialized rotation.

During final self-review, an additional Anthropic ordering contract was added:

```bash
$PY -m pytest -o addopts= -p pytest_asyncio.plugin -p no:cacheprovider --tb=short -q \
  tests/agent/test_pool_revert_after_cooldown.py::test_borrowed_anthropic_reclaim_locks_source_before_owner_store
```

Observed RED: `1 failed`; the exact owner store was entered before the Anthropic source singleton.

## Implementation

### Important 1 — compressor same-owner atomicity

- Added a component-owned `threading.RLock` to `ContextCompressor`.
- Preparation snapshots generation/session state under that lock.
- Commit holds it across generation/session validation, live publication, atomic durable reset, and generation publication.
- Abort uses the same boundary and restores live state after a durable commit only when the durable CAS proves that ticket still owns the winner state.
- Deterministic contracts prove that two generation-zero commits produce exactly one durable write and that a lost cross-instance durable CAS cannot roll live state back.

### Important 2 — credential pool/auth hierarchy

- Terminal publish, ordinary reclaim commit, and abort now acquire the pool lock before the exact owner auth-store lock.
- Live generation/mutation epoch and the durable row are revalidated only after both locks are held.
- The deterministic normal `_persist()` versus reclaim interleaving completes both threads and preserves the newer normal-persist CAS winner.

### Important 3 — authoritative single-use source transaction

- Borrowed Codex/xAI reclaim uses the canonical active-profile-to-provider-source transaction and then the exact owning row store; it never takes root before active.
- Borrowed Anthropic reclaim uses the existing source-specific `claude_code` or `hermes_pkce` singleton lock before the exact owner row store.
- Sync, refresh POST, authoritative source write, durable row CAS, and live publication are serialized for the borrowed reclaim transaction.
- A two-profile shared-root Codex contract proves one POST, one rotation, winner adoption by both callers, root-only persistence, and no profile-local pool fork.

## GREEN and regression evidence

Focused contract progression:

- Compressor findings: `2 passed, 9 deselected`.
- Pool hierarchy and shared-root rotation findings: `2 passed, 18 deselected`.
- Anthropic source-before-owner self-review contract plus adjacent findings: `3 passed, 18 deselected`.
- Full two modified focal files: `32 passed in 2.81s`.
- Expanded credential/runtime focal selection: `164 passed in 8.68s`.

Exact final 21-file selection from the accepted plan:

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

Final post-self-review result: `930 passed, 4 skipped in 52.10s`.

One earlier full-selection attempt reported `928 passed, 4 skipped, 1 failed` because pytest collected two unrelated background `unraisable` recursion warnings in `TestCooldownReentryAbort`. The exact failed test immediately passed alone (`1 passed`); the unchanged selection then passed (`929 passed, 4 skipped`), and the final post-self-review selection above passed with the added contract (`930 passed, 4 skipped`). No production change was made for that environmental flake.

Static verification:

- Ruff on the accepted production/test file set: `All checks passed!`
- `python -m py_compile` on all modified Python files: passed.
- `python scripts/check_compat_pointers.py`: `no in-tree dependency on the 2085 plugin-compat pointers`.
- `git diff --check` and staged diff check: passed.

The canonical `scripts/run_tests.sh` wrapper was attempted first but the sandbox denied its worker spawn with `PermissionError` and collected zero tests. Per the accepted ledger ruling, direct hermetic pytest through `/Volumes/MauxOs/Hermes/hermes-agent/.venv/bin/python` was used; this was not counted as a runner pass.

## Final self-review

- Critical: `0` open.
- Important: `0` open.
- Minor: no new Minor. The previously accepted target-only read-only `_is_sole_credential()` deferral remains documented and non-blocking.
- Compatibility: direct `ContextCompressor.update_model` behavior, policy-off routing, exact-row durable ownership, and existing ABA/cooldown compensation contracts remain covered by the final suite.
- Residual concern: only the documented one-off pytest `unraisable` flake above; it did not reproduce in isolation or either subsequent exact-selection pass.

## Verdict

The HZ-009 branch is ready for final review of this fix pass. Promotion, merge, push, and any live-runtime action remain out of scope.
