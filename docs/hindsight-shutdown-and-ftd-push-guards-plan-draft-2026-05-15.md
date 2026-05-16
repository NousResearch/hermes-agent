# Hindsight shutdown lifecycle + component FTD push guards — draft implementation plan

## Scope

Implement a complete, durable fix for two concrete control-plane risks:

1. **Hindsight short-process shutdown race** in Hermes: post-response memory sync/prefetch/retain work can race interpreter shutdown and emit `cannot schedule new futures after interpreter shutdown` or leave async clients unclosed.
2. **NJ legal corpus component FTD push safety**: case-law and statutes autonomous worktrees may push only their own component branches and must be technically prevented from pushing `main`, sibling branches, tags, deletes, or force/non-fast-forward updates.

This is not a temporary mitigation. The final state must be committed-quality code/config with regression tests and operational verification.

## External facts checked

- Git `pre-push` hooks receive pushed refs on stdin as `<local-ref> <local-sha1> <remote-ref> <remote-sha1>` and can reject a push by exiting non-zero. Source: `git-scm.com/docs/githooks`.
- Git worktrees share repo config by default; per-worktree config requires `extensions.worktreeConfig=true`, then `git config --worktree ...`. Source: `git-scm.com/docs/git-worktree`.
- Python shutdown races of this family happen when background/atexit code starts threads or schedules futures after interpreter shutdown begins. Source: Python/CPython issue discussions and recurring library reports around `RuntimeError: can't create new thread at interpreter shutdown` / `cannot schedule new futures after interpreter shutdown`.

## Current code evidence

- Hindsight provider has a module-global async loop and per-provider writer/prefetch threads.
- Provider `shutdown()` drains its writer and closes its client but intentionally does not stop the module-global loop because sibling providers may share it.
- Each provider currently registers a provider-specific `atexit` shutdown callback only after a retain is queued.
- `_run_sync()` always creates/reuses the module-global loop and schedules the coroutine without checking a process-level shutdown flag.
- `queue_prefetch()` checks only provider-local `_shutting_down` before creating a daemon thread; the prefetch thread does not re-check shutdown immediately before scheduling async work.
- `hindsight_retain` / `hindsight_recall` / `hindsight_reflect` tool paths do not fail closed once shutdown begins.
- `AIAgent.close()` currently cleans process/browser/client resources but does not call `shutdown_memory_provider()`. CLI/gateway paths may call it elsewhere, but the agent object itself is not a fail-safe hard owner of provider teardown.
- NJ component worktrees are on `ftd/case-law-engine` and `ftd/statutes`, both ahead by one commit, with no current `core.hooksPath`.

## Design goals

### Hindsight

- No new async work after process shutdown begins.
- No per-provider `atexit` callbacks racing each other or reviving the shared loop.
- Multiple provider instances can coexist in a gateway process; shutting one provider must not break sibling providers.
- Process exit must close all known Hindsight clients and stop the module-global loop after all providers are drained.
- Normal per-session `shutdown()` remains provider-local and safe for gateway session expiry.
- Short one-shot `hermes chat -q ...` must explicitly close memory provider before interpreter exit where possible; `atexit` remains a seatbelt, not the primary path.
- Tests must reproduce the lifecycle rules without requiring a live Hindsight daemon.

### FTD push guards

- Use per-worktree hook config, not shared `.git/hooks`, because case-law/statutes have different allowed branches.
- Block everything except fast-forward pushes from the component branch to its identical remote branch.
- Block main/master, sibling branches, tags, deletes, and force/non-fast-forward pushes.
- Do not enforce path scopes in the first technical guard because shared migration/schema/support files may legitimately cross component boundaries; path scope remains audit/review policy until a narrower allowlist is proven.
- Preserve existing repo working state and do not merge to `main`.

## Implementation plan

### Phase 1 — Hindsight lifecycle hardening

1. Add module-global lifecycle state in `plugins/memory/hindsight/__init__.py`:
   - `_PROCESS_SHUTTING_DOWN = threading.Event()`
   - `_PROVIDERS = weakref.WeakSet()` guarded by `_PROVIDERS_LOCK`
   - `_ATEXIT_REGISTERED = False`
   - `_LOOP_STOP_TIMEOUT = 5.0`
2. Register each initialized provider in `_PROVIDERS`.
3. Replace per-provider `atexit.register(self._atexit_shutdown)` with one module-level process finalizer registered idempotently.
4. Add `_process_atexit_shutdown()`:
   - set `_PROCESS_SHUTTING_DOWN` first;
   - snapshot providers from the weak set;
   - call each provider shutdown with a process-finalizer flag that permits closing clients but prevents new work;
   - after all providers are drained/closed, stop and join the module-global event loop thread exactly once.
5. Harden `_get_loop()` and `_run_sync()`:
   - if `_PROCESS_SHUTTING_DOWN` is set, close the coroutine object if possible and raise a controlled `RuntimeError` instead of creating/scheduling work;
   - if the loop is absent/stopped and process shutdown has started, do not recreate it;
   - catch `RuntimeError` from `run_coroutine_threadsafe`, close the coroutine, and re-raise cleanly.
6. Harden provider operations:
   - `_run_hindsight_operation()` refuses work if provider or process shutdown is active.
   - `sync_turn()` skips on provider/process shutdown.
   - `queue_prefetch()` skips on provider/process shutdown and the spawned prefetch body re-checks before scheduling work.
   - `handle_tool_call()` returns a JSON tool error for retain/recall/reflect if shutdown is active.
   - `on_session_switch()` does not enqueue flush-on-switch after process shutdown begins.
7. Keep provider-local `shutdown()` semantics intact but make it idempotent. Do not stop the shared loop from ordinary provider shutdown.
8. Add a private helper `_stop_shared_loop()` used only by process finalizer/tests.
9. Modify `AIAgent.close()` to call `shutdown_memory_provider()` once as part of hard teardown. Guard with an instance boolean so repeated `close()` calls are safe. Keep `release_clients()` unchanged; it must not kill memory state on cache eviction.

### Phase 2 — Hindsight tests

Add tests to `tests/plugins/memory/test_hindsight_provider.py` or a focused new test file:

1. `_run_sync()` refuses to schedule after `_PROCESS_SHUTTING_DOWN` and closes the coroutine.
2. `sync_turn()` and `queue_prefetch()` are no-ops after process shutdown.
3. Tool calls fail closed after process shutdown.
4. Process finalizer shuts down multiple providers without using per-provider atexit callbacks and then stops the shared loop.
5. Ordinary provider `shutdown()` does not stop the shared loop when another provider exists.

Add/adjust `run_agent` tests if needed:

6. `AIAgent.close()` calls `shutdown_memory_provider()` exactly once; `release_clients()` does not.

### Phase 3 — Component pre-push guards

For each worktree:

1. Enable per-worktree config safely:
   - `git config extensions.worktreeConfig true`
   - `git config --worktree core.hooksPath .githooks`
2. Install executable `.githooks/pre-push` in that worktree with its component branch embedded:
   - case-law: `ftd/case-law-engine`
   - statutes: `ftd/statutes`
3. Hook logic:
   - assert current branch equals allowed branch;
   - read stdin ref updates;
   - reject deletes (`local_sha` all zeros);
   - reject non-branch refs (tags/notes/etc.);
   - reject `refs/heads/main` and `refs/heads/master`;
   - reject any remote ref other than `refs/heads/<allowed_branch>`;
   - if `remote_sha` is nonzero, require `git merge-base --is-ancestor "$remote_sha" "$local_sha"` to block force/non-fast-forward pushes.
4. Add `README.md` in `.githooks/` explaining the guard and branch policy.
5. Optionally add a small repo-local helper script only if needed; otherwise hooks are enough.

### Phase 4 — Verification

Hindsight/Hermes:

- Run targeted Hindsight provider tests.
- Run targeted `run_agent` close/release tests.
- Run memory/gateway shutdown tests already present if relevant:
  - `tests/cli/test_cli_shutdown_memory_messages.py`
  - `tests/gateway/test_shutdown_memory_provider_messages.py`
- Run syntax/import checks.
- Smoke-test current Hindsight service with no secrets:
  - `hermes memory status`
  - `curl -fsS http://127.0.0.1:9177/version`
  - verify `127.0.0.1:9177` bind.
- Run a one-shot Hermes memory operation or minimal provider wrapper smoke where feasible without polluting real memory; clean test banks if created.

FTD push guards:

- Verify each worktree branch, hooksPath, hook executable bit.
- Simulate allowed push input to the hook and expect exit 0.
- Simulate blocked cases and expect nonzero:
  - current branch mismatch if possible or by temporary hook invocation environment;
  - remote ref `refs/heads/main`;
  - sibling component branch;
  - tag ref;
  - delete push;
  - non-fast-forward update.
- Do not perform destructive branch switches. Do not merge. Do not push unless explicitly needed for verification; hook simulation is enough for local guard correctness.

## Risks and trade-offs

- A process-level shutdown flag means late memory tool calls during teardown return errors instead of trying best-effort persistence. Correct: after teardown begins, accepting new work is less safe than dropping it.
- The process finalizer uses weakrefs, so unreferenced providers may not be closed by it; explicit agent close remains the primary cleanup. Correct: if nothing references a provider, its owned resources should already be unreachable except shared loop.
- A 10s writer join can still abandon pending retains if Hindsight/Ollama is wedged. Correct: indefinite interpreter shutdown hangs are worse. Log abandonment.
- Pre-push hooks are local client-side guards, not server enforcement. GitHub branch protection/rulesets should still protect `main`. The hook prevents local autonomous mistakes; it cannot stop someone bypassing hooks with `--no-verify` or pushing from another clone.

## Deliverables

- Final plan/spec markdown with Claude opposition findings incorporated.
- Hindsight lifecycle code fix + tests.
- Per-worktree `.githooks/pre-push` and README files for case-law/statutes.
- Verification transcript summary with exact commands and results.
