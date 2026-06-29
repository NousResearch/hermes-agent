# Hindsight shutdown lifecycle + NJ component FTD push guards

## Scope

Implement a durable fix for two risks:

1. Hindsight background retain/prefetch work can outlive Hermes session teardown and race CPython interpreter shutdown, producing `cannot schedule new futures after interpreter shutdown`, `coroutine was never awaited`, or unclosed aiohttp client warnings.
2. NJ legal corpus autonomous FTD worktrees need local technical push guards so case-law and statutes profiles cannot accidentally push `main`, sibling branches, tags, deletes, or non-fast-forward updates.

## Reviewed design

Claude Code performed an oppositional review and returned `PASS_WITH_CHANGES` in `docs/hindsight-shutdown-and-ftd-push-guards-claude-opposition-2026-05-15.md`. Blocking edits incorporated here:

- Use a strong provider registry, not weakrefs.
- Close coroutine objects both before scheduling and when `run_coroutine_threadsafe` fails.
- Make atexit order non-load-bearing by setting a process shutdown flag first.
- Put `shutdown_memory_provider()` idempotency inside the method, not only in `AIAgent.close()`.
- Audit `AIAgent.close()` callers before adding memory shutdown.
- Push guards validate stdin ref lines, not current branch.
- Push guards iterate every stdin line.
- Treat `extensions.worktreeConfig` as repo-global and verify sibling worktrees.
- Distinguish non-fast-forward from unknown remote tip in hook stderr.
- Make `.githooks/` untracked via per-worktree `.git/info/exclude` and `chmod +x` the hook.
- Add subprocess shutdown, prefetch, session-switch, idempotency, and multi-ref hook verification.

## Hindsight design

### Process state

Add module-global lifecycle state in `plugins/memory/hindsight/__init__.py`:

- `_PROCESS_SHUTTING_DOWN = threading.Event()`
- `_PROVIDERS: set[HindsightMemoryProvider] = set()` behind `_PROVIDERS_LOCK`
- `_ATEXIT_REGISTERED = False`
- `_LOOP_STOPPED = False`

The provider registry is strong on purpose. It keeps initialized providers reachable until explicit `shutdown()` or process finalization closes their clients. `shutdown()` unregisters after successful local teardown.

### Shared loop

`_get_loop()` must refuse to create/reuse the shared async loop after process shutdown begins. `_run_sync(coro, timeout)` must:

1. keep `coro` in a local variable;
2. if process shutdown is set, call `coro.close()` if present and raise `RuntimeError`;
3. call `asyncio.run_coroutine_threadsafe(coro, loop)` in `try`;
4. if scheduling raises `RuntimeError`, close `coro` and re-raise.

### Process finalizer

Replace provider-specific atexit callbacks with one module-level finalizer:

1. set `_PROCESS_SHUTTING_DOWN` first;
2. snapshot strong providers under lock;
3. call `provider.shutdown()` for each;
4. stop and join the shared loop once.

Because the flag is set before draining/stopping, any later atexit callback that tries to schedule Hindsight work fails closed instead of reviving or using a closed loop.

### Provider operations

- Register initialized providers in the strong registry.
- `sync_turn()`, `queue_prefetch()`, `_ensure_writer()`, `_run_hindsight_operation()`, and `on_session_switch()` must skip/fail closed when provider or process shutdown is active.
- The prefetch thread must re-check process/provider shutdown after it starts and before async scheduling.
- Memory tools return JSON `tool_error(...)` when shutdown is active.
- Ordinary provider `shutdown()` remains idempotent and provider-local. It drains the writer, joins prefetch briefly, closes its client, unregisters itself, and does not stop the shared loop.
- Process finalizer only stops the shared loop after all providers are drained.
- Add a test helper `_reset_shutdown_state_for_tests()` so pytest state does not poison later tests.

### AIAgent lifecycle

Audit result: current `release_clients()` is the gateway cache-eviction path and must not tear down memory. `close()` is hard teardown. `shutdown_memory_provider()` is called directly in CLI/background review paths, so idempotency must live inside `shutdown_memory_provider()`.

Implementation:

- Initialize `self._memory_shutdown_done = False` before memory manager construction.
- At top of `shutdown_memory_provider()`, return if already done; set the flag immediately.
- `close()` calls `shutdown_memory_provider()` unconditionally and relies on the method guard.
- `release_clients()` remains unchanged.

## Tests

Run and/or add tests covering:

- `_run_sync()` refuses work after process shutdown and closes coroutine.
- `_run_sync()` closes coroutine when `run_coroutine_threadsafe` raises.
- `sync_turn()` and `queue_prefetch()` no-op after process shutdown.
- Prefetch thread re-checks shutdown after spawn.
- `on_session_switch()` does not enqueue flushes after process shutdown.
- Hindsight tool calls fail closed during shutdown.
- Module finalizer shuts down multiple providers and stops loop.
- Ordinary provider `shutdown()` does not stop shared loop.
- Subprocess integration: child process constructs/uses provider then exits without explicit shutdown; stderr must not contain `cannot schedule new futures`, `coroutine was never awaited`, or `Unclosed client session`.
- `shutdown_memory_provider()` idempotent under direct call and `close()`; `release_clients()` does not shut memory down.

## FTD push guard design

Install untracked per-worktree hooks in:

- `/Users/johngalt/Projects/nj-legal-corpus-case-engine` allowing only `refs/heads/ftd/case-law-engine`
- `/Users/johngalt/Projects/nj-legal-corpus-statutes` allowing only `refs/heads/ftd/statutes`

Implementation:

1. Enable `extensions.worktreeConfig=true` once at the shared NJ repo level. This is repo-global.
2. In each component worktree, set `git config --worktree core.hooksPath .githooks`.
3. Add `.githooks/` to each worktree's `.git/info/exclude` so hooks are untracked and branches remain clean.
4. Write `.githooks/pre-push`, `chmod +x`.
5. Hook reads every stdin line and rejects unless:
   - `local_ref == refs/heads/<allowed>`
   - `remote_ref == refs/heads/<allowed>`
   - `local_sha` is not all zeroes
   - refs are branch refs, not tags or other namespaces
   - if `remote_sha` is not all zeroes, `git merge-base --is-ancestor "$remote_sha" "$local_sha"` returns 0
6. Non-fast-forward (`merge-base` exit 1) and unknown remote object (`>=128`) both block, with distinct stderr messages.
7. Empty stdin is a no-op.
8. README in `.githooks/` documents policy and bypass limits: `--no-verify`, pushing from another clone, or server-side permissions can bypass local hooks.

## FTD verification

- Verify component worktrees are on their expected branches.
- Verify `core.hooksPath=.githooks` only in component worktrees.
- Verify primary `/Users/johngalt/Projects/nj-legal-corpus` and sprint9 verify worktree do not inherit `core.hooksPath`.
- Simulate hook stdin cases:
  - allowed branch line exits 0;
  - multi-line allowed + disallowed exits nonzero;
  - push to main exits nonzero;
  - sibling branch exits nonzero;
  - tag push exits nonzero;
  - allowed branch delete exits nonzero;
  - unknown remote object/non-fast-forward exits nonzero with clear message.

## Smoke verification

- Run targeted pytest suites for Hindsight provider and run_agent lifecycle.
- Run existing CLI/gateway shutdown memory tests.
- Run `hermes memory status` or equivalent local status command and derive the active URL instead of hardcoding `127.0.0.1:9177`.
- Check active local bind remains localhost-only.
- Do not push code without explicit approval.
