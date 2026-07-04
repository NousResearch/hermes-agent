# Hindsight shutdown lifecycle + FTD push guards implementation report

## Result

Implemented the accepted plan in `plans/2026-05-15-hindsight-shutdown-and-ftd-push-guards.md`.

Claude review artifacts:

- Plan opposition: `docs/hindsight-shutdown-and-ftd-push-guards-claude-opposition-2026-05-15.md`
- Code opposition: `docs/hindsight-shutdown-code-claude-review-2026-05-15.md`

The code review returned `PASS_WITH_NOTES`. Follow-up fixes applied for the material notes:

- `_stop_shared_loop()` now attempts to cancel pending loop tasks before stopping/closing the shared loop.
- `_stop_shared_loop()` logs if the loop thread fails to stop within the bounded timeout.
- `HindsightMemoryProvider.shutdown()` unregisters the provider in a `finally` block.
- The instance `_run_sync()` wrapper now supports `allow_during_shutdown` so shutdown paths do not bypass the wrapper.

A reviewer note about reusing a provider after `shutdown()` was intentionally not adopted: current tests assert post-shutdown `sync_turn()` is dropped. Treat provider `shutdown()` as terminal for that provider instance.

## Implemented changes

### Hermes/Hindsight

Files changed:

- `plugins/memory/hindsight/__init__.py`
- `run_agent.py`
- `tests/plugins/memory/test_hindsight_provider.py`
- `tests/hermes_cli/test_agent_memory_shutdown_lifecycle.py`

Behavior:

- Replaced per-provider atexit callbacks with one process-wide Hindsight finalizer.
- Added strong provider registry so initialized providers remain reachable until explicit provider shutdown or process finalization.
- Added process shutdown flag; `_get_loop()` and `_run_sync()` refuse new async work after shutdown begins.
- `_run_sync()` closes unscheduled coroutine objects when shutdown blocks scheduling or `asyncio.run_coroutine_threadsafe()` raises.
- Process finalizer drains providers, then stops the shared event loop once.
- Provider `shutdown()` drains queued retain jobs, joins prefetch, closes client resources, unregisters the provider, and does not stop the shared loop.
- `sync_turn()`, `queue_prefetch()`, prefetch worker, `on_session_switch()`, and memory tool calls fail closed during provider/process shutdown.
- `AIAgent.shutdown_memory_provider()` is idempotent and is called from `AIAgent.close()`; `release_clients()` still does not tear down memory.

### FTD profile config

Changed both component profiles:

- `/Users/johngalt/.hermes/profiles/nj-case-law-ftd/config.yaml`
- `/Users/johngalt/.hermes/profiles/nj-statutes-ftd/config.yaml`

`display.background_process_notifications` is now `result` for both profiles.

Both profile gateways were restarted and verified live:

- `nj-case-law-ftd` PID 65558
- `nj-statutes-ftd` PID 65589

### NJ component push guards

Installed untracked per-worktree hooks:

- `/Users/johngalt/Projects/nj-legal-corpus-case-engine/.githooks/pre-push`
- `/Users/johngalt/Projects/nj-legal-corpus-statutes/.githooks/pre-push`

Worktree config:

- Case-law worktree: `core.hooksPath=.githooks`, allowed ref `refs/heads/ftd/case-law-engine`
- Statutes worktree: `core.hooksPath=.githooks`, allowed ref `refs/heads/ftd/statutes`

Guard behavior:

- Allows only the component branch to the same remote branch.
- Blocks `main`/`master`, sibling component branches, tags/non-branch refs, remote deletes, unknown remote tips, and non-fast-forward updates.
- Reads every stdin ref line.
- `.githooks/` is ignored through each worktree's `.git/info/exclude`.

Primary and sprint verification worktrees do not inherit `core.hooksPath`.

## Verification run

Commands executed and passed:

```bash
python -m pytest tests/plugins/memory/test_hindsight_provider.py \
  tests/hermes_cli/test_agent_memory_shutdown_lifecycle.py \
  tests/gateway/test_shutdown_memory_provider_messages.py \
  tests/gateway/test_agent_cache.py \
  tests/hermes_cli/test_kanban_boards.py -q
# 220 passed
```

```bash
python -m py_compile plugins/memory/hindsight/__init__.py run_agent.py \
  tests/plugins/memory/test_hindsight_provider.py \
  tests/hermes_cli/test_agent_memory_shutdown_lifecycle.py \
  tests/hermes_cli/test_kanban_boards.py
# exit 0
```

Runtime checks:

- `hermes memory status`: Hindsight active and available.
- Hindsight API listener: `127.0.0.1:9177`.
- Gateway status: default gateway and both FTD profile gateways loaded/running.
- FTD liveness: `ftd_liveness_check.py` exits `0`.
- `ftd_status.py --kanban`: no currently registered FTD projects.
- Hook simulations: allowed branch succeeds; `main`, sibling branch, tag, delete, and unknown remote commit are blocked for both component worktrees.

## Residual risks

- Local Git hooks are guardrails, not authority. `--no-verify`, another clone, or permissive server-side settings can bypass them. GitHub branch protection/rulesets should still protect `main`.
- The default gateway process was verified running but was not restarted from this live Discord session to avoid killing the session before reporting. The changed code is on disk and covered by tests; it will load on next default gateway restart.
- Component worktrees each have one local commit ahead of origin. No push was performed.
