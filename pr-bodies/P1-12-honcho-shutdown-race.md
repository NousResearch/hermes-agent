## Summary

Fix a race condition in `HonchoSessionManager.shutdown()` where new messages could be enqueued between `flush_all()` and the shutdown sentinel, causing data loss.

## Bug

`shutdown()` at `plugins/memory/honcho/session.py:548` called `flush_all()` then `put(_ASYNC_SHUTDOWN)`. Between these two operations, concurrent `save()` calls could enqueue new items that would never be flushed — the async writer would process them but the shutdown sentinel would arrive after, and the `join(timeout=10)` would abandon them.

## Fix

Added a `_shutting_down` flag to `HonchoSessionManager`:

1. `shutdown()` now sets `self._shutting_down = True` before calling `flush_all()`
2. `save()` checks `self._shutting_down` and returns early (no-op) if true
3. This eliminates the flush-then-sentinel gap entirely

## Impact

- **Severity**: P1 — data loss of unsent messages during shutdown
- **Scope**: Honcho memory plugin (`plugins/memory/honcho/session.py`)
- **Risk**: Minimal — flag is per-manager instance, only affects the async write path

## Testing

- `tests/plugins/memory/test_honcho_shutdown_race.py`: 4 regression tests
  - `test_save_dropped_after_shutdown_flag`: save() is a no-op after shutdown
  - `test_shutdown_sets_flag_before_flush`: flag is set before flush_all()
  - `test_concurrent_save_and_shutdown_no_race`: no orphaned items after concurrent save+shutdown
  - `test_shutdown_idempotent`: calling shutdown() twice does not crash
