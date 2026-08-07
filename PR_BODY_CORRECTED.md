## Summary

Coalesce multiple background process completions and watch events that share the same gateway route so the agent receives **one synthetic turn** instead of one turn per process — fixing the session flood documented in #70300.

This is a two-pronged approach, each at the correct ownership seam:

1. **Standard completions** — batched at `_run_process_watcher()` via `_enqueue_process_completion_notification()` with a fixed 100 ms batching window. Every completion (including singles) pays the full window — there is no zero-latency shortcut because true zero-latency requires limiting coalescing to work already available in the same loop tick, which is incompatible with a time-window batch.

2. **`watch_match` / `watch_disabled` events** — coalesced at the post-turn drain via `_coalesce_and_inject_watch_events()`, grouped by event type and `session_key`.

## Why this supersedes #70319 and #71898

| Feature | #70319 | #71898 | **This PR** |
|---------|:---:|:---:|:---:|
| Correct insertion point (watcher path) | no | yes | yes |
| Correct insertion point (drain path) | yes | no | yes |
| Coalesce standard completions | yes | yes | yes |
| Coalesce `watch_match` events | no | no | yes |
| Coalesce `watch_disabled` events | no | no | yes |
| Bounded batching delay (documented) | no | no | yes |
| Aggregate summary (X succeeded, Y failed) | no | no | yes |
| Elapsed time per process | no | no | yes |
| Dedup/retry per process | no | yes | yes |
| Flush during delivery | no | yes | yes |
| Duplicate primary -> try next identity | no | yes | yes |
| Formatter failure -> resolve all waiters | no | yes | yes |
| Lazy init for `object.__new__` tests | no | yes | yes |
| E2E tests with real watchers | no | yes | yes |
| Drain coalescing tests | no | no | yes |

## Implementation

### `gateway/run.py` (+327/-10)

- **`GatewayRunner.__init__`**: Added batch state (`_completion_notification_batches`, `_completion_notification_batch_tasks`, lock, window=100ms)
- **`_enqueue_process_completion_notification()`**: Fans in concurrent completions sharing one route. All completions (including singles) are grouped for the full 100 ms batching window, then delivered as one synthetic turn.
- **`_flush_process_completion_batch()`**: Delivers one batch after the window. Handles CancelledError (resolves waiters as retryable), duplicate primary fallback, records sibling identities on success, resolves all waiters on failure. Every waiter resolves to an explicit result — `True` (delivered), `False` (retryable), or `None` (permanently unroutable / lifecycle duplicate).
- **`_format_coalesced_process_completions()`**: Compact format with per-process exit codes, elapsed time, bounded output (10 entries, 120-char command / 800-char output tails), aggregate summary over ALL entries ("3 completed: 2 succeeded, 1 failed"). No emoji — the LLM reads `exit_code=0` the same as any visual indicator.
- **`_coalesce_and_inject_watch_events()`**: Groups `watch_match`/`watch_disabled`/`completion` events by type+session_key at the post-turn drain. Single events pass through unchanged; multiple events produce one batched message.
- **`_format_gateway_process_notification()`**: Extended to handle standard `completion` events (previously returned `None`).
- **`_run_process_watcher()`**: Changed from direct `_deliver_completion_notification()` -> `_enqueue_process_completion_notification()`.
- **Post-turn drain**: Changed from per-event loop -> `_coalesce_and_inject_watch_events()`.
- **Shutdown drain**: `_stop_impl_body` cancels pending completion-batch flush tasks and awaits them (3-second deadline) BEFORE adapter teardown, so waiters are resolved as retryable while adapters are still available.

### Example batched output

```
[IMPORTANT: 3 background processes completed for this session.
Treat these results as one completion batch and send at most one
consolidated user-facing response.

  proc_a: exit_code=0, reason=exited, 12.3s
  success output here

  proc_b: exit_code=1, reason=error, 8.7s
  error output here

  proc_c: exit_code=0, reason=exited, 5.1s
  also ok

Summary: 2 succeeded, 1 failed.
If a result does not change the current conclusion, absorb it silently.]
```

### Tests

**`tests/gateway/test_completion_delivery.py`** (34 test functions): three concurrent watchers -> one batched turn, single completion uses original notification, different routes never coalesce, in-flight arrival schedules next flush, failed delivery retries all entries, coalesced success records every identity, duplicate primary -> fresh sibling delivered, format failure resolves all waiters, batch format includes exit codes + summary, entry truncation after 10, output truncation >800 chars, single completion bounded latency (<1s), batch summary counts failures in omitted tail, cancellation during batch window resolves waiters retryable, cancellation during delivery resolves waiters retryable.

**`tests/gateway/test_background_process_notifications.py`** (35 test functions): single watch event pass-through, multiple watch_match coalesced, different session_keys not coalesced, mixed types coalesced separately, ID truncation after 5, empty list noop.

```
$ python3 -m pytest tests/gateway/test_completion_delivery.py tests/gateway/test_background_process_notifications.py -q
69 passed   # scope: tests/gateway/test_completion_delivery.py + test_background_process_notifications.py
```

## Edge cases covered

- **Flush during delivery**: Detaches batch before adapter I/O; new completions schedule the next flush
- **Duplicate primary**: Tries next batch identity when first entry is a lifecycle duplicate
- **Formatter failure**: Resolves all waiter futures with `False` so watchers retry
- **Lazy init**: Covers tests that construct `GatewayRunner` with `object.__new__`
- **None-safe key**: All batch key fields use `str(field or "")`
- **Cancellation safety**: CancelledError during batch window or adapter-blocked delivery resolves all waiters as retryable; no unresolved Future, no stale batch entry, no stale task key

## Known limitations

- **Sibling identity race (P5)**: The primary identity is claimed in-flight before delivery, but sibling identities in the batch are only recorded after the adapter returns. A concurrent replay of a sibling during this window can form a second synthetic turn. Mitigation tracked in [follow-up issue].
- **No persistence layer**: Completions resolved as retryable during shutdown are not persisted — they disappear when the gateway process exits. This is consistent with the existing (pre-PR) behaviour for single-process notifications.
- **Tri-state return (`True`/`False`/`None`)**: `_deliver_completion_notification()` returns `Optional[bool]` with three overloaded meanings (delivered / retryable / dropped). A follow-up should replace this with an explicit `CompletionDisposition` enum to separate permanent-unroutable from temporary-unavailable and ensure every waiter path is explicit.

## Review history

- **Round 1** (`8840cc4`): Removed unimplemented threshold flush, fixed `zero-latency` claim -> bounded window, removed stale docstrings, addressed P1 lifecycle and P3 summary scope.
- **Round 2** (`3449790`): Fixed CancelledError handling (P1 cancellation hole), shutdown drain ordering, added omitted-tail summary regression test, removed remaining stale docstrings.
- **Round 3** (this update): Corrected PR body to match HEAD, fixed co-author trailer format, aligned shutdown comments with actual behaviour.

---

Closes #70300
Supersedes #70319 and #71898
