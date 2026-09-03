# Layers from issue #102271

Issue: "Tool-invocation progress bubbles never carry the `_interim_send`
metadata marker (`gateway/run.py`'s `ctx._progress_metadata`)". Tool-progress
status lines were emitted on stream-is-the-message adapters (relay Slack
native, etc.) without the consumer-required `_interim_send` flag, so the
first such bubble during a streaming turn sealed the live stream with status
text and the real final posted as a duplicate while later frames were
silently swallowed by the seal tombstone.

1. **L1 — `_progress_metadata` assembly** (`gateway/run.py:31726-31750`):
   the assembly currently flows through `_non_conversational_metadata()` only,
   which is a Discord-only no-op for every other platform. The assembly must
   also wrap the result in `_interim_metadata()` so the dict (or None) is
   always marked `_interim_send=True`. → `gateway/run.py:31741`

2. **L2 — Relay adapter consumer** (`gateway/relay/adapter.py:1830, 1993`):
   `send_for_platform` and `send` both pop `_interim_send` from the metadata
   before sealing the open native stream — without L1, every tool-progress
  bubble for stream-is-the-message adapters is misclassified as the turn
   final. This is the consumer side of the contract; verify L1 wires the
   marker through unchanged.

3. **L3 — Stream consumer commentary lanes** (`gateway/stream_consumer.py:2723,2768`):
   `_send_commentary` already sets `_interim_send=True` on its metadata
   inline. Verify the L1 fix does NOT collide with that path: a tool-progress
   bubble must be interim, and a separate commentary send is also interim —
   the markers stack, never clash.

4. **L4 — `_progress_metadata` may be None** (truthiness gate at
   `gateway/run.py:5426`, `if ctx._progress_metadata:`): today the dict is
   sometimes None when no thread metadata could be resolved. The fix must
   preserve the truthiness contract (callers still rely on `if
   ctx._progress_metadata:` to skip the edit branch). Therefore wrap with
   `_interim_metadata()` which guarantees a non-None dict with the marker
   set — both progress-message detection AND interim-marking satisfied.

5. **L5 — `_non_conversational_metadata()` is a no-op outside Discord**:
   non-Discord platforms pass through unchanged. The fix must apply
   `_interim_metadata()` AFTER `_non_conversational_metadata()` so the wrap
   composes correctly across platforms. On Discord the
   `_non_conversational_metadata` returns the merged dict with
   `non_conversational=True`, and the subsequent `_interim_metadata()` wrap
   adds `_interim_send=True` to the same dict — both markers coexist.

6. **L6 — Slack native task cards + fallback drain sites**: lines 5251, 5259,
   5278, 5350, 5482, 5587, 5637, 5653, 5661, 5673 all use
   `metadata=ctx._progress_metadata`. Fixing at the assembly site covers
   every consumer in one shot — no per-call-site patch.

7. **Edge: idempotency** — `_interim_metadata(None)` returns
   `{"_interim_send": True}`. If `_progress_metadata` already contains the
   key (e.g., from a future code path), the wrap merges rather than
   overwriting, so `_interim_send=True` stays `True`.

## Test coverage

- New regression test: `tests/gateway/test_progress_metadata_interim_marker.py`
  verifies the assembly site produces `_interim_send=True` on every branch
  (with thread, without thread, Slack native cards) by importing the helper
  and exercising the relevant code path against upstream's `gateway.run`.
- Existing tests: `tests/gateway/test_interim_send_lanes.py` pins the
  `_interim_metadata` helper contract (we don't change it).
