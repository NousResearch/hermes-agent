# Mobile Client Contract

Hermes exposes its additive Mobile Client Contract on the existing
authenticated `/api/ws` JSON-RPC transport. Clients must negotiate features
from `gateway.ready`; a contract major or server version alone does not imply a
capability.

## Revisioned conversation synchronization

When `conversation.sync` version 1 is advertised, `session.create` and
`session.resume` include the same `synchronization` envelope:

- `snapshot` is authoritative conversation state. It identifies the server
  process, live event stream, stable conversation lineage, current stored
  session tip, and process-local live session. It also includes a revision,
  event watermark, messages, inflight turn, active tool descriptors, pending
  interactions, and runtime status.
- `recovery` reports `complete`, `gap`, or `reset`, the cursor at the snapshot
  watermark, and any replayable events after the supplied cursor.
- Every session event carries `schema_major`, `stream_id`, and a monotonically
  increasing `sequence` within that stream.

The replay store is bounded in memory by both the advertised event and byte
limits. `gap` means an event needed after the supplied cursor was evicted.
`reset` means the cursor is absent, invalid, from another server process, or
from another reconstructed live stream. Clients must never treat either
outcome as complete replay.

## Snapshot and event barrier

Hermes serializes snapshot-visible live state, event sequence allocation,
replay retention, and event transport enqueueing at one per-stream boundary.
Snapshot capture takes the conversation history lock before that stream
boundary. The returned watermark therefore covers every state transition
published before the snapshot.

A client should:

1. Buffer events for the returned `stream_id` while create or resume is in
   flight.
2. Install the complete snapshot at its `watermark`.
3. Discard buffered or replayed events at or below the watermark.
4. Apply only events for the same server and stream whose sequence is greater
   than the watermark, in sequence order.
5. Replace local state from the snapshot whenever recovery is `gap` or
   `reset`.

Assistant `message.delta` events additionally carry one `turn_id` and an
absolute `offset`. The `conversation.sync.delta_offsets.unit` capability names
the unit as `utf8_bytes`. Clients can therefore ignore an overlapping prefix
instead of duplicating text after replay or transport coalescing.

This slice does not claim durable mutation idempotency or addressable,
recoverable approvals. Those features require separately advertised
capabilities.
