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

Synchronization state and replay retention begin when an authorized mobile
transport first attaches to a live session. A session that has never negotiated
sync does not pay the per-delta replay copying cost. If a legacy transport later
attaches to a previously negotiated session, retention continues for the next
mobile reconnect, but the legacy response and event shapes remain unchanged.

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

## Durable consequential mutations

When `mutation.idempotency` version 1 is advertised, its `methods` list is the
complete set of mobile methods covered by durable receipts. Each covered
request requires a non-empty `client_request_id`. Prompt submission,
interruption, and approval response additionally require the stable
`expected_stored_session_id`; approval response also requires the
Hermes-issued `approval_id`.

Receipts are scoped to the authenticated provider and subject. Repeating the
same request identity with equivalent normalized semantics returns the stored
result with `mutation.deduplicated` set to `true`. Reusing it with different
semantics returns `mutation_conflict`. A request abandoned after execution may
have begun is reported as `mutation_outcome_unknown` and is never executed
again automatically. Clients can inspect a known receipt with the advertised
`mutation.status` method.

This guarantee applies only to the methods named by the capability on a
mobile-scoped connection. Existing legacy transports retain their prior
request shapes and behavior.

## Recoverable approval lifecycle

When `interaction.lifecycle` version 1 names `approval` in `kinds` and
`approval.respond` in `response_methods`, approval requests use the
`approval.lifecycle` version 1 schema. Each request carries a stable
Hermes-owned `approval_id`, server-redacted presentation fields, creation and
expiry times, current state, and resolution metadata. Pending descriptors are
part of the authoritative synchronization snapshot and remain addressable
after reconnect.

Hermes emits `approval.request` for creation and one of `approval.resolved`,
`approval.expired`, or `approval.stale` for a terminal transition. Every event
and snapshot descriptor carries the same `approval_id`. A mobile response must
name that identity, the stable conversation lineage, a choice, and a durable
client request identity. Identical retries replay the stored mutation result;
changed semantics conflict. Short-lived terminal tombstones distinguish
`already_resolved`, `expired`, `stale`, and `not_found` outcomes without
consuming another pending approval.

Approval payload and resolution metadata redaction is server-owned. Mobile
clients must not infer authorization from presentation fields: the response is
available only when the lifecycle capability, mutation coverage,
`conversation.control` grant, live reconciled state, and valid Hermes resource
identities all agree. ID-less legacy desktop and stdin responses retain their
existing FIFO behavior.
