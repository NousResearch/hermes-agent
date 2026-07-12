# Mobile Client Contract implementation notes

The normative client-facing Mobile Client Contract is documented in
[Programmatic Integration](../website/docs/developer-guide/programmatic-integration.md#nativemobile-websocket-contract).
That guide owns the authentication flow, wire schemas, compatibility rules,
scope allowlist, reconciliation algorithm, mutation failure semantics, approval
lifecycle, and end-to-end reconnect sequence. Keep wire-level changes there
rather than duplicating the contract in this implementation note.

Hermes advertises the additive contract, schema, and capability fields in the
`gateway.ready` event for every accepted WebSocket. Mobile dispatch authority,
synchronized event shapes, and durable mutation requirements apply only when
the effective `gateway.ready.authorization.audience` is `hermes.mobile`.

## Capability boundary

The contract is implemented on the existing authenticated `/api/ws` JSON-RPC
transport. It does not introduce another runtime or source of conversation
state. `tui_gateway/mobile_contract.py` owns:

- protocol, contract, and client-facing schema majors;
- independently advertised capability descriptors;
- the supported mobile scopes;
- the fail-closed method, parameter, and scope policy; and
- the stable effective authorization shape copied from the consumed ticket.

An advertised contract major or Hermes release version never implies an
individual capability. Legacy dashboard and stdio grants keep their prior
authority and wire behavior; only the `hermes.mobile` audience enters the
mobile allowlist.

## Synchronization boundary

`tui_gateway/mobile_sync.py` owns one `SessionEventStream` for each reconstructed
live session. The server process identity is stable for one process, while each
stream gets a new stream identity. A restart or reconstructed stream therefore
produces an explicit reset instead of pretending replay is complete.

Snapshot capture and event publication use this lock order:

1. acquire the conversation history lock;
2. acquire the per-stream reentrant lock;
3. mutate snapshot-visible state, allocate a sequence, retain the replay frame,
   and enqueue transport delivery under that stream boundary.

The snapshot watermark consequently covers every state transition published
before capture. Clients can either replay a complete interval from their exact
cursor or install the returned snapshot. Replay eviction reports a gap; a
missing, invalid, foreign-process, or foreign-stream cursor reports a reset.
Neither outcome may masquerade as complete recovery.

Replay is process-local and bounded by both event count and encoded byte size.
Retention begins when a mobile-audience transport first attaches. If a legacy
transport later owns the session, Hermes continues retaining sequenced copies
for a future mobile reconnect while sending the legacy transport its unchanged
event shape.

## Durable mutation receipts

`tui_gateway/mobile_mutations.py` owns SQLite-backed at-most-once receipts that
are independent of a live TUI session. Receipt identity is scoped to the
effective authenticated provider and subject. Its fingerprint binds the method,
Hermes resource identity, and method-specific semantic parameters.

Only methods listed by `mutation.idempotency.methods` enter this path. A
same-principal retry with the same fingerprint replays the stored outcome;
changed semantics conflict. A reservation that might have executed but cannot
be completed safely becomes `outcome_unknown` and is never released for
automatic duplicate execution. On process startup, another process's unfinished
reservation is terminalized the same way.

Prompt receipts have a stronger completion boundary than handler return. The
gateway binds an opaque proof tag to the exact user turn and marks the receipt
complete only after that turn is proven in durable history. Queued, streaming,
and reconnect paths share one condition-driven receipt coordinator per live
session; a lost proof becomes `outcome_unknown`, not a successful receipt.

## Approval lifecycle

`tools/approval.py` owns approval identities and lifecycle state. Callers cannot
choose reserved lifecycle fields. Public descriptors are recursively and
forcibly redacted at the approval-core boundary before any gateway event or
snapshot sees them.

Each pending approval has one Hermes-issued identity. Mobile resolution targets
that exact identity, while ID-less FIFO and resolve-all behavior remain legacy
only. The terminal callback runs after the core state transition and before the
blocked waiter is released, so the sequenced terminal event cannot be overtaken
by downstream tool or turn events.

Pending approvals and terminal tombstones are process-local. A transport
reconnect can recover them while the same process and live stream retain the
session, but a server/stream reset invalidates that live approval state.
Completed `approval.respond` mutation receipts remain durable independently of
the in-memory tombstone.

## Conformance ownership

The public conformance path lives with the authenticated FastAPI/WebSocket tests
in `tests/hermes_cli/test_dashboard_auth_ws_auth.py`. It must exercise the real
ticket, WebSocket, dispatcher, synchronization, receipt, and approval seams in
one generic flow. Unit tests under `tests/tui_gateway/` continue to cover race,
overflow, redaction, persistence, and compatibility edge cases at their owning
modules.
