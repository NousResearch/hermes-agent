# Request-local A2A authentication facts

`A2ASecurityContext.authenticate_request()` creates a frozen `A2AAuthentication`
after server-side credential verification. It contains a peer, an explicit method
(`peer_bearer`, `shared_bearer`, or `local_socket`) and a server-generated request
identifier. It contains no credential. Authentication acceptance and identity selection
are unchanged: named entries precede the shared token, and duplicate configured tokens
retain the existing last-name-wins identity. Observation is deliberately stricter:
shared/named collisions and credentials assigned to multiple distinct names return
no facts (`None`), not a new 401 or a different identity. A repeated entry for the
same name is not ambiguous. Classification concerns the presented credential, not
other credentials configured for that name. Local socket identity and shared-token
identity are never named-peer proofs. `authenticate_request()` is an observation
API, not a replacement for the existing `authenticate()` admission decision.

The HTTP handler passes the object explicitly through SendMessage/SendStreamingMessage
preparation into `MessageEvent.a2a_auth`. Direct internal calls without an object do
not acquire provenance merely by passing a peer string. Existing Tasks, framing,
redaction, trust gates, futures and the direct response protocol are unchanged.

## Gateway and tools

Native admission serializes events before execution. Its owner-only envelope carries
an opaque `a2a_auth_ref`, resolved against a weak, process-local registry. Resolution
requires the same platform, profile, peer, context, message id and text digest.
It returns the original immutable object, never deserializes an authentication claim.
The live ingress/event/turn owns its lifetime; there is no durable authority store.
Missing, altered or expired references return `None`. A process restart cannot
reconstitute authentication from a stored sender name or request id.

Turn preparation copies the event field into the ephemeral `SessionContext` and
`set_session_vars` binds it to a typed ContextVar. The existing context-copying
executor passes it to in-process tool handlers. Every new bind defaults to `None`;
clear/reset also remove it. It is excluded from prompt rendering, session serialization,
environment-variable fallback and subprocess export.

The production `a2a_current_authentication` tool reads
`gateway.session_context.get_a2a_authentication()` without task/audit lookup.
It is registered through the existing public PluginContext machinery in the `a2a`
toolset, declared in `plugin.yaml`, and uses the existing A2A opt-in configuration
gate. It is not added to every platform's core tools. The declaration stays stable
across turns; request availability is checked in the handler, not a cached gate.

Call with `{}` only. Both the closed schema and handler reject identity, method,
task, context, or other arguments. Dispatcher kwargs cannot supply authentication.
Without live native provenance it returns `available: false`. Otherwise it returns
only `available`, `peer`, `method`, `named_peer`, and `authorization_granted: false`.
Shared/local facts are available but `named_peer` is false; no credential, request
identifier, task identifier, message, transcript or private file data is returned.

These facts do **not** authorize operations or certify operator approval. Existing
manual scope, deny policy, privacy restrictions and effect-specific approval checks
remain unchanged, including the existing inbound platform hint. A named peer is
not the operator. This tool is not a filesystem executor or a machine-enforced
approval receipt, and cannot prove an exact user approval timestamp, use budget,
or reservation. Model/live approved-action end-to-end acceptance remains untested;
registry/native transport tests use a controlled turn double, not a live model.
Past tool results copied into history are not current-request proof: call again in
the current turn; do not use absence as permission or reconstruct facts from text.
Trusted Python plugins are within the trust boundary; a frozen dataclass is not a
sandbox against code already executing inside Hermes.

## Forwarding and non-goals

The current `a2a.forward` owner RPC accepts peer attribution but has no attested
per-request HTTP authentication contract. Its existing request/response behavior is
preserved; no auth object or new auth field is sent. The target's new event has no
proof, so a consumer requiring named-peer provenance must refuse. Extra auth claims
on this wire are rejected by its existing closed parameter schema. Neither an
`a2a_identity` policy nor `HERMES_A2A_PEER` is upgraded into authentication.

Cross-process provenance transfer, restarted/expired requests, delegated subprocesses
and other unsupported transports are intentionally unsupported, not silently trusted.
This patch does not add a generic executor, change prompt trust tiers, remove Tasks,
modify retry/revival policy, or fix unrelated same-context reply-routing/task-access
isolation. Those transport concerns require separate review; identity-context tests
must not be presented as a comprehensive multi-tenant authorization audit.
