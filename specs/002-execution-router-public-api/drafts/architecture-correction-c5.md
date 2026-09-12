# Architecture Correction Set C5 — Cancellation, Event Binding and Pagination

Stage: ARCHITECTURE_CORRECTION
Status: ACCEPTED
Project-Version: 0.1.1
Architecture-Version: 1
Reviewed-Plan-Record-SHA256: `29417434f995f59dc162b137104993c905443fdf70b1eb5f8c64b9b5f58bb72a`
Affected-Task: T002
Supersedes-On-Acceptance: C4 callback signature; C4 event field list only as explicitly stated below
Acceptance-State: ACCEPTED
Accepted-Source-SHA256: `63fb0951747260c95ce33ee81f968ef8a63aaf7fa43ae97de6db0a197de2905b`
Owner-Acceptance: `Принять весь C5 по указанному SHA-256`
Accepted-At: `2026-09-11T11:21:00+03:00`
Integrated-Into-Canonical-Plan: YES
Implementation-Authorization: T002_ONLY

## Purpose and source check

The current T002 pass exposed exactly three architecture blockers: the provider cannot observe cooperative cancellation, the frozen requested event does not bind all exact router inputs, and an attempt-local sequence is used as though it were session-global pagination. Implementing any of these choices without accepted normative text would create public contract behavior by implementer discretion.

The corrections below preserve the accepted product scope: one policy-neutral host capability, exactly three execution kinds, host-owned authority, no credentials or dispatch authority for the plugin, no second store, no global sequence, and no publication, installation, consumer integration, pilot or LIVE effect.

The pre-release compatibility claim was checked against current project state immediately before this draft: Git HEAD is the local upstream-baseline merge `0b3c4e19ed457772dc7c45d413a259d0c11f1a8e`; the repository has no remotes and no tags; `agent/execution_router.py` is not tracked at HEAD; canonical T007 and T008 remain unchecked; canonical product/task boundaries state that local implementation does not constitute release or consumer integration. Therefore no released `execution_router` contract or compatibility consumer exists. Current uncommitted T002 code and tests are implementation work, not a released compatibility surface.

No proposed rule below conflicts with accepted product scope or inspected source truth.

## C5-01 — Cooperative cancellation signal in the provider callback

### Recommended normative replacement text

Replace the C4 one-argument provider callback rule in ARC-003 and ARC-006 with the following:

Public provider contract `1.0` remains a runtime-checkable Python Protocol with a frozen descriptor and exactly one synchronous callback:

`resolve_execution_route(request: ExecutionRouteRequestV1, cancellation: ExecutionRouterCancellationSignalV1) -> ExecutionRouteDecisionV1 | None`

`ExecutionRouterCancellationSignalV1` is a host-owned immutable read-only capability. Its public API is exactly:

- `is_cancelled() -> bool`, returning the current host cancellation state;
- no setter, `cancel()`, mutable attribute, wait/join primitive, callback registration, host object reference or state-mutation method.

The host retains the private cancellation controller. The provider receives only the immutable read side and may poll `is_cancelled()` to return promptly. The host sets cancellation when any of the following first occurs while resolution is open:

1. the 250 ms monotonic deadline expires;
2. the owning provider is targeted for unload or disable;
3. the active provider generation is revoked or superseded.

Cancellation is cooperative only. At 250 ms the host atomically closes arbitration as `router_error/timeout` regardless of provider cooperation, persists the closed outcome through the existing lifecycle authority, abandons every late return or exception, and never waits for or joins callback termination. Unload or generation revocation similarly closes an open resolution fail-closed, signals cancellation, and rejects late output. The cancellation signal cannot authorize dispatch, credentials, tools, retries, config mutation, lifecycle mutation or any other host-state change.

This correction supersedes C4’s one-argument signature. Contract version remains `1.0` because the contract has not been released, tagged, published, installed or consumed as a compatibility dependency; the corrected two-argument signature is the only signature eligible for the first release. No released consumer migration or backward-compatibility adapter is required.

### Alternatives and exclusions

- Rejected: retain the one-argument callback and describe daemon abandonment as cooperative cancellation. Abandonment protects host arbitration but gives a cooperative provider no cancellation observation mechanism.
- Rejected: expose `threading.Event`, a future/task, worker/thread handle, mutable token, host lifecycle object or cancellation callback registry. Each exposes more host machinery or mutation surface than required.
- Rejected: join, await, interrupt, kill or otherwise wait for arbitrary callback termination after arbitration closes. Version 1 continues to make no hard-kill, descendant-cleanup, process-isolation or sandbox guarantee.
- Rejected: add async, network, model or tool work to the supported callback contract.
- Rejected: support both one-argument and two-argument callbacks. A compatibility overload would preserve an unreleased defect and make provider admission ambiguous.

### Compatibility consequences

- Current uncommitted providers, fixtures and tests using the C4 one-argument form must be changed together under T002 before any release.
- Runtime protocol/admission checks must reject the one-argument form as incompatible contract `1.0`; there is no silent fallback invocation.
- Existing request, decision, consent tuple, execution-kind and deadline semantics remain unchanged. The signal adds no plugin authority and does not change product scope.

### Required tests

1. RED→GREEN provider-admission test: one-argument callback is rejected; exact two-argument callback is admitted.
2. RED→GREEN deadline test: host sets cancellation by 250 ms, returns timeout without waiting for callback termination, and discards a later decision/exception.
3. RED→GREEN unload and generation-revocation tests: each sets the same signal, closes the open resolution fail-closed and prevents late output from changing the durable result or events.
4. Immutability/authority test: provider cannot cancel, reset, wait on, register through or otherwise mutate host state via the public signal.
5. Compatibility test: no-router behavior remains native and creates neither callback nor synthetic route event.

## C5-02 — Exact requested-event binding fields

### Recommended normative replacement text

Extend the exact frozen `ExecutionRouteEventV1` field list from C4 by exactly three nullable fields:

- `request_digest`;
- `instruction_digest`;
- `eligibility_revision`.

Field legality is closed by event type:

- `route_requested` requires non-null exact `request_digest` and non-null exact `eligibility_revision` copied from the validated immutable request.
- For `route_requested`, `instruction_digest` is the exact SHA-256 of the post-redaction/post-truncation UTF-8 instruction bytes actually delivered to the provider; it is non-null when instruction text is present and null if and only if the request carries the accepted explicit no-instruction representation.
- `route_accepted`, `route_started`, `route_not_started` and `route_finished` require all three binding fields to be null. Their correlation is through the same `(request_id, attempt_id)` and the authoritative preceding `route_requested`; duplicating the values in every event is not permitted.
- `request_digest` and non-null `instruction_digest` are exact lowercase 64-character hexadecimal SHA-256 values. `eligibility_revision` obeys its existing exact request field constraints.
- No route event payload stores raw instruction text, a candidate list, system prompt, conversation history, credentials or arbitrary plugin payload.

The existing `execution_route_attempts` durable row stores exact `request_digest`, exact `eligibility_revision`, and a new nullable `instruction_digest`. Creation/replay of the requested lifecycle validates all three values against the exact immutable request. A duplicate `(request_id, attempt_id)` with any different binding value is a conflicting replay and fails closed. Event reconstruction copies requested-event binding values only from this authoritative attempt row.

This correction supersedes C4’s exact event field list only by adding these three fields and their closed nullability rules. All other C4 event fields, transitions, attempt-local sequence semantics, SessionDB ownership, recovery and deletion rules remain unchanged.

### Alternatives and exclusions

- Rejected: store only `request_digest` and assume it is sufficient for observation. That does not expose the accepted exact instruction/eligibility bindings required to distinguish disclosed input and authority revision.
- Rejected: store raw instruction text in `event_json` or the attempt row. The event/read contract is a digest projection, not an instruction archive.
- Rejected: duplicate all three binding values on every lifecycle event. The authoritative requested event plus exact attempt correlation is sufficient and minimizes persisted/public data.
- Rejected: create a new event, binding table, aggregate store or plugin-owned journal. The existing SessionDB attempt/event authority is extended minimally.

### Compatibility consequences

- The unreleased frozen event constructor, canonical JSON fixtures, SessionDB schema and event readers must be updated atomically under T002.
- Existing uncommitted databases/tests may require the normal project-local schema reconciliation used by this implementation branch; there is no released database or event consumer compatibility promise.
- Decision semantics, route identities, event transitions, retention ownership and contract version `1.0` remain unchanged.

### Required tests

1. RED→GREEN event-shape matrix covering required/non-null/null legality for all five event types, including explicit no-instruction versus empty-string instruction.
2. RED→GREEN binding test proving `route_requested` carries the exact request digest, exact delivered-instruction digest and exact eligibility revision from the validated request.
3. RED→GREEN durable replay test proving instruction digest is stored on the attempt and any mismatch in the three bindings fails closed without a new event or sequence increment.
4. Persistence/recovery/readback test proving bindings survive SessionDB reopen and owning-session deletion removes the same rows under existing policy.
5. Redaction test proving neither event JSON nor the public read projection contains raw instruction text, full prompt/history, candidate list or credentials.

## C5-03 — Attempt-local sequence and opaque session pagination

### Recommended normative replacement text

`sequence` remains monotonic only within exact `(request_id, attempt_id)`. It is never session-global and no global sequence is introduced.

The read contract has two distinct cursor modes:

1. Attempt-local sequence mode:
   - `after_sequence` and `before_sequence` are optional attempt-local bounds.
   - Supplying either bound requires both exact `request_id` and exact `attempt_id` in the same request.
   - Omitting either identity, supplying only one identity, or using identities that do not form one authoritative attempt fails closed before querying.
   - Results are ordered by `sequence`, then `event_id`, within that exact attempt. Bounds are exclusive.

2. Session-wide page-token mode:
   - The first page is requested without a page token and is ordered deterministically by exact `(timestamp_utc_ms, event_id)` ascending within the already authorized owning session and normalized filters.
   - If more records exist, the host returns an opaque next-page token bound to the authorized session/scope, normalized filters, contract/token version and the final returned `(timestamp_utc_ms, event_id)` cursor.
   - A subsequent page uses only that host-issued token for position. The host validates token structure, integrity, version, scope/session binding and filter binding before querying; malformed, altered, expired/unsupported-version or mismatched tokens fail closed and return no records.
   - Page-token mode is mutually exclusive with `after_sequence`, `before_sequence`, caller-supplied timestamp cursors and a different filter set. The caller cannot provide or edit the internal timestamp/event cursor.
   - Pagination uses the existing authoritative event rows and introduces no global sequence, aggregate store, mutable cursor record, dispatch/replay authority or stronger delivery guarantee.

`event_id` is the deterministic tie-breaker for equal timestamps. Filtering never changes the cursor ordering key. Existing maximum page size of 100 records remains unchanged.

### Alternatives and exclusions

- Rejected: apply `after_sequence` across a session. Different attempts legitimately reuse the same local sequence values, causing omissions or duplicates.
- Rejected: invent a session-global sequence or renumber persisted events. That adds authority and migration machinery not required by the accepted product.
- Rejected: expose raw `(timestamp_utc_ms, event_id)` as caller-controlled session pagination fields. A host-issued opaque token is required to bind scope and filters fail-closed.
- Rejected: persist mutable page cursors or create a pagination store. The token represents the deterministic read cursor; event authority remains unchanged.

### Compatibility consequences

- The unreleased read API must replace session-wide `after_sequence` behavior with the two explicit modes. Callers using sequence bounds must provide exact request and attempt IDs.
- Session-wide callers must consume the returned opaque token and may not construct continuation cursors.
- Event storage, attempt-local sequence generation, event contract version `1.0`, page ceiling and permission scope remain unchanged.

### Required tests

1. RED→GREEN attempt-local tests proving sequence values may repeat across attempts without cross-attempt omission and both IDs are mandatory for either sequence bound.
2. RED→GREEN exclusive `after_sequence`/`before_sequence` boundary tests for one exact attempt, including invalid identity pairs.
3. RED→GREEN session pagination tests with equal timestamps proving stable `(timestamp_utc_ms, event_id)` ordering, no duplicate/omitted records across pages and unchanged filters.
4. Fail-closed tests for malformed, altered, wrong-version, wrong-session/scope and filter-mismatched tokens, and for combining page tokens with sequence or caller timestamp cursors.
5. Authority test proving pagination is read-only and creates no global sequence, cursor row, aggregate store, replay, retry, dispatch, completion or mutation method.

## Scope and acceptance effect

C5 changes no accepted product requirement, execution kind, consent authority, credential boundary, routing policy, dispatcher ownership, retry budget, event owner, delivery guarantee or release boundary. It adds no second store, global sequence, worker-termination guarantee, plugin mutation capability, consumer implementation, commit, publication, installation, pilot or LIVE behavior.

Acceptance of the exact whole-file C5 draft resolves only these three architecture blockers and authorizes their integration into the corresponding normative ARC-003/004/006 clauses under the existing project governance. Acceptance does not itself integrate this draft into `plan.md`, does not mark or complete T002, and does not authorize a commit or any later task.

All other confirmed T002 implementation findings remain required RED→GREEN work under T002. C5 neither waives those findings nor converts partial source/tests into T002 completion.

## Acceptance metadata placeholders

- Exact accepted-source SHA-256 presented for acceptance: `63fb0951747260c95ce33ee81f968ef8a63aaf7fa43ae97de6db0a197de2905b`
- Owner decision: `ACCEPT`
- Owner acceptance reference/text: `Принять весь C5 по указанному SHA-256`
- Accepted at: `2026-09-11T11:21:00+03:00`
- Integrated into canonical `plan.md`: `YES`
- T002 completion effect: `NONE`
