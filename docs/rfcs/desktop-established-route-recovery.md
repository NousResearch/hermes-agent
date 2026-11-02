# RFC: Established gateway recovery across renderer replacement

**Author:** guicybercode

**Date:** 11 September 2026

**Status:** Proposed. Maintainer decision required before implementation.

**Discussion:** [108734](https://github.com/NousResearch/hermes-agent/issues/108734)

## Decision requested

An established gateway route should keep its recovery policy when its Desktop
renderer is replaced. This RFC proposes an asynchronous Electron main service
that owns the existing gateway client and recovery episode. A replacement view
attaches to that service and hydrates from backend truth. First boot remains
bounded, and explicit disconnect, route retirement and application quit remain
terminal actions.

The service would retain independent clients for each window and exact route.
Sharing one physical socket between windows is outside this proposal: backend
viewer membership and orphan handling currently depend on transport identity.
Keeping that separation preserves each window's existing session attachment
semantics while moving their lifetime outside the renderer.

Approval must record the ownership choice, the establishment boundary, detach
limits and hydration requirements below. This document records no maintainer
approval and changes no runtime behavior. Accepting it is the decision phase of
108734; implementation and the specified tests remain necessary afterward.

## Evidence and current boundaries

The source contract was rechecked at
`main@c6f87deb2c38d75518c793790f1cc9afa37f0695`.
The relevant boot, shared client, route identity, replay and session lifecycle
sources are unchanged from the original audit at
`main@0f1668ef76a4401d1d799647199c1a8337c1747c`.
The hook diagnostic described below ran against that earlier baseline.
The audit also includes a diagnostic using the real boot hook and shared client
with a fake socket and main IPC boundary. After at least six failed dials, the
retained hook recovered when the endpoint returned. Recreating the hook during
the outage exhausted six attempts and remained in an error state after endpoint
recovery. The continuity assertion passed for the retained hook and failed for
the replacement. HMR survival was explicitly drained. This is a hook lifecycle
reproduction, distinct from the native rehearsal below.

A production Desktop build on the current source baseline was also exercised
through real Electron 40.10.2, preload IPC and renderer WebSockets on macOS
(Darwin 25.2.0). The loopback gateway supplied fixture HTTP and RPC responses;
HTTP stayed available while WebSocket upgrades received 503 responses. Each
case confirmed establishment through a focus triggered liveness ping, which is
gated by `bootCompleted`, and counted seven rejected upgrades. Keeping the
renderer recovered when upgrades were restored. Reloading it during the outage
instead exhausted six new attempts and displayed the boot recovery overlay.
After upgrades were restored, no new connection appeared within 30 seconds.
The intended continuity assertion passed for the retained renderer and failed
for the reloaded renderer.

The reload kept Electron main alive and cleared a marker in the renderer's
JavaScript context. Chromium reused its renderer process PID, so this proves
document and application replacement through reload, not process crash
recovery. The gateway and credentials were unchanged within each case. The
rehearsal used isolated Hermes home and Electron user data, with no real account.
It does not cover backend session effects, TLS trust, WAN behavior, replay,
load, other operating systems or the complete acceptance matrix below.

1. [useGatewayBoot](../../apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts)
   initializes `bootCompleted` inside its effect. Initial remote boot has one
   attempt plus five retries; an established client instead schedules further
   reconnect attempts with backoff and raises a warning after prolonged failure.
   Production cleanup closes gateways. The survivor path is specific to HMR.
   Merged [106521](https://github.com/NousResearch/hermes-agent/pull/106521)
   correctly combines renderer dial failure with main's boot classification.
   That fix stays in place.
2. [JsonRpcGatewayClient](../../apps/shared/src/json-rpc-gateway.ts) owns pending
   calls, heartbeat state, event watermarks and the backend replay epoch.
   Its `PendingCall` entries hold only Promise resolvers and a timer.
   `request()` deletes them on timeout, abort or send failure, and
   `rejectAllPending()` rejects and deletes them on transport loss. This map
   cannot classify operation outcomes for a replacement renderer.
   `fetchReplay()` returns without a request when a fresh instance has no
   watermarks. A reconnect of the same instance and creation of a replacement
   client therefore have different continuity inputs.
3. [Canonical route matching](../../apps/desktop/electron/connection-route-identity.ts)
   preserves the full registration envelope before dialing.
   [90149](https://github.com/NousResearch/hermes-agent/issues/90149) remains the
   architecture authority for registration generations, activation receipts and
   resource ownership. This checkout does not expose its proposed generation
   bound `RouteKey` throughout Desktop. The generation in
   [backend connection state](../../apps/desktop/electron/backend-connection-state.ts)
   and the activation epoch in the
   [gateway store](../../apps/desktop/src/store/gateway.ts) fence different work;
   neither substitutes for a registration generation.
4. [Session replay](../../tui_gateway/methods_session.py) already returns
   `events`, `latest_seq`, `truncated` and `epoch` from `session.events.since`.
   The shared client handles epoch changes and holds live events during replay,
   but currently does not consume `truncated` as a hydration failure. Session
   resume and history are separate recovery paths. These RPCs do not currently
   promise one atomic snapshot of history, running state and event watermarks.
5. [Session lifecycle](../../tui_gateway/session_lifecycle.py) tracks multiple
   transports, cancels orphan reaping on reattachment, and preserves explicit
   interrupt semantics. Retaining a socket affects when that policy observes
   client absence. Recovery ownership must account for this effect explicitly.
6. The current `hermes:connection:active-route` handler in
   [Electron main](../../apps/desktop/electron/main.ts) records routes by
   `event.sender.id` and removes ownership on WebContents destruction.
   [Renderer lifecycle handling](../../apps/desktop/electron/window-renderer-lifecycle.ts)
   schedules bounded reloads but does not maintain a document authorization
   generation. Neither supplies the privileged IPC document contract proposed
   below. The existing `will-navigate` guard permits a development URL prefix
   or packaged `file:` URLs; that navigation policy is not sufficient IPC trust.

The boot hook, shared client, route identity, gateway store and backend session
contracts above were checked again on 13 September 2026 at
`main@1fec70ea48e90567284f7f811e2ca709c86a4a52`; those sources are unchanged
from the recorded source baseline. This was a source review. The native
rehearsal remains historical evidence from its stated baseline, and the new
acceptance cases below still require implementation and execution.

Open [103683](https://github.com/NousResearch/hermes-agent/pull/103683), by
laserguidedcake, was inspected at
`c8612a87358af55a980e4e532b5d99e7951a5ea9`.
Its [frame bridge](https://github.com/laserguidedcake/hermes-agent/blob/c8612a87358af55a980e4e532b5d99e7951a5ea9/apps/desktop/electron/ws-bridge.ts)
resolves headers in main, correlates dials, settles cancellation and retires
WebContents owned sockets. It addresses private CA trust and is relevant work
to reuse. It does not transfer the renderer client's JSON RPC or recovery
state. Its retirement behavior is appropriate to its present scope.

The bridge author's [subsequent feedback](https://github.com/NousResearch/hermes-agent/issues/108734#issuecomment-5643611728)
emphasizes that TLS trust and recovery ownership are separate requirements.
Whichever component owns recovery, its physical transport must support the
configured trust policy, including private certificate authorities. A renderer
reattachment protocol alone does not satisfy that requirement. Preserve the
bridge's trust handling when integrating it, and validate each supported host
with the actual TLS stack rather than inferring WebSocket trust from HTTP.

## Ownership options and recommendation

1. **Retain logical state in main, keep sockets in each renderer.** This changes
   fewer transport paths and preserves browser WebSocket behavior. It requires
   a protocol to transfer replay cursors and uncertain requests after abrupt
   renderer loss, plus fencing so old and new renderers cannot both dial or
   publish. State recorded only during cleanup is insufficient after a crash.
   Private CA connections still require the physical transport with trust
   support from 103683 or an equivalent supported bridge; retaining recovery
   metadata does not change Chromium WebSocket trust.
2. **Own the client and recovery in an asynchronous main service.** This keeps
   one retry policy and replay cursor set per window route across renderer
   replacement. It also requires a separate bounded operation ledger in main;
   retaining the client's pending map alone is insufficient. Typed IPC for
   attachment and updates adds serialization and memory costs in main.
3. **Own them in a supervised utility process.** This also isolates transport
   processing from the main event loop. It adds another process lifecycle and
   recovery boundary. A utility process crash still needs an explicit degraded
   reattachment contract.

Option 2 is the recommendation because it preserves the existing client state
without introducing a state handoff for every frame. Use the client's
`socketFactory` seam and build on the relevant transport work in 103683.
Do not add another localhost server or move synchronous runtime discovery into
the service. Revisit option 3 if measurements show that bounded event handling
still harms main responsiveness. Browser dashboard clients retain their direct
WebSocket path.

## Identity and attachment contract

Use the exact route model owned by 90149:
`connectionId`, `connectionGeneration` and `profile`. Preserve its runtime and
persistence profile distinctions on resource references. A route must never be
reconstructed from a URL, SSH target, `primary`, `lastUsed` or the active view.
Two registrations at one URL with different credentials remain different routes.
Backend replay epoch and socket attempt generation are separate coordinates.

Canonical registration generation propagation is a prerequisite for enabling
this protocol. Extend that canonical contract through descriptors, activation
and resource references; do not introduce a recovery specific route namespace.
Legacy descriptors stay explicitly unregistered and keep their existing path
until they can satisfy the canonical identity contract. An ambiguous legacy
descriptor cannot attach to a registered episode.

The main service stores an episode under its existing window ownership and
exact route. Window ownership is a consumer scope, not part of route identity.
The window can retain several profile routes just as background gateways do
today. Foreground changes select which state is displayed; they cannot retarget
an established resource or cancel another profile's episode.

The proposed privileged IPC must authorize the current trusted document on
every attachment, RPC, cancellation, acknowledgement and snapshot request.
Main resolves the window from `event.sender`; a supplied window ID grants no
authority. `event.senderFrame` must exist, equal that WebContents' current
`mainFrame`, and belong to the document generation main authorized for this
window. Subframes, detached frames and a valid handle presented by another
window are rejected before route lookup or retained data access.

Document trust compares `senderFrame.url` with the actual entry point main
loaded. In a packaged build, validate the canonical app entry file URL and its
allowed path, not merely the `file:` scheme or its opaque origin. In development,
require the exact configured development origin and allowed app entry path;
string prefix matching is
insufficient. Route fragments may change within that document. A remote gateway
URL, preview page, OAuth document or `data:` repair page never becomes a trusted
renderer origin for this recovery service.
These are requirements for the new service, not claims that the current IPC
handlers already enforce document trust.

Main owns a document generation as well as an attachment generation. At the
start of navigation that replaces the main document, and on
`render-process-gone`, it revokes the old document and its handles before any
replacement can attach. A same document route change does not revoke them.
Authorization resumes only after the expected app document has committed and
passed the frame and URL checks. Repeat those checks and generation fencing
after asynchronous work before dispatching an RPC or publishing its result.
Revocation blocks stale IPC and deliveries; it does not cancel backend effects
already sent. Those operations remain in the ledger for authorized recovery.

Main issues each attachment handle for that authorized window instance,
WebContents, document generation, exact route and monotonically increasing
attachment generation. Reloading within one WebContents replaces the document
and attachment generations. A secondary window may attach only to its own
authorized episodes; it cannot claim the primary window's handle or results.
Closing a window retires its ownership, so a recreated window cannot inherit
its episodes even if a numeric ID or route is reused.

The versioned attachment response carries the exact route, episode identity,
document and attachment generations, recovery state and a snapshot revision.
Updates and RPC results carry the same coordinates. Register the update channel
before returning the snapshot; queue updates until the view acknowledges that
revision.
Older revisions and revoked handles cannot publish. A late attach, dial or
snapshot result after route removal settles as stale and releases its resources.
Incompatible protocol versions return an explicit unsupported result and use
the existing boot path, without claiming established recovery continuity.

## Lifecycle and readiness

Readiness consists of four independent facts:

1. **Descriptor resolved:** main has a valid route and current transport inputs.
2. **Gateway reachable:** the WebSocket handshake completed and the expected
   gateway protocol responded on that socket attempt. Track these as separate
   stages below. A listening TCP forward or cached descriptor proves neither.
3. **Session attached:** the backend acknowledged the exact owned session and
   its current runtime binding.
4. **View hydrated:** the current attachment reconciled its transcript, running
   state and pending input with a valid snapshot and subsequent updates.

Main would own the following establishment state machine. These are proposed
service states; the current client resolves `connect()` on WebSocket open and
does not implement this handshake.

| State | Required transition evidence |
| :--- | :--- |
| `unestablished` | Main admits the exact route and creates its first boot budget. Descriptor resolution alone leaves it here. |
| `transport-open` | The current socket attempt completes its WebSocket handshake under the current registration generation. |
| `gateway-ready` | That attempt delivers a valid `gateway.ready` event, or a correlated response to the existing bounded `ping` probe. Preserve the current liveness compatibility rule that a JSON RPC method not found response proves a responsive older gateway. Transport open alone is insufficient. |
| `established` | Main accepts the authorized renderer's boot completion acknowledgement for this episode and the current readiness receipt. |

The renderer sends that acknowledgement at the successful completion boundary
currently calling `completeDesktopBoot()` and setting `bootCompleted` in
`useGatewayBoot`. For initial `boot()`, this means profile adoption has settled,
then the workspace seed, required config refresh and session list refresh have
settled, followed by the cancellation check. Preserve the existing fallbacks:
profile adoption can use its fallback, workspace seeding is best effort, and a
failed session list fetch leaves a usable empty sidebar. In `softSwitch()`,
preserve its own `ownsSwitch()` checks and best effort config and session refresh
semantics. Do not turn those tolerated failures into boot failures or require
a stored session. HMR adoption can reuse an established receipt only when main
already owns that episode; its local boolean cannot establish a new one.

Main issues the readiness receipt after `gateway-ready`, bound to the episode,
exact route, socket attempt, document generation and attachment generation.
It accepts the renderer acknowledgement only while all remain current and the
socket is still ready. This transition commits the establishment latch in main;
the renderer receives confirmation before treating establishment as retained.
An acknowledgement lost after that commit is reconciled by reading the same
episode. A missing, stale or duplicate acknowledgement cannot create a new
episode or reset its budget. These completion acknowledgements are distinct
from acknowledgements of hydration revisions.

Before that commit, losing transport readiness returns to `unestablished` with
the same budget; replacement during `transport-open` or `gateway-ready` inherits
all consumed attempts and remaining deadlines. A replacement must perform its
own boot completion work and acknowledge a receipt for its new document.
After the commit, establishment remains latched through transient transport
loss while live transport, session and view readiness may become unavailable.

A replacement renderer has no previous heap transcript. It may display stale
content only if a separately trusted persistent cache is available and validated
for the exact route, profile and stored session. This RFC introduces no such
cache. Without one, the replacement shows loading or stale status without a
transcript until authoritative hydration; retained replay watermarks do not
reconstruct missing content.

First boot retains the existing bounded budget and actionable failure state.
Replacing a renderer during first boot preserves the remaining budget instead
of minting another one. Once established, transient outages retain the existing
ongoing reconnect policy, bounded individual attempts, backoff and warning
behavior while the episode has a consumer or remains within detach grace.
Renderer replacement alone changes neither the episode nor its failure count.
Wake, online and manual retry signals are coalesced by the owner; only one dial
may be active for an episode.

Confirmed authentication rejection requires sign in; timeouts and transient
ticket failures remain connectivity failures. Each OAuth dial obtains a fresh
ticket. A successful handshake cannot reuse an earlier route generation's
credentials or authorize publication into a newer route.

Explicit disconnect cancels this window's intended route recovery and clears
its retained establishment state. It does not stop another window's client.
Editing a material registration field or removing it revokes that generation
in every affected window before closing its transports. A subsequent connection
is a new episode. Display name changes alone do not replace the route.

Application quit immediately cancels attempts, closes ports and sockets, and
releases existing backend leases. It adds no daemon or startup persistence.
Explicit Stop remains the existing session interrupt operation; renderer
detachment is not an implicit Stop. Existing local backend shutdown, detached
messaging gateway and server orphan policies remain their owners' decisions.

## Detach bounds and session continuity

Proposed starting limits for maintainer review are 60 seconds of renderer
absence per episode, 512 queued events and 4 MiB of serialized retained data per
episode, and 64 MiB of retained data across at most 64 service episodes. Allow
at most 128 outstanding requests per episode and 1024 across the service. The
byte limits include queued updates, hydration staging, operation ledger metadata
and retained request results; a single retained frame cannot exceed the episode
byte limit. Every ledger entry, including terminal and uncertain entries, counts
toward the same 128 per episode and 1024 service request entry limits. These
are proposed limits, not current Desktop behavior. Refuse additional admissions
with an explicit capacity result instead of evicting a live consumer. Use the
existing request deadlines and avoid an unbounded queue hidden inside IPC.

A renderer crash or reload starts grace only after its last attachment leaves.
Reattaching the authorized replacement cancels that timer. Intentional window
close retires that window's episodes immediately. A timer captures the episode
and attachment generation so an expired callback cannot close a new attachment.

During grace the owner may keep its existing transport and backend lease.
Grace expiry stops retries, closes the client and releases the lease even when
backend work remains. It does not issue `session.close` or interrupt shared work;
the actual socket close lets the backend apply its normal client absence policy.
An existing turn lease must not silently extend renderer grace forever. A later
reattach creates a fresh episode and visibly reconciles backend state.

Slow consumers do not delay other windows. Exceeding an event or byte limit
invalidates that consumer's hydration revision, drops its queued deltas and
requires resynchronization. If a snapshot exceeds the limit, return a stale
view requiring the existing bounded history loading path instead of buffering
the full transcript. Event coalescing may reduce cosmetic traffic; terminal
turn and approval changes cannot be silently discarded as successfully applied.

History is owned by the backend. Retained client memory is an optimization, not
a durable transcript. For a session previously observed through sequence 41,
reattachment must establish the current backend epoch and runtime binding before
using that watermark. A valid replay supplements a compatible hydrated cache.
It cannot turn a replacement renderer's empty cache into a complete snapshot.

Hydration requires an explicit ordering barrier between the backend snapshot
and subsequent events. Reuse `session.resume`, `session.history`, session info,
pending approval and clarify snapshots, and `session.events.since`. Extend their
existing contract with a snapshot revision or sequence boundary where needed;
do not pretend separate RPC reads are already atomic. For a valid boundary H,
the snapshot includes state through H and only events after H are applied as
deltas. Events arriving during hydration are held within the same byte limits.
No append only replay may duplicate content already included in the snapshot.

An epoch change, runtime rebinding, `truncated: true`, unavailable replay,
unprovable boundary or buffer overflow invalidates completeness. The view
remains stale and performs authoritative resynchronization before accepting
further deltas. An older backend without the boundary capability must use its
existing resume behavior and report degraded continuity. Approval and running
state require their current backend snapshots, including events without a
session sequence. Background recovery never steals the foreground selection.

## Requests whose outcome is uncertain

The service would own a bounded operation ledger separate from
`JsonRpcGatewayClient.pending`. Allocate its entry before handing any request
to the client, recording an operation ID, method and safety class, exact route
and episode, originating document and attachment generations, existing resource
identities, deadline, attempted send state, RPC correlation ID and any terminal
result. Classify methods explicitly as reads or mutations; an unclassified method
is a mutation. JSON RPC IDs correlate replies and never prove idempotency.

The implementation needs a narrow request lifecycle seam in the existing client
to record a send attempt before invoking the physical transport, distinguish a
proven failure before send, and observe responses and local settlement. A
rejected Promise or absence from `pending` is insufficient evidence. Ledger
updates run independently of renderer listeners and survive client timeout,
abort and `rejectAllPending()` cleanup within the episode's retention limits.
Keep correlation metadata in that ledger after pending entry deletion so a
late result can be reconciled without sending the request again.

| Outcome | Evidence retained by main |
| :--- | :--- |
| Never sent | Admission or dispatch ended before any transport send attempt, with positive evidence that no request crossed that boundary. |
| Pending | A send was attempted and its response deadline has not settled. Transport acceptance does not prove backend execution. |
| Completed | A correlated backend success response or authoritative operation reconciliation establishes the result. RPC completion is distinct from completion of any background work it started. |
| Failed | A correlated backend error establishes the reported failure. Any retry still follows that method's existing contract, not a generic transport policy. |
| Outcome uncertain | Sending may have occurred but no authoritative result is available after timeout, transport loss, local cancellation or an ambiguous send error. A sent mutation never becomes an ordinary retryable failure. |

A replacement attaches with its new authorized handle and may query operations from
earlier attachments of the same authorized window episode and exact route.
The originating generations remain audit metadata, not an equality requirement
against the replacement caller. All returned snapshots and results are fenced
to the current document and attachment; another window or route cannot obtain
them. Reattachment and repeated lookup never dispatch the operation again.

Pending entries retain their original deadlines. Terminal or uncertain entries
remain until the authorized current view acknowledges their outcome, or the
episode is retired by grace expiry, disconnect, route retirement or quit.
Capacity exhaustion rejects new admissions before send rather than silently
evicting unacknowledged entries. An acknowledged or retired entry may be released;
a later lookup returns `outcome unavailable`, never `never sent` or retryable.
Retention expiry, missing entries and main process restart provide no evidence
that a mutation did not execute. This ledger is bounded main memory, not a
durable transaction log.

If a prompt was accepted and its reply was lost, reconcile the owned stored
session, current runtime, history and live operation state. Do not resubmit
`prompt.submit`, an approval response or another side effect merely because the
renderer Promise disappeared. Preserve existing backend idempotency keys only
where the specific operation already supports them. A visible user message in
history alone does not prove that the requested work completed.

Where existing identities cannot establish the outcome, retain an explicit
uncertain result and give the user a way to inspect the session and decide the
next action. Recovery never automatically resubmits a mutation, including one
with an existing idempotency key.
Cancellation always settles local bookkeeping; it implies backend cancellation
only when the existing operation supports and acknowledges it.

## Implementation and acceptance evidence

Keep implementation in topical modules. Main composes the service and owns its
operation ledger, preload exposes scoped operations, the existing client supplies
RPC and replay through the required lifecycle seam, and the boot hook becomes
a consumer that acknowledges boot completion. Extract the relevant current policy
before adding behavior to the large hook or main facade. Coordinate any bridge
adaptation with 103683 and preserve its contributor provenance.

Implement in this order: canonical route and attachment fencing, retained
recovery ownership, hydration boundaries and uncertain request reconciliation,
then lifecycle integration tests. The chosen contract must be tested on the
unfixed baseline and implementation. A document merge cannot supply those
runtime receipts.

Use the real Electron, preload and renderer boundary through
[createSandbox and launchDesktop](../../apps/desktop/e2e/fixtures.ts), with
temporary Hermes home and Electron user data. Build the app before launching.
The existing [OAuth recovery rehearsal](../../apps/desktop/e2e/remote-oauth-recovery.spec.ts)
shows the real IPC pattern. A controllable gateway should count completed dial
attempts and hold or release handshakes deterministically. Do not substitute a
React remount for renderer replacement or assume a fixed outage duration.

The lifecycle comparison needs a generous explicit test timeout beyond the
suite's default where real dial deadlines require it. Deterministic unit tests
may inject time and sockets; native tests must retain real Electron processes.
Reproduction receipts record OS, commit, main PID, renderer replacement evidence,
attempt counts, route and generation, and gateway side effects.

1. Establish route A, make it unreachable and count failures past the initial
   boot budget. Compare an unchanged renderer with a replacement during the
   same outage, including replacement during a pending dial and after a
   replacement renderer has exhausted its boot retries. Main stays alive.
   Restore the same gateway and credentials. Both recover without another user
   retry; neither starts a new first boot episode. Separately prove that a
   route which never established still exhausts its bounded boot budget and
   repeated renderer replacement does not replenish that budget. Replace the
   document after WebSocket open, after gateway readiness, during config refresh
   and just before main receives the completion acknowledgement. Assert that
   only a current acknowledgement establishes the episode, including a reply
   lost after main commits it. Exercise each boot path's tolerated fetch errors.
2. Replace the renderer after observing sequence 41. Emit events before, during
   and after snapshot loading. Verify the final transcript and current running,
   approval and clarify state against the backend, without duplicate deltas.
   Repeat with backend restart, replay truncation and hydration buffer overflow;
   each requires explicit resynchronization. Use the existing
   [replay tests](../../apps/shared/src/json-rpc-gateway-replay.test.ts) for
   deterministic ordering and epoch cases.
3. Keep two windows and two profiles active with colliding session IDs. Reload
   and close one window while the other streams. Verify it cannot send through,
   cancel or hydrate from the other's attachment. Also use two registrations
   with the same URL and different header or credential envelopes. Only the
   exact authorized route may receive each request. Reject IPC from subframes,
   stale documents during navigation, untrusted packaged or development URLs,
   a secondary window holding the primary's handle, and a closed and recreated
   window. Use the real `senderFrame` boundary and assert no RPC dispatch or
   retained data disclosure on rejection.
4. Edit or remove a route during dial and during snapshot loading. Allow the
   obsolete work to finish late. Verify no activation or publication occurs,
   pending work settles once, listeners are released and a replacement route
   can connect. Include remove and recreate of the same registry ID.
5. Accept a prompt on the backend, drop its response and replace the renderer.
   Count backend submissions and resulting side effects. There is one
   submission, and reattachment reports the existing operation or an explicit
   uncertain outcome. Repeat for approval responses and explicit Stop, including
   transport close, timeout, abort and an ambiguous send error. Verify the
   authorized replacement can query the old attachment's operation without
   reissuing it, while another route or window cannot. Exhaust ledger capacity
   and expire retention; neither event may make a sent mutation retryable.
6. Keep a local TCP forward listening while its gateway is unavailable. Verify
   descriptor resolution does not publish transport or session readiness.
   Recovery succeeds only after the actual WebSocket and RPC path works.
7. Hold a renderer absent past grace, stall a consumer past its limits, and
   quit during an active dial. Verify bounded retained bytes and entries,
   settled work, released leases and closed sockets. Another window remains
   usable. Use a real temporary backend to verify viewer detach and orphan
   behavior rather than only observing a mocked close callback.
8. Rehearse token and OAuth routes, configured headers and TLS verification
   through the selected physical transport. Preserve fresh ticket minting,
   early event correlation and late open cancellation from bridge work.
   Run platform specific TLS cases on the actual supported hosts and retain
   browser dashboard direct WebSocket coverage.

Runtime diagnostics should use the existing local logging facilities and include
window scope, canonical route generation, episode, attachment generation, dial
attempt, readiness stage and resynchronization reason. Never log credentials,
ticket URLs, prompt contents or retained message payloads. No outbound telemetry
is introduced.

This proposal is ready for an ownership decision. Closure of 108734 additionally
requires the accepted contract to be implemented and the real boundary receipts
above to pass. It makes no claim about transcript loss, offline access to a
remote host, the AppHang cause in 103786 or backend group orchestration in 97681.
