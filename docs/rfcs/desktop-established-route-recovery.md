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
   one retry policy, pending request ledger and replay cursor set per window
   route across renderer replacement. It adds typed IPC for attachment and
   updates, and requires bounded serialization and memory use in main.
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

Main issues an attachment handle bound to the calling WebContents, authorized
window, exact route and a monotonically increasing attachment generation.
After renderer loss, main authorizes the replacement for that same window and
revokes the prior handle before attaching it. A renderer cannot select a window
by supplying its numeric ID. Reloading within one WebContents still increments
the attachment generation. A newly created window cannot claim another closed
window's episode.

The versioned attachment response carries the exact route, episode identity,
attachment generation, recovery state and a snapshot revision. Updates and RPC
results carry the same coordinates. Register the update channel before
returning the snapshot; queue updates until the view acknowledges that revision.
Older revisions and revoked handles cannot publish. A late attach, dial or
snapshot result after route removal settles as stale and releases its resources.
Incompatible protocol versions return an explicit unsupported result and use
the existing boot path, without claiming established recovery continuity.

## Lifecycle and readiness

Readiness consists of four independent facts:

1. **Descriptor resolved:** main has a valid route and current transport inputs.
2. **Transport connected:** the WebSocket handshake completed and the expected
   gateway protocol is usable. A listening TCP forward or a cached ready
   descriptor does not prove this fact.
3. **Session attached:** the backend acknowledged the exact owned session and
   its current runtime binding.
4. **View hydrated:** the current attachment reconciled its transcript, running
   state and pending input with a valid snapshot and subsequent updates.

An episode becomes established after the initial transport and gateway boot
RPC work succeed, at the semantic boundary currently setting `bootCompleted`.
Establishment must not require a session to exist: an empty but usable workspace
is valid. Session and view readiness remain separately observable. A replacement
renderer can display cached content as stale before it is hydrated.

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
byte limits include queued updates, hydration staging and retained request
results; a single retained frame cannot exceed the episode byte limit. These
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

The service owns request correlation while an attachment exists and during its
bounded replacement grace. A replacement view can learn that a request was
never sent, is pending, completed, failed, or has an unknown outcome. Every entry
includes its exact route and existing resource or operation identity. JSON RPC
request IDs correlate replies; they do not prove backend idempotency.

If a prompt was accepted and its reply was lost, reconcile the owned stored
session, current runtime, history and live operation state. Do not resubmit
`prompt.submit`, an approval response or another side effect merely because the
renderer Promise disappeared. Preserve existing backend idempotency keys only
where the specific operation already supports them. A visible user message in
history alone does not prove that the requested work completed.

Where existing identities cannot establish the outcome, retain an explicit
unknown result and give the user a way to inspect the session and decide the
next action. Do not introduce generic automatic retries for mutations.
Cancellation always settles local bookkeeping; it implies backend cancellation
only when the existing operation supports and acknowledges it.

## Implementation and acceptance evidence

Keep implementation in topical modules. Main composes the service, preload
exposes scoped operations, the existing client supplies RPC and replay, and
the boot hook becomes a lifecycle consumer. Extract the relevant current policy
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
   repeated renderer replacement does not replenish that budget.
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
   exact authorized route may receive each request.
4. Edit or remove a route during dial and during snapshot loading. Allow the
   obsolete work to finish late. Verify no activation or publication occurs,
   pending work settles once, listeners are released and a replacement route
   can connect. Include remove and recreate of the same registry ID.
5. Accept a prompt on the backend, drop its response and replace the renderer.
   Count backend submissions and resulting side effects. There is one
   submission, and reattachment reports the existing operation or an explicit
   unknown outcome. Repeat for approval responses and explicit Stop.
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
