# Relay ↔ Connector Contract (v1, EXPERIMENTAL)

> **Status:** EXPERIMENTAL. This contract MAY CHANGE without a deprecation
> cycle until at least two real Class-1 platforms (Discord + Telegram) have
> validated it. Evolution during the experimental phase is **additive-only**,
> gated by `contract_version`. A breaking change updates both repos in lockstep.

This document is the formal interface between the **Hermes gateway** (Python,
`gateway/relay/`) and the **connector** (Node/TypeScript,
`NousResearch/gateway-gateway`). The connector implementer's first action is to
read this file.

The gateway runs a generic `RelayAdapter` that dials **out** to the connector,
receives a `CapabilityDescriptor` at handshake, then exchanges normalized
`MessageEvent`s (inbound) and actions (outbound) over a per-turn bidirectional
WebSocket. The gateway never learns which concrete platform is fronting it; the
connector owns all platform-specific socket/identity logic.

---

## 1. Handshake

1. Gateway opens the transport (`connect`).
2. Gateway calls `handshake()`; connector returns a `CapabilityDescriptor`
   (section 2) describing the platform this adapter instance fronts.
3. Gateway configures the adapter from the descriptor (char limit, length unit,
   draft/edit/thread/markdown capabilities) and registers an inbound handler.
4. Connector then streams inbound events and accepts outbound actions.

`contract_version` (currently `1`) is carried in the descriptor. The gateway
ignores unknown descriptor fields (forward-compat) and fills missing optional
fields from defaults.

---

## 2. CapabilityDescriptor (handshake payload)

JSON object. Source of truth: `gateway/relay/descriptor.py`.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `contract_version` | int | yes | Contract version (additive-only within a version). |
| `platform` | string | yes | Platform name (e.g. `"discord"`, `"telegram"`). |
| `label` | string | yes | Human-readable label. |
| `max_message_length` | int | yes | Char limit; gateway exposes as `MAX_MESSAGE_LENGTH`. 0 → treat as 4096. |
| `supports_draft_streaming` | bool | yes | Native draft-streaming preview support. |
| `supports_edit` | bool | yes | Edit-based streaming possible; if false, consumer degrades to one-message-per-segment. |
| `supports_threads` | bool | yes | `create_handoff_thread` capability. |
| `markdown_dialect` | string | yes | `"plain"`, `"markdown_v2"`, `"discord"`, … (drives `supports_code_blocks`). |
| `len_unit` | string | yes | `"chars"` (builtin len) or `"utf16"` (Telegram UTF-16 code units). |
| `emoji` | string | no | Display emoji (default 🔌). |
| `platform_hint` | string | no | System-prompt platform hint. |
| `pii_safe` | bool | no | Redact PII in session descriptions. |
| `supports_context` | bool | no | Whether the connector can supply surrounding channel/group **context** for an addressed turn on this platform (Model A on-demand history fetch — Discord/Slack/Matrix; Model B passive buffer — Telegram/Signal/WhatsApp). Default false ⇒ no `context` is attached to inbound events. See §3. |

Most fields are a projection of the gateway's existing `PlatformEntry`; the
runtime-only fields (`len_unit`, `supports_*`, `markdown_dialect`) come from the
live platform adapter's capability methods.

---

## 3. Inbound: `MessageEvent` envelope

The connector normalizes each platform wire event into a `MessageEvent`
(`gateway/platforms/base.py`) and delivers it to the gateway. **Inbound is
delivered over the gateway's OUTBOUND `/relay` WebSocket** (see the transport
note below) — the connector pushes an `inbound` frame down the socket the
gateway already dialed. The gateway keys the session via `build_session_key()`
from the embedded `SessionSource` — so populating the right discriminators is
the single highest-correctness responsibility of the connector.

### Inbound transport (WS back-channel, not HTTP)

The gateway dials **out** to the connector's `/relay` WebSocket for the
handshake + outbound actions (§4) + its own `/stop` egress (§5). Inbound rides
the **same socket** in the other direction: the connector pushes an `inbound`
frame (and `interrupt_inbound` for §5) down the gateway's outbound WS. There is
**no gateway-side inbound HTTP endpoint** — a gateway need not (and, when hosted,
cannot) expose any inbound port; everything flows over the connection it
initiated.

**Multi-instance routing.** The connector instance that owns a platform's socket
(and thus produces inbound events) is generally **not** the instance the gateway
dialed its outbound WS into. The producing instance therefore publishes the
event on the connector's internal **relay bus** (Redis pub/sub; `RelayBus` in
`src/core/relayBus.ts`) keyed by tenant. Every connector instance subscribes and
routes each message to its **local** sessions for that tenant
(`RelayServer.routeBusMessage`); the single instance that actually holds the
gateway's socket delivers it, and instances with no local session for the tenant
no-op. Cross-instance delivery is thus an in-cluster Redis hop, not a public
HTTP call.

Frames (connector → gateway, over the WS):

- `{"type":"inbound", "event": <MessageEvent>, "bufferId"?}`
- `{"type":"interrupt_inbound", "session_key", "chat_id"}` (§5)
- `{"type":"passthrough_forward", "forward": <PassthroughForward>, "bufferId"?}` (§5.1)

**Channel context on inbound (design relay-channel-context).** When the source
platform's descriptor advertised `supports_context` (§2) and the chat is
multi-party (`chat_type` ∈ group/channel/thread/forum, never `dm`), the
connector MAY attach two optional, additive fields to the inbound `MessageEvent`:

- `context`: an array of read-only surrounding messages (same channel, oldest→
  newest) — nearby non-addressed chatter the connector fetched (Model A) or
  buffered (Model B). REFERENCE ONLY: it never triggers the agent (the trigger
  decision was already made connector-side on the addressed event alone). The
  gateway renders it into `MessageEvent.channel_context` (the same read-only
  injection path history-backfill uses).
- `context_error`: bool, true when the platform is context-capable but the
  fetch/buffer failed and the connector fail-opened to an empty `context`
  (observability marker; surfaced connector-side via the delivery span).

Both absent ⇒ byte-identical to today. A connector that never sends them, or a
`dm`, or a no-context platform, yields no `channel_context`.

`PassthroughForward` is the wire form of a forwarded passthrough-plane request
(Class-2/3 webhooks — Discord interactions, Twilio): `{platform, botId, method,
path, headers: [[k,v],…], bodyB64}`. The body is base64-encoded so arbitrary
bytes survive the newline-delimited-JSON transport; the gateway base64-decodes
back to the exact bytes the connector forwarded (the connector already verified
the provider signature and stripped any shared-identity credential at the edge —
§6 — so the gateway re-processes a sanitized, token-free body and acts on it via
the token-less `follow_up` path). See §3.1.

**Trust.** The WS upgrade is authenticated with the gateway's per-gateway secret
(§6.1), so the channel is trusted end to end — inbound frames are not separately
HMAC-signed (the authenticated socket subsumes the per-delivery origin proof the
old HTTP path needed). The relay-bus hop is inside the connector trust domain
(same as the lease/buffer/capability stores).

> Earlier drafts of this contract delivered inbound over a signed **HTTP POST**
> to a `gatewayEndpoint` (`HttpGatewayDelivery` + a gateway-side
> `inbound_receiver`), HMAC-signed with a per-tenant delivery key. That required
> every gateway to expose a reachable inbound URL — impossible for hosted
> gateways, which have no public IP. The WS back-channel above replaces it; the
> per-tenant delivery key is retained at provision for forward-compat but is no
> longer used for inbound. The **passthrough plane** (Class-2/3 webhooks like
> Discord interactions / Twilio) historically still used `gatewayEndpoint` for
> its post-ACK forward; Phase 5 §5.1 moves that forward onto the WS too (the
> `passthrough_forward` frame above), so a hosted gateway needs zero public
> inbound surface and `gatewayEndpoint` is retired once the cutover lands.

### 3.1 Passthrough-plane forward (§5.1)

The passthrough plane answers the provider's latency-critical ACK at the
connector EDGE (e.g. Discord's deferred interaction response within ~3s), then
does a **fire-and-forget** forward of the real request to the gateway. That
forward needs no response back (the provider was already satisfied), so it rides
the same outbound WS as `inbound` via a `passthrough_forward` frame rather than
an HTTP POST. The gateway processes the decoded request through its normal agent
path (a Discord interaction is decoded to a `MessageEvent` and handled like a
message; the reply egresses over the outbound / `follow_up` path). `bufferId` is
present when the forward was buffered (Phase 5 §5.3 buffered-only flip) and the
gateway acks it after durable handoff.



### SessionSource fields (the wire surface)

Source of truth: `SessionSource.to_dict()` in `gateway/session.py`. These are
every key the gateway accepts on the wire. `platform`, `chat_id`, `chat_type`,
`user_id`, `user_name`, `thread_id`, `chat_name`, and `chat_topic` are always
present (may be `null`); the rest are included only when set.

| Field | Type | Always sent | Meaning |
| --- | --- | --- | --- |
| `platform` | string | yes | Platform name (matches the descriptor's `platform`). |
| `chat_id` | string | yes | Primary conversation id (channel/chat). Session-key discriminator. |
| `chat_type` | string | yes | `dm` / `group` / `channel` / `thread` / `forum`. |
| `chat_name` | string\|null | yes | Human-readable chat name. |
| `user_id` | string\|null | yes | Message author id. Session-key discriminator. |
| `user_name` | string\|null | yes | Author display name. |
| `thread_id` | string\|null | yes | Thread/forum-topic id when in a thread. Session-key discriminator. |
| `chat_topic` | string\|null | yes | Channel topic/description (Discord, Slack). |
| `user_id_alt` | string | no | Platform-specific stable alt id (Signal UUID, Feishu union_id). |
| `chat_id_alt` | string | no | Alternate chat id (e.g. Signal group internal id). |
| `scope_id` | string | no | Platform-neutral **scope** discriminator: Discord guild / Slack workspace / Matrix server. **REQUIRED for Discord/Slack scope isolation.** Session-key discriminator. (Canonical name as of the D-Q2.5 wire migration.) |
| `guild_id` | string | no | **Legacy alias, no longer read by the connector.** As of D-Q2.5c the connector reads and writes only `scope_id`; the gateway's agent-wide `SessionSource.to_dict()` still emits `guild_id` (mirrored to `scope_id`) for non-relay session persistence, so it may still appear on the wire but the connector ignores it. Do not depend on it. |
| `parent_chat_id` | string | no | Parent channel when `chat_id` refers to a thread. |
| `message_id` | string | no | Id of the triggering message (for pin/reply/react). |

> `is_bot` (author-is-a-bot/webhook classification) exists on the gateway-side
> dataclass but is **intentionally NOT on the wire** in v1 — it is not part of
> `to_dict()`. Do not add it to the connector's `SessionSource` until it is
> first added here and to `to_dict()` (additive bump).

### SessionSource discriminators per platform

| Platform | chat_id | chat_type | user_id | thread_id | scope_id |
| --- | --- | --- | --- | --- | --- |
| **Discord** | channel id | `dm`/`group`/`thread` | author id | thread channel id (threads) | **guild id** (REQUIRED for server isolation) |
| **Telegram** | chat id | `dm`/`group`/`forum` | from id | forum topic id (forums) | — |

**Get Discord's `guild_id` wrong and two servers collide into one session.**
This is the #1 High-severity risk. The gateway's `build_session_key()` is the
conformance oracle: for a given `SessionSource`, the connector's normalization
must produce the same key the Python adapter would. (The Phase-1 stub tests
assert known-input → known-key.)

### Bot identity vs tenant (single-bot consolidation, Appendix A)

The envelope carries the **originating bot identity** as a field **distinct from
tenant**. Tenant is resolved from the event's own discriminator (Discord
`guild_id`, Telegram `chat_id`, webhook path/subdomain) — **never** from which
token/socket/process delivered it. This keeps one shared bot able to front many
tenants (Phase 6) without overloading an existing field.

### Author-first resolution + the account-link (DM) path (Phase 7)

Phase 7 adds **self-serve, per-user onboarding to a shared bot**, which changes
*which* discriminator resolves the instance for a routed inbound message — and
adds a management path for users to bind their own account.

**Author-first resolution (the multi-tenant-guild rule, D-7.2).** A single
Discord guild may hold **many** tenants — different members each linked to their
own agent. So for delivery the connector resolves the destination instance from
the **authenticated author binding** (`user_instance_binding`, keyed by
`(tenant, platform, platform_user_id)` via `resolveByUser`), **NOT** by a
guild→instance route. Concretely:

- A routed message authored by a **linked** user reaches **only that user's**
  instance — even when a second linked user in the **same guild** is served by a
  different instance (each reaches only their own).
- A message authored by an **unlinked** user resolves to **no** instance and is
  dropped (**fail-closed** — never broadcast to the guild's other tenants).
- The author id used is the **authentic `user_id` off the observed event**, the
  same `SessionSource.user_id` documented above — never a value asserted by a
  gateway or carried in a management frame.

This is the per-`user_id` owner-only routing the connector enforces in
`WsGatewayDelivery` (the gateway-side multi-tenant-guild E2E driver
`gateway_multitenant_guild_driver.py` is the cross-repo oracle).

**The account-link (DM) path.** A user binds their account to an instance with a
one-time code, redeemed by DMing the shared bot:

1. The owner triggers a link from the Portal (or a self-hosted CLI). The
   connector mints a short-lived **link code** for the **authenticated**
   instance (`POST /manage/link`; instanceId comes from the caller's principal —
   a NAS-signed `aud=agent:{instanceId}` token or the instance's own per-gateway
   secret — **never** the request body).
2. The user sends `/link <code>` as a **direct message** to the shared bot from
   the account they want to bind.
3. The connector's inbound observer **consumes** that DM (it is not routed to any
   agent) and writes the `user_instance_binding` using the **authentic
   `user_id`** off the observed DM event. From then on, author-first resolution
   routes that user's messages to the bound instance.

**Opt-out is connector-authoritative.** Deprovisioning an instance
(`POST /manage/deprovision`) drops its author bindings (so its users stop
resolving to it) **and** revokes its per-gateway secret (so its socket can no
longer authenticate — the next WS upgrade is closed **4401**). A gateway that
sees a **4401 close after a previously-successful handshake** treats it as a
terminal revocation: it stops reconnecting and reports the relay platform as
**disabled** (not a retryable error). A 4401 *before* any successful handshake
stays retryable (a cold-start / not-yet-provisioned race, not a revocation).

### 3.2 Going-idle / buffered-flip primitive (§5.3)

A scale-to-zero PRIMITIVE (not the behaviour — nothing here decides to sleep or
suspends a machine; a later workstream consumes these frames). It lets a gateway
enter a drain/idle transition without losing inbound that arrives while it is
gone, by making the connector buffer for that instance and replay on reconnect.

Three frames (all keyed by the connection's **authenticated** per-instance id —
read off the stored secret record at the WS upgrade, never asserted in a frame):

- `{"type":"going_idle"}` (gateway → connector) — emitted as part of the
  gateway's EXISTING drain transition (the adapter sends it before tearing down
  the socket). Asks the connector to flip this instance to **buffered-only**.
- `{"type":"going_idle_ack"}` (connector → gateway) — the connector has flipped:
  live delivery has stopped and subsequent inbound for this instance buffers
  durably. The gateway **stays serving until this ack** (so an event landing in
  the flip window is delivered live, not lost — the same SUBSCRIBE-before-serve
  ordering discipline as the bus). Only after the ack is it safe to close.
- `{"type":"inbound_ack", "bufferId"}` (gateway → connector) — durable receipt of
  a buffered `inbound` delivery (which carries its `bufferId`) replayed on
  reconnect. The connector acks the buffer entry only after this, giving
  drain-without-dup on the **delivery leg**: an instance that dies mid-drain
  redelivers exactly the unacked tail; an acked entry never redelivers.

**Buffer + drain.** While flipped, the connector appends inbound to a durable
per-instance delivery-leg buffer (`delivery:<instanceId>`) instead of pushing it
live. On the gateway's **reconnect** (a NET-NEW reconnect loop re-dials +
re-handshakes after an unexpected close), the new handshake triggers the
connector to drain that backlog over the new socket **in order, ack-gated**,
then clear the flip so live delivery resumes. This reuses the same
`drainWithoutDup` machinery as the Discord→connector ingest leg, applied to the
connector→gateway delivery leg. Connector-authoritative throughout: a gateway can
only flip/drain ITS OWN instance.

> NOT in scope (deferred behaviour): the autonomous idle timer that DECIDES to
> drain, the actual machine suspend, and the NAS suspended-health model. The
> primitive is "when the gateway drains, relay flips to buffered + replays on
> reconnect, with no loss/dup"; WHAT triggers the drain is out of scope.

### 3.3 Wake poke (§5.2)

The other half of the sleep/wake loop: how a SUSPENDED gateway finds out it has
buffered work waiting. A PRIMITIVE — nothing here suspends a machine; it wires
the wake SIGNAL so a future scale-to-zero behaviour layer can rely on "buffered
⇒ wake poked."

- **Registration.** The gateway registers a **wake URL** at enroll/provision —
  any reachable URL the connector can GET to wake it (a Fly autostart hostname,
  a dashboard host). Self-hosted: `hermes gateway enroll --wake-url <url>` (or
  `GATEWAY_RELAY_WAKE_URL` / `gateway.relay_wake_url`). Managed/NAS: stamped into
  the container env beside `GATEWAY_RELAY_URL`. Forwarded in the
  `/relay/provision` body as `wakeUrl` and stored per-instance on the connector's
  secret record (gateway-asserted but safely scoped — same posture as
  `instanceId`; the org/tenant stays token-verified, so a gateway can only
  register a wake target for ITS OWN instance). DISTINCT from the retired
  `gatewayEndpoint`: a **poke target**, not a delivery target.
- **The poke.** When a buffered-only (going-idle) destination receives its FIRST
  buffered event, the connector issues a **payload-free, unsigned GET** to that
  instance's registered `wakeUrl`, **directly** (NOT NAS-mediated — relay stays
  NAS-independent). It carries no tenant data and no inbound: it only says "you
  have buffered work, reconnect." Tenant authority is re-established the normal
  way when the gateway re-dials (the authenticated WS upgrade), so a leaked/
  guessed wake URL can at worst cause a spurious reconnect of ITS OWN instance.
  Rate-limited per instance (one poke per cooldown window, not per event), and
  best-effort — a failed poke is swallowed; the gateway still drains whenever it
  next reconnects on its own. No new frame: the wake is an out-of-band HTTP GET,
  not a relay-WS message (the socket is down — that's the whole point).

> NOT in scope (deferred behaviour): the actual machine suspend (Fly
> `autostop:"suspend"`) and the autonomous idle timer that decides to sleep. The
> primitive is "buffered event for a sleeping instance ⇒ its wakeUrl gets poked";
> WHAT makes the instance sleep (and wake-to-serve) is the behaviour layer.

### 3.4 Obligations on a future scale-to-zero behaviour layer

§3.2 and §3.3 ship the **primitives**; this section is the **contract a separate
scale-to-zero behaviour workstream must honour to consume them safely.** It owns
the *decision* to suspend, the actual machine suspend, and the platform/health
model — none of which live here — but it MUST hold these guarantees, which the
primitives assume:

1. **Register a `wakeUrl` before the instance can ever be suspended.** A
   suspended instance with no registered `wakeUrl` is a black hole — buffered
   inbound never triggers a poke, so it sleeps through its own traffic until
   something else reconnects it. The behaviour layer MUST ensure a reachable
   wake target is registered (self-hosted: `--wake-url`; managed: stamped) as a
   precondition of allowing suspend. A wake URL that is unreachable while the
   machine is suspended (e.g. points at the suspended machine itself with no
   platform autostart in front) is equivalent to none.
2. **Drain through `going_idle` → await `going_idle_ack` BEFORE tearing down the
   socket or suspending.** Never suspend with an un-acked flip in flight. The
   ack is the connector's confirmation that delivery for this instance is now
   buffered-only; a machine that suspends after sending `going_idle` but before
   the ack can drop the inbound that races the flip. The gateway already gates
   socket teardown on the ack (Q-5.3c); the suspend step MUST sit *after* a
   clean drain completes, not race it.
3. **Keep the NET-NEW reconnect loop live as a precondition of suspend.** The
   wake→drain contract is "poke ⇒ the gateway re-dials ⇒ the connector drains on
   the reconnect handshake." If the reconnect loop is disabled, a poke lands on a
   machine that never re-dials and the buffer strands. The behaviour layer must
   not suspend an instance whose relay transport won't reconnect on wake.
4. **Treat suspended ≠ down in the health model (Q-5.3b).** A suspended instance
   is healthy-asleep, not failed. The health/monitoring layer MUST distinguish
   the two (e.g. via the platform machine-state) so a suspended instance is not
   restarted, alerted on, or reaped as unhealthy — that would defeat the suspend
   and can race the wake/drain.
5. **The wake poke is best-effort and rate-limited — do not assume exactly-once
   or immediate wake.** At most one poke per cooldown window per instance, and a
   failed poke is swallowed. The behaviour layer must not rely on the poke as a
   guaranteed/prompt signal; correctness still rests on "the gateway drains
   whenever it next reconnects." A belt-and-suspenders wake (e.g. a scheduled
   job that also reconnects) is the behaviour layer's call, not the primitive's.
6. **Suspend only when genuinely idle — and idle is connector-observable, not
   gateway-guessed.** WHAT counts as idle (no in-flight turn + no inbound for N
   min) is the behaviour layer's policy, but it must compose with the existing
   drain machinery (`gateway_state` running→draining) rather than introduce a
   parallel relay-only idle path — the same integration constraint §3.2 places
   on `going_idle`.

These are guarantees the behaviour layer OWES the primitives; the primitives owe
the behaviour layer only what §3.2/§3.3 already specify (a flip-on-going_idle,
a durable per-instance buffer + ack-gated reconnect drain, and a poke on the
first buffered event for a flipped instance).

---

## 4. Outbound: action set

The gateway calls the transport with action dicts. Source of truth:
`gateway/relay/transport.py` + `gateway/relay/adapter.py`.

| `op` | Fields | Result |
| --- | --- | --- |
| `send` | `chat_id`, `content`, `reply_to?`, `metadata?` | `{success: bool, message_id?, error?}` |
| `edit` | `chat_id`, `message_id`, `content`, `metadata?` | `{success: bool, error?}` |
| `typing` | `chat_id` | `{success: bool}` |
| `follow_up` | `session_key`, `kind`, `content`, `metadata?` | `{success: bool, message_id?, error?}` |

`get_chat_info(chat_id)` is a separate proxied call returning at least
`{name, type}`. Media actions follow the same envelope shape (deferred to a
later contract revision; additive).

**`follow_up` (A2 capability action).** Some inbound payloads carry a credential
that acts on the **shared** bot identity (e.g. a Discord interaction follow-up
token). Per §6 the connector strips that at the edge and binds it in its
capability vault keyed by the session; it **never reaches the gateway**. To use
it, the gateway issues `follow_up` naming the **session it is already in**
(`session_key`) plus the capability `kind` (e.g. `discord.interaction_token`) —
**never a token**. The connector resolves the real value from its vault,
enforces the tenant match (tenant B can never wield tenant A's capability), and
egresses. `success: false` when the capability is absent/expired or the tenant
doesn't match — the gateway has nothing to retry with, by design (a leaked
gateway holds zero capability material). Source of truth:
`gateway/relay/transport.py` (`send_follow_up`) + `gateway/relay/adapter.py`.

---

## 5. Interrupt (`/stop`) routing

- **Gateway → connector:** `send_interrupt(session_key, reason?)` egresses a
  mid-turn `/stop` over the outbound WS. The connector MUST forward it to the
  gateway instance running that `session_key` (the routing invariant).
- **Connector → gateway:** an inbound interrupt for a `session_key` is delivered
  as an `interrupt_inbound` frame down the gateway's outbound WS (§3 transport
  note) — routed cross-instance via the relay bus to whichever instance holds
  the socket — and bridged by the adapter's `on_interrupt(session_key, chat_id)`
  into the existing per-session interrupt mechanism, cancelling exactly that turn
  (siblings untouched).

Both directions ride the gateway's outbound WS: the gateway→connector `/stop`
egresses over it, and the connector→gateway interrupt rides the same `inbound`
back-channel as a normalized event.

---

## 6. Trust boundary & signed-body handling (A2)

**The connector is the sole crypto/identity boundary. The gateway re-validates
nothing.**

Webhook signatures (Discord ed25519, Twilio HMAC, WeCom BizMsgCrypt) are
computed over exact raw bytes, and some payloads are *encrypted* with a shared
secret. The connector fronts a **shared** bot for many tenants and holds every
tenant's platform secrets, so it:

- **verifies / decrypts at the edge** (the only place the secrets live),
- **normalizes** the payload into a tenant-scoped `MessageEvent` (§3),
- **strips any shared-identity capability** out of the payload and binds it in
  its capability vault, keyed by the session (see §4 `follow_up`),
- **forwards only the sanitized `MessageEvent`** — never the raw signed body.

The gateway therefore performs **no** platform signature/crypto verification on
the relay path; it trusts the normalized event. This is an enforced invariant on
the gateway side (`tests/gateway/relay/test_relay_sheds_crypto.py`: the relay
package imports/calls no platform-crypto).

**Why not "forward the signed body byte-for-byte so the gateway re-validates"?**
That earlier model is incoherent under an untrusted, disposable tenant gateway:

- Re-validating Twilio HMAC / WeCom crypto would require handing the gateway the
  **shared signing secret** — which is itself the leak, and on a shared bot it's
  a *cross-tenant* leak.
- WeCom payloads are encrypted with the shared secret; the connector must decrypt
  at the edge just to route, so forwarding ciphertext would again require giving
  the gateway the secret.
- A Discord interaction token lives **inside** the signed JSON body — you cannot
  both preserve the bytes and strip the credential; they are the same bytes.

So byte-preservation is abandoned deliberately: the connector re-serializes the
sanitized event and the gateway trusts it. This also unifies the passthrough and
relay planes — both are "verify at the edge → emit a normalized event," differing
only in transport. See `docs/capability-trust-boundary.md` (connector repo:
`gateway-gateway`) for the full A2 rationale and the connector-side vault.

### 6.1 Channel authentication (the connector⇄gateway link itself)

A2 makes the connector the sole holder of platform secrets while the gateway may
be **customer-managed and internet-exposed**, so the connector⇄gateway channel
is itself authenticated. The gateway holds an enrollment- or provision-issued
**per-gateway secret** (`hermes gateway enroll` → connector `/relay/enroll`, or
managed self-provision → `/relay/provision`) that authenticates its outbound WS
upgrade. It is an HMAC-SHA256 scheme with a multi-secret rotation verify list
(gateway side: `gateway/relay/auth.py`; connector side:
`src/core/relayAuthToken.ts`).

| Leg | Credential | Mechanism |
|-----|-----------|-----------|
| Gateway → connector WS upgrade | per-gateway secret | An `Authorization` bearer header on the `/relay` upgrade. The token is `base64url(payload:exp:sig)` where `payload = gatewayId` and `sig = HMAC(payload:exp, secret)`. Connector verifies and rejects the upgrade (**close 4401**) on mismatch/absence/revocation. The authenticated tenant comes from the connector's store, never the `hello` frame. |
| Connector → gateway inbound (`inbound` / `interrupt_inbound` frames) | — (rides the authenticated WS) | Inbound is pushed down the gateway's already-authenticated outbound socket (§3), so no per-message signature is needed. A **per-tenant delivery key** is still issued at enroll/provision and retained for forward-compat, but is no longer used to sign inbound. |

This is the **channel** authenticator — distinct from platform crypto, which the
relay path still sheds entirely (§6). The gateway holds zero platform secrets;
the per-gateway secret authenticates only the connector link. Full threat model +
enrollment/rotation/kill-switch design: `docs/connector-gateway-auth-design.md`
(connector repo).

---

## 7. Per-instance delivery & the management plane (Phase 6)

Phases 1–5 treat the connector as a single-tenant front: inbound events for a
tenant fan out to that tenant's gateway socket(s). **Phase 6 makes delivery
per-INSTANCE** — a shared bot can front many users/agents in one tenant (one
Discord guild, one Telegram bot) without cross-delivery — and adds a small
**management plane** the agent (or a managed Portal) uses to declare who-sees-what
and what's-relevant. All of this lives **connector-side**; the gateway's only new
responsibility is to **declare its relevance policy** at boot (§7.3).

### 7.1 The delivery gate (connector-side, informational)

For each inbound event the connector decides which instances receive it by
composing three AND-ed filters. The gateway does not implement these — they run
in the connector — but they define the delivery semantics the gateway relies on:

| Layer | Question | Source of truth |
| --- | --- | --- |
| **owner / scope ∧ principal** | May this instance *see* this author here? | per-user `user_id → instance` bindings (the owner floor) + per-instance `(guild, channel)` scope grants + an `owner-only` / `allow-list` / `any` principal policy. |
| **visibility floor** | Can the instance's bound owner actually `VIEW_CHANNEL` this in Discord? | live Discord ACL (effective permissions), fail-closed. Narrows an over-broad scope grant downward. |
| **relevance** | *Given* it may see it, should the agent engage? | the relevance policy declared in §7.3 (address-gating / free-response / allow-bots). |

The composition only ever **narrows** delivery (`deliver ⇔ authorized ∧ visible
∧ relevant`); the **owner floor bypasses the relevance layer** (an author's own
message always reaches their own instance — you don't @mention your own agent).
A message authored by an unbound user reaches no instance (fail-closed). The
full design + invariants live in the connector repo
(`NousResearch/gateway-gateway`); this section is the gateway-facing summary.

### 7.2 Management routes (connector-side, authenticated)

The connector mounts authenticated management routes. They share the **same
dual-auth** as the WS upgrade: either a managed NAS-signed `aud=agent:{instanceId}`
RS256 JWT, **or** the gateway's own per-gateway secret bearer (§6.1
`make_upgrade_token`). In both cases the connector resolves the authoritative
`{tenant, instanceId}` from its **stored** record — **never** from the request
body (a body-asserted `instanceId` is ignored).

| Route | Purpose |
| --- | --- |
| `POST /manage/link` | Issue a short-lived code to bind a platform account to the authenticated instance (the `/link <code>` flow; the connector reads the authentic `user_id` off the inbound event). |
| `POST /manage/scope`, `/manage/scope/release` | Claim / release a `(guild, channel)` scope for the authenticated instance. A channel is owned by at most one instance (non-overlap is a PK constraint). |
| `POST /manage/principal` | Set the instance's principal policy (`owner-only` \| `allow-list` \| `any`). |
| `POST /manage/dm-default` | Set the user's DM-default instance (DM tie-break when a user linked more than one). |
| `POST /relay/policy` | Declare the instance's **relevance policy** (§7.3). |

These are connector-owned (the management plane is not part of the gateway's
agent path); the gateway only calls `POST /relay/policy` (§7.3). The others are
driven by the managed Portal / `hermes` CLI.

### 7.3 Relevance-policy declaration (the gateway's responsibility)

The relevance layer (§7.1) is the per-tenant parity for the gateway's own
behaviour knobs (`require_mention`, `free_response_channels`,
`{PLATFORM}_ALLOW_BOTS`). So the **same** behaviour governs relay delivery, the
gateway projects those knobs into a **platform-agnostic** policy and POSTs it to
`POST /relay/policy` at boot (after its per-gateway secret is resolved).

Body (`gateway/relay/__init__.py` `relay_relevance_policy()` → `send_relay_policy()`):

| Field | Type | Projected from | Meaning |
| --- | --- | --- | --- |
| `platform` | string | the fronted platform (`relay_platform_identity`) | which platform this policy applies to. |
| `requireAddress` | bool | `require_mention` | a non-owner message must @mention / reply-to the bot to be relevant. |
| `freeResponseScopes` | string[] | `free_response_channels` | scope (channel) ids where `requireAddress` is waived. Same scope vocabulary as §7.1's scope grants. |
| `allowOtherBots` | bool | `{PLATFORM}_ALLOW_BOTS ∈ {mentions, all}` | admit bot-authored messages (default off). |

Auth is the per-gateway upgrade token (§6.1), so the connector attaches the
policy to the authenticated instance. The gateway is the **source of truth** and
re-declares **every boot** (a full replace, mirroring the `routeKeys` upsert at
provision — self-healing). When the projected policy is all-default the gateway
sends nothing (the connector's absent-row default already matches). The POST is
**fail-soft**: a failure logs and boot proceeds — relevance is an optimization
layered on the authorization gate (§7.1), never a boot dependency. There is **no
new gateway inbound surface** and **no new credential** — it reuses the
per-gateway secret and the same host as `/relay/provision`.

> A relevance drop happens **before** the connector wakes a scaled-to-zero agent
> (Phase 5), so excluded chatter never spins an agent up — relevance is the
> primary scale-to-zero lever as well as a correctness filter.

---

## 8. Versioning policy

- `contract_version` is an int; bump **only** for additive changes during the
  experimental phase (new optional fields, new `op`s).
- A breaking change (renamed/removed field, changed semantics) requires a
  coordinated update of both repos and a version bump.
- The connector's first PR references the commit SHA of this file it implements
  against.
