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
> longer used for inbound. `gatewayEndpoint` remains only for the **passthrough
> plane** (Class-2/3 webhooks like Discord interactions / Twilio), which is a
> separate synchronous-forward path and out of scope for this section.



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
| `guild_id` | string | no | Discord guild / Slack workspace / Matrix server scope. **REQUIRED for Discord server isolation.** Session-key discriminator. |
| `parent_chat_id` | string | no | Parent channel when `chat_id` refers to a thread. |
| `message_id` | string | no | Id of the triggering message (for pin/reply/react). |

> `is_bot` (author-is-a-bot/webhook classification) exists on the gateway-side
> dataclass but is **intentionally NOT on the wire** in v1 — it is not part of
> `to_dict()`. Do not add it to the connector's `SessionSource` until it is
> first added here and to `to_dict()` (additive bump).

### SessionSource discriminators per platform

| Platform | chat_id | chat_type | user_id | thread_id | guild_id |
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

## 7. Versioning policy

- `contract_version` is an int; bump **only** for additive changes during the
  experimental phase (new optional fields, new `op`s).
- A breaking change (renamed/removed field, changed semantics) requires a
  coordinated update of both repos and a version bump.
- The connector's first PR references the commit SHA of this file it implements
  against.
