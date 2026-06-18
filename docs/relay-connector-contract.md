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
delivered over a signed HTTP POST, not the outbound `/relay` WebSocket** (see
the transport note below). The gateway keys the session via `build_session_key()`
from the embedded `SessionSource` — so populating the right discriminators is
the single highest-correctness responsibility of the connector.

### Inbound transport (signed HTTP POST, not the outbound WS)

The gateway dials **out** to the connector's `/relay` WebSocket for the
handshake + outbound actions (§4) + its own `/stop` egress (§5). Inbound,
however, is delivered the other way: the connector **POSTs** the normalized
event to the gateway's inbound endpoint (`HttpGatewayDelivery` on the connector;
`gateway/relay/inbound_receiver.py` on the gateway). The reason is
multi-instance: the connector instance that owns a platform's socket (and thus
produces inbound events) is generally **not** the instance a given gateway
dialed its outbound WS into, so inbound must target a tenant **endpoint** (which
may load-balance across gateway instances) rather than ride one gateway's
outbound socket. Each delivery is HMAC-signed with the per-tenant **delivery
key** (§6.1); the gateway verifies the signature over the exact raw bytes before
accepting the event. Two POST targets:

- `POST {gatewayEndpoint}`            → `{"type":"message", "event": <MessageEvent>}`
- `POST {gatewayEndpoint}/interrupt`  → `{"type":"interrupt", "session_key", "reason"?}` (§5)

> An earlier draft of this contract delivered inbound over the WS `inbound`
> frame. That only works single-instance and predates the multi-instance
> socket-ownership + channel-auth model; the signed-HTTP path above is the
> shipped design.

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
  as a **signed HTTP POST** to `{gatewayEndpoint}/interrupt` (§3 transport note),
  and bridged by the adapter's `on_interrupt(session_key, chat_id)` into the
  existing per-session interrupt mechanism, cancelling exactly that turn
  (siblings untouched).

The gateway→connector `/stop` rides the outbound WS; the connector→gateway
interrupt rides the same signed-HTTP inbound path as a normalized event.

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
is itself authenticated. The gateway holds two enrollment-issued credentials
(`hermes gateway enroll` → connector `/relay/enroll`): a **per-gateway secret**
and a **per-tenant delivery key**. Both are HMAC-SHA256 schemes with a
multi-secret rotation verify list (gateway side: `gateway/relay/auth.py`;
connector side: `src/core/relayAuthToken.ts` + `src/core/deliverySigning.ts`).

| Leg | Credential | Mechanism |
|-----|-----------|-----------|
| Gateway → connector WS upgrade | per-gateway secret | An `Authorization` bearer header on the `/relay` upgrade. The token is `base64url(payload:exp:sig)` where `payload = gatewayId` and `sig = HMAC(payload:exp, secret)`. Connector verifies and rejects the upgrade (**close 4401**) on mismatch/absence/revocation. The authenticated tenant comes from the connector's store, never the `hello` frame. |
| Connector → gateway inbound POST | per-tenant delivery key | Two headers: `x-relay-timestamp` (unix seconds) and `x-relay-signature` (hex `HMAC(ts.rawBody, deliveryKey)`). Gateway verifies over the **exact raw bytes** within a ±300s replay window before accepting the event; rejects **401** otherwise. |

This is the **channel** authenticator — distinct from platform crypto, which the
relay path still sheds entirely (§6). The gateway holds zero platform secrets;
these two keys authenticate only the connector link. Full threat model +
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
