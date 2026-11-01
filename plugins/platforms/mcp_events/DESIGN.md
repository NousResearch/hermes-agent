# MCP Events Receiver — Design (DRAFT)

Event-driven agent wake-ups for Hermes, following the proposed MCP Events
extension (protocol `2026-07-28`): an MCP event *emitter* pushes signed webhook
deliveries, and the subscribed agent wakes and acts — no cron, no human message.

## Why a plugin, not a core feature

The A2A plugin is the precedent: a whole protocol adapter as a plugin with zero
core edits, built on `ctx.register_platform()` / `ctx.register_tool()`. MCP Events
is the same shape — an open inter-agent protocol at the edge, not core surface.
Per the Footprint Ladder it sits at rung 4 (plugin): it needs a persistent
listener and per-session capability, which neither a CLI command nor a
service-gated tool provides.

**In-tree policy note (for maintainers):** the June 2026 rule keeps
third-party *product* plugins (vendor SaaS, analytics backends) out of the tree.
This is not one: MCP Events is an open protocol extension in the same category
as A2A (a Linux Foundation standard, in-tree at `plugins/platforms/a2a/`).
No vendor SDK, no product tie-in — stdlib only. If maintainers disagree, the
plugin is self-contained and can move to the catalog untouched.

## Two directions (receiver-side scope)

Hermes is the **subscriber**, not the emitter:

- **Outbound — client tools** (`mcp_events` toolset): `mcp_events_list` (the
  emitter's `events/list`), `mcp_events_subscribe` (`events/subscribe` with our
  callback URL + signing secret), `mcp_events_unsubscribe`, `mcp_events_subscriptions`.
- **Inbound — platform adapter**: stdlib `http.server` on a daemon thread (the
  A2A "register outside a loop" lesson — no asyncio needed at `register()` time).
  `POST /mcp/events/webhook/<local-id>` verifies the Standard Webhooks signature,
  dedupes, rate-limits, frames the payload as untrusted input, and routes it
  through the normal `MessageEvent` → `handle_message` path keyed by
  `mcp-events:<subscription>` — the agent that wakes is the one serving the
  user, with full memory/context, not a clone. The HTTP thread answers 200
  immediately; it never blocks on the agent's turn (emitters retry with backoff,
  so a slow 200 would multiply deliveries).

The emitter direction (Hermes *emitting* events others subscribe to) is
deliberately out of scope — a follow-up, not this PR.

## Protocol notes (proposed extension, not core spec)

MCP Events is a *proposed* extension implemented against MCP 2.0 /
protocol `2026-07-28`; it is not part of the core spec (whose push path is the
`subscriptions/listen` SSE stream). This plugin implements the documented
draft as OpenAI's receiver does:

- `server/discover` capability probe for `events` (advisory — subscribe tries anyway).
- `events/list`, `events/subscribe`, `events/unsubscribe` as JSON-RPC on the
  emitter's MCP endpoint, with the `Mcp-Method` header matching the body
  (2026-07-28 transport requirement; mismatch is rejected with -32020).
- Delivery signing: Standard Webhooks — `webhook-id` / `webhook-timestamp` /
  `webhook-signature` headers, `v1,<base64(HMAC-SHA256(secret, id.ts.body))>`,
  multiple space-separated signatures tolerated (secret rotation), timestamp
  freshness window (default ±300s), 256 KiB per-event ceiling.
- The subscriber supplies the `whsec_`-prefixed secret at subscribe time; the
  emitter signs with it; we verify on receipt. The callback URL embeds a local
  id (the emitter's id is only known after subscribe returns); the store
  resolves either.

## Security (on by default)

- **Bind safety:** no `MCP_EVENTS_WEBHOOK_SECRET` ⇒ 127.0.0.1 only. Remote
  exposure additionally needs `mcp_events.public_base_url` (else subscribing is
  refused fail-fast — a remote emitter could never deliver).
- **Delivery auth:** HMAC only, constant-time compare; nothing in the body is trusted.
- **Emitter allow-list:** `mcp_events.trusted_emitters`; network-exposed binds
  with no allow-list and no `allow_all_emitters` fail closed.
- **SSRF:** emitter/callback URLs must be public http(s); private/link-local/
  metadata addresses blocked even in localhost mode (loopback allowed there).
- **Injection:** every payload is defanged (ChatML/role-prefix/override patterns)
  and framed with an untrusted-input boundary before reaching the agent. Events
  can never reach operator slash commands.
- **Anti-storm:** per-emitter sliding-window rate limit (default 120/min) plus a
  per-subscription storm guard (default 60/min) — an emitter firing 10k events
  must not produce 10k turns. Trips are audit-logged, not silent.
- **Idempotency:** seen webhook-ids are acked without re-waking (bounded, TTL'd).
- **Audit:** append-only `~/.hermes/mcp_events_audit.jsonl` for every accepted,
  dropped, subscribed, and unsubscribed event.
- **Secrets:** `.env` holds only the secret; every behavioral knob is
  `config.yaml → mcp_events:` (root AGENTS.md: ".env is for secrets only").

## State placement

Subscriptions persist at `~/.hermes/mcp_events_subscriptions.json` (via
`get_hermes_home()` — never a hardcoded path), outside the context-compaction
pipeline. On (re)connect, expired subscriptions are re-subscribed; emitters that
rotate ids get the old record retired. Rate-limit windows and idempotency sets
are in-memory by design — a restart resets them, which is the safe direction.

## Files

```
plugins/platforms/mcp_events/
├── plugin.yaml      # manifest (kind: platform); only the secret is env
├── __init__.py      # register(): platform adapter + client tools
├── adapter.py       # inbound webhook receiver (stdlib http.server)
├── protocol.py      # JSON-RPC framing, Standard Webhooks verify, subscription store
├── security.py      # scoped secrets, SSRF guard, injection filter, limits, audit
├── tools.py         # outbound client tools (list/subscribe/unsubscribe/subscriptions)
├── DESIGN.md
└── README.md
```

## Named emitters

The tools take an emitter *name* configured under `mcp_events.emitters:` (with
`url` and optional `headers`). This fixes two problems with raw URLs, found in
end-to-end use against
[mcp-events-bridge](https://github.com/hookdeck/mcp-events-bridge):

- **Identity.** The URL otherwise becomes the emitter's identity everywhere:
  the tool replies, the session's `user_id`, each delivery's framing, the audit
  log and the gateway logs. A named emitter shows as its name in all of those;
  the URL (and its host, for the per-emitter rate limit) stays internal.
- **Credentials.** A URL is the only credential channel a bare tool call
  offers, and MCP servers without OAuth commonly carry a secret in the URL path
  (a capability URL — mcp-events-bridge and Zapier's MCP URLs do). That URL
  then reaches the model context in every framed delivery, the session store,
  the audit log and the agent's replies; no redaction can recover it. With
  named emitters, the URL and any `Authorization` headers resolve from
  config/.env at call time and never pass through the model or the logs (the
  way `MCP_EVENTS_WEBHOOK_SECRET` already stays out of them).

Secret placement follows the repo's env policy: a URL or header set that
carries a secret lives in `MCP_EVENTS_EMITTER_<NAME>_URL` / `_HEADERS`
(profile-scoped, name uppercased with non-alphanumerics as `_`), which wins
over the config.yaml entry. Unnamed `emitter_url` calls remain for ad-hoc
public emitters (SSRF-guarded as before); a named emitter skips the URL guard
because operator configuration *is* the allow decision.

## Deliberately out of scope (future, not this PR)

- **Emitter direction** (Hermes exposing `events/*` for others to subscribe to).
- **Multi-round-trip / `subscriptions/listen`** interop with core-spec push.
- Per-event authorization policies beyond the emitter allow-list.
- OAuth for emitters (named-emitter headers cover bearer tokens; a full OAuth
  grant flow is a follow-up once there is a concrete emitter that needs it).
