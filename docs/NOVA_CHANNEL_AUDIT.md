# NOVA Channel Audit — what Hermes already provides

> **Correction (later than this document).** `multiplex_profile_allowlist` was removed
> from Hermes' `GatewayConfig`; the served-profile set is now computed from the profile
> directory and configuration cannot narrow it. The grant is still enforced — by NOVA
> replacing `gateway.profile_routes` — but by one mechanism rather than two. See
> `docs/NOVA_CHANNEL_SECURITY.md`, *Correction: one tier of this enforcement is gone*.

**Phase A. No implementation. Nothing below is inferred from a filename, a README, or an
unused abstraction** — every claim names the file and the call site it came from, and
anything not verified says `UNVERIFIED` rather than `YES`.

The headline, because it changes the whole shape of Phase 9:

> **Hermes already has a complete, production-grade messaging platform system**: 22 bundled
> platform plugins, a registry with a documented plugin seam, per-profile credential
> scoping that fails closed, channel→profile routing, default-deny access control, webhook
> signature verification with a replay window, and a durable outbound delivery ledger.
>
> **NOVA must not build any of that.** The entire job is a commercial facade.

---

## 1. Where the capability lives

| Concern | File | What it is |
|---|---|---|
| Adapter base class | `gateway/platforms/base.py` (4,234 lines) | `BasePlatformAdapter` — send, edit, typing, drafts, attachments, threads |
| Registry | `gateway/platform_registry.py` | `PlatformEntry`, `create_adapter(name, cfg)`, scoped register/unregister |
| **Plugin seam** | `hermes_cli/plugins.py:770` | `ctx.register_platform(name, label, adapter_factory, check_fn, …)` |
| Platform enum | `gateway/config.py:198` | 26 built-in values **plus dynamic members** for plugin platforms (`_missing_`) |
| Per-platform config | `gateway/config.py:386` | `PlatformConfig` — enabled, token, home_channel, `extra` |
| Config file | `gateway/config_loader.py:342` | `<HERMES_HOME>/config.yaml`, key `gateway.platforms.<name>` |
| **Channel → agent routing** | `gateway/profile_routing.py` | `ProfileRoute`, `match_profile_route`, `gateway.profile_routes` |
| Routing call site | `gateway/run.py:4275-4306` | resolves a profile per inbound message, **rejects unserved profiles** |
| Served-profile set | `hermes_cli/profiles.py:707` | `profiles_to_serve(multiplex, profile_allowlist)` |
| Credential scoping | `agent/secret_scope.py:110` | `get_secret` — fails closed under multiplexing |
| Adapter credential read | `gateway/platforms/_shared.py:17` | `get_scoped_secret(name, default)` |
| Webhook verification | `gateway/platforms/webhook.py` | HMAC-SHA256, Svix, timestamp replay window, idempotency cache |
| Outbound durability | `gateway/delivery_ledger.py` | pending / attempting / delivered / failed / abandoned, in `state.db` |
| Access control | `gateway/platforms/base.py:1889-1905` | default-deny; `{PLATFORM}_ALLOW_ALL_USERS` to open |

### The plugin seam is explicitly a zero-core-patch path

`gateway/platforms/ADDING_A_PLATFORM.md` opens with:

> "Create a plugin directory … The adapter inherits from `BasePlatformAdapter` and registers
> via `ctx.register_platform()` … This requires **zero changes to core Hermes code**."

That sentence is the answer to "what is the narrowest extension point" and it is written by
the runtime's own authors. NOVA's patch budget stays at 1.

---

## 2. The 23 audit questions, answered

**1. Does it actually work?** Yes, for the bundled set. 22 platform plugins ship under
`plugins/platforms/`, plus 12 built-in adapters under `gateway/platforms/`. The repository
carries a large live test corpus under `tests/gateway/` (Slack streaming, Discord slash
commands, Telegram reply modes, WhatsApp formatting, webhook adapters, email charset
handling). **Per-platform functional status is not something this audit verified** — see the
matrix, where every provider is marked from code reading, not from a live connection.

**2. How is it configured?** Two layers. `<HERMES_HOME>/config.yaml` under
`gateway.platforms.<name>`, and environment variables, with **env winning over YAML**
(`load_gateway_config`: *"Priority: env > ~/.hermes/config.yaml > legacy gateway.json"*).
Each plugin declares its variables in `plugin.yaml` as `requires_env` / `optional_env` with
`password: true` on the secret ones.

**3. Where are credentials stored?** Environment variables, resolved through
`get_scoped_secret` → `agent.secret_scope.get_secret`. Under multiplexing a profile's
`<profile>/.env` is the authoritative source and a miss returns the default rather than
another profile's value. **This is the same credential model NOVA already uses** (`key_env`,
`.env` on `materialize.NEVER_WRITE`) — no new mechanism is needed.

`PlatformConfig.token` can also carry a raw secret from `config.yaml`, and `to_dict()`
includes it. **NOVA must never use that field** — see §4.

**4. How does an inbound message enter Hermes?** The adapter owns its transport and pushes
a `PlatformEvent` into the gateway (`gateway/platforms/event.py`, `gateway/run_inbound.py`).
Transport varies by provider: long-polling (Telegram), websocket/Socket Mode (Slack,
Discord), IMAP polling (Email), HTTP webhook (WhatsApp Cloud, MS Graph, generic webhook),
or an external bridge process (WhatsApp Baileys, `scripts/whatsapp-bridge/`).

**5. How does Hermes identify the user/contact/channel?** A `SessionSource` carrying
`platform`, `chat_id`, `guild_id`, `thread_id`, `parent_chat_id` and a sender identity.
WhatsApp additionally collapses phone number / JID / LID to one identity through
`gateway/whatsapp_identity.py::expand_whatsapp_aliases`.

**6. How is the message routed to an agent/profile?** `gateway.profile_routes` in
config.yaml. Rules match on `platform` + optional `guild_id` / `chat_id` / `thread_id`, and
are sorted **most-specific-first** (`specificity = 2·guild + 4·chat + 8·thread`). This is
the single most important finding for Phase D: **channel→agent routing already exists and
NOVA must compile to it rather than reinvent it.**

**7. How does an agent send a response?** `adapter.send()` / `send_draft()`, with the stream
consumer driving progressive edits. Cross-platform and out-of-process delivery go through
`gateway/delivery.py` and the plugin's `standalone_sender_fn`.

**8. Background messages?** Yes — `home_channel` (`{PLATFORM}_HOME_CHANNEL`) is the target
for cron and notification delivery, with `cron_deliver_env_var` routing `deliver=<name>` cron
jobs without touching `cron/scheduler.py`.

**9. Attachments?** Supported in the base adapter and the media pipeline
(`gateway/media_fetch.py`, `media_policy.py`, `media_cache.py`, `media_repair.py`).
Per-provider coverage `UNVERIFIED`.

**10. Groups?** Yes — `guild_id` / group chat ids are first-class in routing and in the
WhatsApp group/broadcast/newsletter JID handling. Per-provider `UNVERIFIED`.

**11. Threads/replies?** Yes — `thread_id`, `parent_chat_id`, `reply_to_mode`
(`off`/`first`/`all`), and Discord forum/thread parenting. Per-provider `UNVERIFIED`.

**12. Message history?** Sessions and transcripts are runtime-owned
(`gateway/session_*.py`, `session_transcript.py`). NOVA does not and should not own this.

**13. Webhook, polling, gateway or long-running process?** All four exist, per provider. In
every case a **long-running gateway process** owns the connection.

**14. What process owns the channel connection?** The Hermes gateway (`gateway/run.py`,
`GatewayRunner`). One gateway process per launch `HERMES_HOME`.

**15. What happens after restart?** Adapters reconnect; `gateway_restart_notification`
optionally announces it. Outbound obligations recorded in the delivery ledger are
recovered — see 16.

**16. Is delivery durable?** **For outbound final responses, yes, and unusually carefully.**
`gateway/delivery_ledger.py` records an obligation before any send and distinguishes
`pending` (never started → redeliver plainly) from `attempting` (crashed mid-await, the
platform *may* have it → redeliver **with a visible recovered marker**). It explicitly
refuses to silently resend an ambiguous send. **Inbound durability is NOT equivalent** —
webhook intake has an idempotency cache with a 1-hour TTL (`webhook.py:178`), which is
de-duplication, not a queue. An inbound message arriving while the gateway is down is lost
unless the provider itself retries.

**17. Retry/error semantics?** Ledger attempts are capped; stale rows expire to `abandoned`.
Ledger failures are best-effort by design and must never block a send. Rate limiting exists
per-adapter (e.g. `gateway/platforms/signal_rate_limit.py`).

**18. Security boundary?** Default-deny on inbound identity: with no allowlist, an adapter
that does not enforce its own policy is denied. `enforces_own_access_policy` is trusted only
when the effective policy is a real allowlist — *never* `open`, described in the source as
"a network-exposed fail-open (SECURITY.md §2.6)". Webhooks verify HMAC-SHA256, support Svix,
bind a timestamp with a replay window, cap body size before reading, and rate limit.

**19. What tenant isolation exists?** **Per-profile, and it fails closed — with one stated
exception.** Under `multiplex_profiles`, each secondary profile's adapters are constructed
and connected inside `_profile_runtime_scope` (`gateway/run.py:1730`), which installs a
secret scope so credentials come from that profile's `.env` and never from `os.environ`.
`get_secret` **raises** `UnscopedSecretError` if a credential is read with no scope while
multiplexing. **The exception: the default profile runs unscoped** and falls back to
`os.environ` (`gateway/platforms/_shared.py:17`). This is the same boundary the Phase 6
readiness audit found for model credentials, and NOVA must state it identically.

**20. Can multiple customers use the same channel provider safely?** Within one deployment,
two agents can hold two different Telegram bots as two profiles with two `.env` files. Across
customers, **no** — NOVA is one deployment per tenant by design, so this question is answered
by the deployment boundary, not by the channel layer.

**21. Can multiple agents use the same channel?** Yes, via `profile_routes`: different
`chat_id` / `guild_id` / `thread_id` on one platform route to different profiles.

**22. Can one agent have multiple channels?** Yes — a profile is a routing *target*, and any
number of routes may point at it.

**23. Genuinely reusable, or coupled to Hermes internals?** **Reusable through a documented,
supported seam.** `ctx.register_platform()` is a public plugin API with a written contract.
The routing, credential and config surfaces are all file-and-env based. NOVA needs no core
patch and no private import.

---

## 3. Capability matrix

Columns marked `UNVERIFIED` were read in source but **not exercised against a live provider
in this environment**. Per the instruction: unknown is not YES.

| Channel | Hermes capability | Real implementation | Inbound | Outbound | Attach | Groups | Threads | Durable (out) | Credential model | Reusable by NOVA |
|---|---|---|---|---|---|---|---|---|---|---|
| Telegram | Plugin | `plugins/platforms/telegram/` | long-poll | YES | UNVERIFIED | YES | YES (topics) | ledger | `TELEGRAM_BOT_TOKEN` | YES |
| Slack | Plugin | `plugins/platforms/slack/` | Socket Mode | YES | UNVERIFIED | YES | YES | ledger | `SLACK_BOT_TOKEN` + `SLACK_APP_TOKEN` | YES |
| Discord | Plugin | `plugins/platforms/discord/` | websocket | YES | UNVERIFIED | YES | YES (forum) | ledger | `DISCORD_BOT_TOKEN` | YES |
| WhatsApp (Cloud) | Built-in | `gateway/platforms/whatsapp_cloud.py` | webhook | YES | UNVERIFIED | UNVERIFIED | UNVERIFIED | ledger | Meta app secret + token | YES |
| WhatsApp (Baileys) | Plugin + bridge | `plugins/platforms/whatsapp/`, `scripts/whatsapp-bridge/` | bridge process | YES | UNVERIFIED | YES | UNVERIFIED | ledger | pairing session | YES, extra process |
| Email | Plugin | `plugins/platforms/email/` | IMAP poll | SMTP | UNVERIFIED | n/a | YES (threads) | ledger | `EMAIL_PASSWORD` etc. | YES |
| MS Teams | Plugin | `plugins/platforms/teams/` | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | ledger | UNVERIFIED | YES |
| Web chat | Built-in | `gateway/platforms/api_server.py` | HTTP | YES | UNVERIFIED | n/a | UNVERIFIED | ledger | `API_SERVER_KEY` | YES |
| Generic webhook | Built-in | `gateway/platforms/webhook.py` | webhook | deliver-only | UNVERIFIED | n/a | n/a | ledger | HMAC secret | YES |
| Signal, Matrix, Mattermost, IRC, LINE, Feishu, WeCom, Weixin, DingTalk, QQBot, Google Chat, SMS, ntfy, Home Assistant, BlueBubbles, SimpleX, A2A, Matrix… | Plugin/built-in | `plugins/platforms/*`, `gateway/platforms/*` | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | ledger | per-plugin env | YES |

**No channel in this table has been tested against a live provider by NOVA.** Phase L will
mark whatever is exercised as LOCAL VALIDATION unless real credentials appear.

---

## 4. What is missing for a commercial product

Everything Hermes has is *operator*-shaped. Nothing is *customer*-shaped. The gaps:

| # | Gap | Why it blocks a commercial product |
|---|---|---|
| 1 | **No tenant concept in channels** | `profile_routes` names profiles; nothing binds a route to a tenant, so nothing refuses a route into another tenant's agent by construction. NOVA already enforces one-tenant-per-deployment; the channel layer must inherit that, not re-open it. |
| 2 | **No declared agent authorization per channel** | A route says "this chat goes to this profile". Nothing says "this connection may *only* reach these agents". The served-profile allowlist is close but is a gateway-wide setting, not a per-connection grant. |
| 3 | **Credentials are operator-shaped** | The customer is expected to know `SLACK_APP_TOKEN`. A commercial product must let them name a channel and never see an env var. NOVA's `key_env` indirection already solves this and simply has not been pointed at channels. |
| 4 | **`PlatformConfig.token` can carry a raw secret** | A YAML-authored token would land in the file NOVA writes and version-controls. NOVA must refuse this field outright and use env indirection only. |
| 5 | **No customer-facing surface at all** | Configuration is YAML + env + a setup wizard. There is no API, no dashboard, no audit of who connected what. |
| 6 | **No NOVA audit of channel events** | Hermes logs; it does not write NOVA's tamper-evident, write-ahead audit. An inbound message that reaches a model is *model-visible* by NOVA's own definition and is currently unlogged by NOVA. |
| 7 | **Inbound is not durable** | Outbound has a ledger; inbound has a 1-hour idempotency cache. A commercial claim of "durable messaging" would be false today and must not be made. |
| 8 | **Default profile credential leak under multiplexing** | Stated in Q19. Not a NOVA bug and not NOVA's to fix in the runtime, but NOVA must report it rather than claim blanket isolation. |

---

## 5. The seam NOVA should use (input to Phase B)

```
Customer  →  NOVA Channel Facade  →  compiles to  →  Hermes config + env names
                                                      ├─ gateway.platforms.<provider>  (enable, non-secret settings)
                                                      ├─ gateway.profile_routes        (channel → agent)
                                                      ├─ gateway.multiplex_profiles    (serve many agents)
                                                      ├─ gateway.multiplex_profile_allowlist (which agents at all)
                                                      └─ <profile>/.env  ← operator writes VALUES; NOVA writes NAMES
```

NOVA writes **no adapter, no transport, no protocol code**. It writes configuration for a
system that already works, and it owns the parts Hermes deliberately does not: the tenant
boundary, the per-connection agent grant, the customer vocabulary, and the audit record.

**Explicitly rejected alternatives**, and why:

- *Writing a NOVA platform plugin per provider.* Re-implements 22 working adapters, and the
  patch budget is not the constraint — the duplication is.
- *NOVA terminating webhooks itself.* Hermes already verifies HMAC, binds timestamps and
  de-duplicates. A second implementation is a second thing to get wrong.
- *A NOVA-side message queue.* That is a second dispatcher, forbidden by Phase K, and it
  would sit in front of a runtime that already has durable outbound delivery.

---

## 6. What this audit did not establish

Stated plainly so nothing downstream over-claims:

- **No live provider connection was made.** Every per-provider capability is source-read.
- **Attachment, group and thread support per provider is UNVERIFIED**, and the matrix says so.
- **Inbound durability under gateway restart was not measured**, only read.
- **The multiplexed two-agent-two-bots case was not run.** Phase L must run it before NOVA
  claims an agent can hold its own channel credential.
- **MS Teams, despite being on the customer wish-list, is entirely UNVERIFIED.** It ships as
  a plugin and was not opened beyond its manifest.
