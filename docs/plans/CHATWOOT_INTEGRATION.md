# Chatwoot Integration — Implementation Guide

A logic-first guide to integrating **Chatwoot** as a messaging platform in a
Hermes-style gateway. It focuses on *what* each piece must do and *why*, not on
the exact code of any one branch — so it stays useful even if file layouts,
function names, or signatures differ from the version you're working in.

> **How to use this doc:** treat every section as a requirement ("the adapter
> must resolve the conversation id like this…"), then find the equivalent hook
> in your codebase. In Hermes, the canonical checklist is
> `gateway/platforms/ADDING_A_PLATFORM.md` — map each item below onto it.
>
> Audience: engineers building the integration. End-user setup (creating the
> bot, pointing the webhook) belongs in a separate user guide.

> **Verified against this repo (Hermes `main`).** The file/line anchors in this
> doc were checked against the current tree. Load-bearing ones:
> `register_platform()` → `hermes_cli/plugins.py:882`; the `PlatformEntry` hook
> surface → `gateway/platform_registry.py:39`; base-adapter helpers
> (`is_network_accessible`, `handle_message`, `build_source`, the cache helpers,
> `MAX_MESSAGE_LENGTH`) → `gateway/platforms/base.py`; own-aiohttp-listener
> webhook template → `gateway/platforms/msgraph_webhook.py` and
> `plugins/platforms/teams/adapter.py:712`; plugin toolset auto-resolution →
> `toolsets.py:709`; plugin send-routing → `tools/send_message_tool.py:683`;
> cron delivery hooks → `cron/scheduler.py:690` (`_plugin_cron_env_var`) and
> `tools/send_message_tool.py:687` (`standalone_sender_fn`). Re-grep before
> relying on any anchor — line numbers drift.

### Built-in adapter vs plugin (decide this first) — **use the plugin path**

> **Decision for this integration: the plugin path.** Every actively-developed
> messaging platform in Hermes is now a plugin — Telegram, Discord, Slack, and
> WhatsApp all live under `plugins/platforms/`, not `gateway/platforms/`. Only a
> handful of legacy adapters (`signal`, `bluebubbles`, `weixin`, `whatsapp_cloud`)
> remain built-in. Build Chatwoot as `plugins/platforms/chatwoot/`. The built-in
> path below is retained only as an appendix for a hypothetical core upstream.

Hermes offers two ways to add a platform, and this choice reshapes most of the
wiring below:

- **Plugin (recommended for a new/community platform).** Ship
  `plugins/platforms/chatwoot/` with a `plugin.yaml` + `adapter.py` that
  subclasses `BasePlatformAdapter` and registers via `ctx.register_platform()`.
  This requires **zero core edits** — the plugin surface auto-handles
  authorization, cron delivery, `send_message` routing, system-prompt hints,
  status display, and setup. Optional hooks cover the edges:
  `env_enablement_fn`, `apply_yaml_config_fn`, `cron_deliver_env_var`,
  `standalone_sender_fn`. See `plugins/platforms/irc/`, `teams/`,
  `google_chat/`. The bulk of §7's manual wiring collapses into the plugin
  registration in this path.
- **Built-in (core contribution).** Edit core files directly per the numbered
  sections of `ADDING_A_PLATFORM.md`. The §7 checklist below is written against
  this path; each row is annotated with the file it touches.

Everything in §2–§4 and §8 (the Chatwoot protocol itself) is identical either
way. Pick the plugin path unless you're deliberately upstreaming Chatwoot as a
first-class built-in.

---

## 1. What Chatwoot is and how it integrates

[Chatwoot](https://www.chatwoot.com/) is an open-source customer-support
platform (cloud or self-hosted). It funnels conversations from many "inboxes"
(website live-chat, email, WhatsApp, social, etc.) into one agent dashboard,
grouped under **accounts**.

Chatwoot integrates automation through an **Agent Bot**:

- A bot is created once and **connected to one or more inboxes**.
- **Inbound (Chatwoot → you):** Chatwoot sends HTTP POST webhook events to the
  bot's configured **outgoing URL** whenever something happens in a connected
  conversation (a message is created, a conversation opens, etc.).
- **Outbound (you → Chatwoot):** the bot replies by calling Chatwoot's
  **Application API**, authenticating with the bot's **access token**.

The integration goal: **a customer messaging a connected inbox is talking to the
Hermes agent**, with each conversation isolated into its own agent session.

### The defining characteristic: request/response, no persistent connection

Unlike streaming platforms (which hold a socket open) this integration is pure
HTTP in both directions:

- **Inbound has no socket to maintain** — Chatwoot calls you. You just need a
  reachable HTTP endpoint.
- **The adapter runs its own aiohttp listener.** Hermes has **no** single shared
  inbound HTTP host to mount onto — every inbound-webhook platform stands up its
  own `aiohttp` server. Follow that precedent: `gateway/platforms/webhook.py`
  (generic webhook server), `gateway/platforms/msgraph_webhook.py` (own port +
  path), and `gateway/platforms/api_server.py`. The Chatwoot adapter should bind
  a configurable host/port/path in `connect()` and tear it down in
  `disconnect()`. There is no "shared HTTP host" dependency to enable first.

```
Customer ──▶ Chatwoot ──HTTP POST──▶ [Chatwoot adapter's aiohttp server]
                                                       │
                                     build event, dispatch ──▶ Agent
                                                       │
Customer ◀── Chatwoot ◀──REST call── Chatwoot adapter ◀── reply ─┘
```

---

## 2. Core concepts you must get right

These four decisions drive the whole implementation. Get them right first.

### 2.1 Identity: how to address a conversation

To post a reply, Chatwoot's API needs **two** identifiers: the **account id**
and the **conversation id**. Both are present in every inbound webhook.

**Encode them together into a single opaque chat id**, e.g.
`"{account_id}:{conversation_id}"`. This makes a reply target self-contained:
anything holding the chat id (the session store, a scheduled job, a
send-message tool) can reply without extra lookups. Provide a small
parse/format helper pair and use it everywhere.

### 2.2 Session isolation: one session per conversation

Each Chatwoot conversation should map to its own agent session so histories
don't bleed together. In most gateways the session key is derived from
`platform + chat_type + chat_id`. Because the chat id already encodes
account+conversation, marking the chat as a **direct/1:1 type** yields a unique
session per conversation automatically. No custom session logic is needed — you
just have to feed the session layer the right chat id and chat type.

### 2.3 Inbound authenticity (webhook trust)

Agent Bots **cannot send custom headers or signatures** on their webhook calls.
The only trust mechanism Chatwoot offers is what you can bake into the outgoing
URL. So:

- Support an **optional shared secret** passed as a query parameter on the
  webhook URL (e.g. `?token=…`), compared using a constant-time comparison
  (`hmac.compare_digest`).
- If no secret is configured, accept the POST but **log a warning** — and rely
  on user-authorization (2.4) as the real gate. This is acceptable when the
  endpoint isn't publicly reachable (e.g. inside a private network/VPC). Use
  `is_network_accessible()` (from `gateway/platforms/base.py`, as
  `msgraph_webhook.py` does) to make that warning louder when the endpoint is
  actually reachable from the public internet.

### 2.4 Outbound authenticity — two tokens, two jobs

The integration uses **up to two Chatwoot credentials**. Get the split right:

- **Agent Bot token (required) → customer-visible replies.** The Agent Bot
  authenticates outbound API calls with its **access token**, sent in the
  **`api_access_token`** request header (the header name is shared across
  Chatwoot token types). Use the **bot** token for normal outgoing replies, not
  a human agent's personal/user token.
- **User/agent access token (optional) → private notes.** Private notes
  (`private: true`) are attributed to an *agent user*, and the bot token is
  limited-scope. Chatwoot's own CLI posts `--private` notes under a **user**
  token (`chatwoot auth login` caches a `user_id`; `CHATWOOT_API_KEY` is a
  profile/user token — see `chatwoot-docs/docs/cli/commands.mdx:133`,
  `scripting.mdx:56`). So the private-note trace feature (§4a) likely needs a
  **separate** user token, exposed as `CHATWOOT_AGENT_TOKEN`.

> **Verify against the live instance before shipping.** The bundled
> `chatwoot-docs` documents only the *user* token flow for Application APIs and
> does **not** bundle the OpenAPI spec, so neither the bot-token-on-messages
> behavior nor the private-note token rule is provable from this repo. Confirm
> both directly:
> 1. `POST …/conversations/{id}/messages` with `message_type: "outgoing"` using
>    the **bot** token → must succeed (proves the reply path).
> 2. The same endpoint with `private: true` using the **bot** token → if it
>    401/403s or the note isn't created, private notes require the user token
>    (`CHATWOOT_AGENT_TOKEN`); wire §4a accordingly.

---

## 3. Inbound event handling (the heart of the integration)

Chatwoot fires many event types; you must **decide which ones cause the agent to
respond** and turn those into your gateway's internal message object. Everything
else should be acknowledged and ignored.

### 3.1 Which events to act on — filter, in order

Apply these filters and bail out early (return "ignore") on any miss:

1. **Event type.** Only act on the *new inbound message* event
   (`message_created`). Ignore updates, conversation lifecycle events, etc.,
   unless you have a specific reason to handle them.
2. **Direction (echo prevention).** Only handle messages **from the contact**,
   not the bot's own replies — otherwise you loop. Chatwoot marks direction with
   a `message_type` field. **Be defensive about its type:** it may arrive as the
   string `"incoming"` **or** as an integer enum. The current spec's message
   read-model uses `integer [0,1,2,3]` (`0` = incoming, `1` = outgoing,
   `2` = activity, `3` = template); the message *create* body uses the strings
   `["incoming","outgoing"]`. Accept both string and integer forms for incoming,
   or you risk silently dropping every message on some Chatwoot versions.
   Filtering strictly to incoming (`0` / `"incoming"`) also naturally excludes
   the `2`/`3` activity and template messages, which aren't customer content.
3. **Private notes.** Skip messages flagged private/internal — those are agent
   notes, not customer messages.
4. **Human handoff (important).** Respect Chatwoot's status convention:
   `pending` means the bot owns the conversation; `open` means a **human agent
   has taken over**. **Do not reply when the conversation status is `open`** so
   the bot never talks over a live agent. (Resolving or re-opening the
   conversation as `pending` hands control back to the bot.) The full status
   enum is `open` / `pending` / `resolved` / `snoozed` — only `pending` is the
   bot's to answer.
5. **Empty content.** If there's no text and no usable attachment, ignore.

> These filters are the difference between "works in a demo" and "works in
> production." The direction-type quirk (2) and the handoff rule (4) are the two
> most commonly missed.

### 3.2 Extracting the conversation id — mind `id` vs `display_id`

Chatwoot exposes conversation identifiers in more than one shape depending on
version and payload:

- an internal numeric `id`,
- a per-account `display_id`,
- sometimes a top-level conversation id field.

The reply API path expects the conversation's **display id**. To be robust,
**resolve the conversation id from several candidates in priority order**
(`conversation.id` → `conversation.display_id` → top-level fallback) rather than
assuming a single field. If your Chatwoot version routes replies by the wrong
one, you'll see reply calls fail with 404 — flip the priority.

### 3.3 Building the internal message

Once the event passes the filters, construct your gateway's inbound message
object with at least:

- **chat id** = the composite `account:conversation` id (§2.1),
- **chat type** = direct/1:1 (§2.2),
- **user id** = the sender's contact id (used for authorization, §5),
- **user/display name** = sender name/email if present,
- **text** = the message content,
- the **raw payload** attached for debugging/advanced use.

### 3.4 Attachments (inbound)

If the message carries attachments, download/cache the ones the agent can use
(images, audio) to local paths and record their media types; pass others
through as document references. Keep this best-effort — a failed attachment
fetch shouldn't drop the whole message.

### 3.5 The webhook endpoint's responsibilities

The adapter's own aiohttp handler (see `webhook.py` / `msgraph_webhook.py`)
that receives Chatwoot's POST should:

1. **Enforce a body-size limit before reading the payload.** Reject oversized
   bodies up front (DoS guard) — `webhook.py` does this.
2. **Validate the optional shared secret** (§2.3); reject mismatches.
3. **Parse JSON**; on malformed JSON, respond with a client-error status.
4. **Deduplicate on message id (idempotency).** Chatwoot retries webhooks and can
   double-fire `message_created`. Keep a bounded in-memory cache of recently seen
   message ids (a `deque` + set, like `webhook.py`'s idempotency cache and
   `msgraph_webhook.py`'s `max_seen_receipts`) and drop repeats so retries don't
   spawn duplicate agent turns.
5. **Convert the payload to an internal event.** If the converter says "ignore"
   (non-actionable), acknowledge with a success status and stop.
6. **Dispatch the agent turn without blocking the response.** Chatwoot expects a
   fast acknowledgement, so hand the message to the normal async processing path
   (fire-and-forget task, via `self.handle_message(event)`) and immediately
   return a success status. The reply is delivered later via the outbound API.
7. **On internal parse errors, still acknowledge** (success status) so Chatwoot
   doesn't retry a payload you can't use.

Suggested response contract (any equivalent works, but be consistent):

| Situation | HTTP status | Meaning |
|---|---|---|
| Chatwoot not enabled / adapter not running | 404 | integration off |
| Body exceeds size limit | 413 | rejected |
| Shared-secret mismatch | 403 | rejected |
| Body isn't valid JSON | 400 | bad request |
| Duplicate (already-seen message id) | 200 | acknowledged, no-op |
| Payload unusable by converter | 200 | acknowledged, dropped (no retry) |
| Non-actionable event (echo, note, handoff, empty) | 200 | ignored |
| Accepted and dispatched | 200 | processing |

> **Why dispatch through `handle_message` / the normal path?** Routing inbound
> Chatwoot messages through the same processing entry point the other platforms
> use means you inherit session management, interrupt handling, queueing, and
> authorization for free — no duplicated plumbing.

---

## 4. Outbound (replying and presence)

Implement these against Chatwoot's Application API, authenticating with the bot
token (§2.4). Parse the composite chat id (§2.1) to get account + conversation.

- **Send text.** POST a message to the conversation's messages endpoint with the
  content and an "outgoing" direction. Chatwoot has no hard length cap, but very
  long single messages render poorly — **chunk** beyond a sane limit. Back this
  with a `MAX_MESSAGE_LENGTH` constant on the adapter (the base-class convention;
  see other adapters). Surface HTTP errors; treat 5xx as retryable, 4xx as
  terminal.
- **Typing indicator (optional, best-effort).** Toggle the conversation's typing
  status on when the agent starts working and off when done. Swallow failures —
  presence is cosmetic.
- **Attachments (outbound).** Upload local files as multipart attachments on the
  messages endpoint (Chatwoot accepts an attachments field). For multipart, send
  only the auth header and let the HTTP client set the multipart content type.
  Wire image/document send paths through this uploader; for remote URLs, download
  to a temp file first.
- **Chat info.** Provide whatever minimal metadata lookup your gateway expects
  (name/type/chat id); a lightweight echo of the chat id is often enough.

The send path needs a **`private` flag** threaded through the outbound message
builder (default `False`). When set, add `private: true` to the message body so
the message posts as an internal note instead of a customer-visible reply. This
is the primitive §4a builds on.

### 4a. Reasoning / tool + skill activity as private notes

**Goal:** surface the agent's *thinking* and its *tool-call / skill-usage*
activity to human agents watching the conversation — as Chatwoot **private
notes** — while the customer only ever sees the final reply.

Hermes already streams reasoning and tool progress generically via
`gateway/display_config.py` (`show_reasoning`, `reasoning_style`,
`tool_progress`, per-platform tiers) feeding `progress_callback` and
`gateway/stream_consumer.py`. **But that path renders as normal,
customer-visible messages** — no platform currently routes it to an internal
channel. So for Chatwoot this is a **new, adapter-specific behavior**:

- **Gate it behind a config flag** — `CHATWOOT_PRIVATE_NOTE_TRACE` (default
  **off**). When off, behave like every other platform (reasoning/tool progress
  follow the normal display config). When on, route reasoning + tool/skill
  progress bubbles to `send(..., private=True)` instead of customer-visible
  bubbles, and keep the customer reply clean.
- **Use the right token.** Post private notes with `CHATWOOT_AGENT_TOKEN` (the
  user/agent token, §2.4) if set; the bot token is used for the customer reply.
- **Degrade gracefully, never at the customer's expense.** If the trace is
  enabled but no agent token is configured (and the bot token can't post
  `private: true` — confirm per §2.4), **skip the private note and log a
  warning**; the customer-facing reply must still go out. A failed private-note
  POST must never block or drop the real reply.
- **Keep it readable.** Private notes are for humans — prefer a compact summary
  (which tool/skill ran, key args, outcome) over raw token streams; reuse the
  existing tool-progress formatting (`gateway/platforms/base.py` tool-progress
  renderer) rather than inventing a new format.

---

## 5. Authorization

Reuse the gateway's existing per-platform authorization mechanism rather than
inventing one. Typically that means honoring:

- a **per-platform allowlist** of permitted sender/contact ids, and
- a **per-platform "allow all"** flag for open access.

The id you match against is the **contact/sender id** from the webhook (the user
id you set on the inbound message in §3.3). Wire Chatwoot into **every** place
the gateway consults these maps — some codebases duplicate the allowlist logic
(e.g. a startup pre-check that warns "no allowlist configured" **and** the
per-message check). Miss one and you either get a misleading warning or, worse,
silently deny every message.

> Practical gotcha: if you see inbound messages arrive but never get a reply and
> no error, check authorization first — an unconfigured allowlist commonly
> defaults to "deny all."

---

## 6. Configuration surface

Expose configuration via your gateway's normal mechanism (config file and/or
environment variables). The minimum:

| Setting | Env var | Required | Purpose |
|---|---|---|---|
| Base URL | `CHATWOOT_BASE_URL` | ✅ | Chatwoot instance URL (strip trailing slash). |
| Bot access token | `CHATWOOT_TOKEN` | ✅ | Agent Bot token → outbound reply auth (`api_access_token`). |
| Agent/user token | `CHATWOOT_AGENT_TOKEN` | optional | User token for **private notes** (§4a). Only needed if the bot token can't post `private: true` (§2.4). |
| Account id | `CHATWOOT_ACCOUNT_ID` | optional | Default account; otherwise derive from each payload. |
| Webhook secret | `CHATWOOT_WEBHOOK_SECRET` | optional (recommended when public) | Shared secret validated as `?token=` on the webhook URL. |
| Listener bind | `CHATWOOT_HOST` / `CHATWOOT_PORT` / `CHATWOOT_WEBHOOK_PATH` | optional | Own aiohttp listener bind (sane defaults, like `msgraph_webhook.py`). |
| Allowed users / allow-all | `CHATWOOT_ALLOWED_USERS` / `CHATWOOT_ALLOW_ALL_USERS` | optional | Authorization (§5); wired via `allowed_users_env`/`allow_all_env`. |
| Private-note trace | `CHATWOOT_PRIVATE_NOTE_TRACE` | optional (default off) | Route reasoning + tool/skill activity to private notes (§4a). |
| Home/target channel | `CHATWOOT_HOME_CHANNEL` | optional | `account:conversation` target for proactive/scheduled delivery (§7 cron). |

Rules of thumb:

- **A platform is "connected"** when it has both a base URL and an access token.
- **The adapter owns its own HTTP listener** (§1) — there's no separate shared
  HTTP host to enable. Make the bind host/port/path configurable (with sane
  defaults, like `msgraph_webhook.py`) and surface it in setup so users know the
  exact webhook URL to paste into Chatwoot's Agent Bot config.
- Store secrets where your gateway keeps secrets; make sure that location is
  actually loaded at startup.

---

## 7. Full wiring checklist (plugin path)

On the plugin path you hand-write only the **adapter** (§1) and its **tests/docs**.
Everything else is declared through a single `register_platform()` call in the
plugin's `register(ctx)` entry point — no core files are edited. `register_platform`
lives at `hermes_cli/plugins.py:882` and forwards every keyword to `PlatformEntry`
(`gateway/platform_registry.py:39`), so the kwargs below are the wiring.

Model the call on `plugins/platforms/teams/adapter.py:1405` (`register(ctx)`). The
concrete Chatwoot contract:

| What you must provide | How (plugin) | Notes |
|---|---|---|
| **Adapter** | `adapter_factory=lambda cfg: ChatwootAdapter(cfg)` | Subclass `BasePlatformAdapter`. Implement `connect`/`disconnect` (start/stop the **own** aiohttp listener), `send` (with the `private` flag, §4/§4a), `send_typing`, `send_image`, `get_chat_info`, attachment sends, and the webhook→event converter. |
| **Dependency check** | `check_fn=check_chatwoot_requirements` | Only `aiohttp` is truly required. |
| **"Connected" test** | `is_connected=...`, `validate_config=...` | Connected = base URL **and** bot token both present. |
| **Required env** | `required_env=["CHATWOOT_BASE_URL", "CHATWOOT_TOKEN"]`, `install_hint="pip install aiohttp"` | Surfaced by the setup wizard. |
| **Env auto-config** | `env_enablement_fn=_env_enablement` | Seed `PlatformConfig.extra` from env (base URL, tokens, account, host/port/path) + `home_channel`, so env-only setups show in `hermes gateway status`. Optional `apply_yaml_config_fn` for a `config.yaml` schema. |
| **Authorization** | `allowed_users_env="CHATWOOT_ALLOWED_USERS"`, `allow_all_env="CHATWOOT_ALLOW_ALL_USERS"` | **Required kwargs** — without them `_is_user_authorized()` has no allowlist to consult and authorization silently fails. This is the plugin equivalent of §5. |
| **Toolset** | *nothing to declare* | Plugin platforms auto-resolve the gateway toolset via registry membership (`toolsets.py:709` checks `platform_registry.is_registered()`). There is **no** `toolset` field on `PlatformEntry` and no `hermes-chatwoot` to define — the old "KeyError on unregistered platform" trap does **not** exist on this path. |
| **Send-message routing** | *automatic* | The generic send-message tool routes via `platform_registry.get()` (`tools/send_message_tool.py:683`). Provide `standalone_sender_fn` for out-of-process sends (cron) — see §7a. |
| **System-prompt hint** | `platform_hint="You are a Chatwoot support agent bot. Markdown renders; …"` | Tell the model it's a support bot and how to send media. |
| **Scheduled/cron delivery** | `cron_deliver_env_var="CHATWOOT_HOME_CHANNEL"` **+** `standalone_sender_fn=_standalone_send` | Both are required — see §7a. |
| **Status / setup UX** | `setup_fn=interactive_setup`, `emoji="💬"`, `label="Chatwoot"` | Setup flow must print the exact webhook URL (host/port/path + `?token=`) to paste into Chatwoot's Agent Bot config. Also add `requires_env`/`optional_env` rich entries in `plugin.yaml` so the wizard shows prompts/descriptions. |
| **Message length** | `max_message_length=<n>` | Backs the chunking in §4. |
| **ID redaction** | in the adapter | Redact the bot **and** agent tokens in all adapter log output. (Optionally add a pattern to `agent/redact.py` if you want masking gateway-wide, but that's a core edit — not required on the plugin path.) |
| **Docs / tests** | you write them | User setup guide + `tests/gateway/test_chatwoot.py` (§9). |

### 7a. Scheduled / proactive delivery — register **both** hooks

`deliver=chatwoot` cron jobs work with zero core edits, but only if the plugin
registers **both**:

- `cron_deliver_env_var="CHATWOOT_HOME_CHANNEL"` — makes `chatwoot` a valid
  delivery target (`_is_known_delivery_platform` → `_plugin_cron_env_var`,
  `cron/scheduler.py:690`), resolves the home target
  (`_resolve_home_env_var` / `_get_home_target_chat_id`, `cron/scheduler.py:703`),
  and joins the `deliver=origin` fallback (`_iter_home_target_platforms`,
  `cron/scheduler.py:755`). `CHATWOOT_HOME_CHANNEL` holds an
  `account:conversation` chat id (§2.1).
- `standalone_sender_fn=_standalone_send` — performs the actual send when cron
  runs **out-of-process** from the gateway (`tools/send_message_tool.py:687`).

> **The common half-wiring bug:** set only `cron_deliver_env_var` and the target
> resolves, the job fires, but the send fails with
> `No live adapter for platform 'chatwoot'` whenever cron runs separately from
> the gateway. Register both. Teams does exactly this
> (`plugins/platforms/teams/adapter.py:1424-1428`).

### 7b. Built-in path (appendix — only if upstreaming into core)

Not recommended; all live platforms are plugins. If you ever upstream Chatwoot as
a first-class built-in, you'd instead edit each core file directly: the `Platform`
enum and `_apply_env_overrides` (`gateway/config.py`), `_create_adapter` and the
`_is_user_authorized` allowlist maps (`gateway/run.py`), the toolset composite
(`toolsets.py`), `PLATFORM_HINTS` (`agent/prompt_builder.py`), the send-message
routing (`tools/send_message_tool.py`), and cron delivery — which is **not** a
`platform_map` but the `_HOME_TARGET_ENV_VARS` dict at `cron/scheduler.py:214`
plus `_KNOWN_DELIVERY_PLATFORMS`. See `gateway/platforms/ADDING_A_PLATFORM.md` §1–§16.

---

## 8. Chatwoot API reference (what you actually call)

Endpoints are under an account+conversation path on the Application API and
authenticate with the bot token in the **`api_access_token`** request header.

| Direction | Call | Body / notes | Token |
|---|---|---|---|
| Send reply | `POST /api/v1/accounts/{account_id}/conversations/{conversation_id}/messages` | `content` + `message_type: "outgoing"`; `private: false` | bot (`CHATWOOT_TOKEN`) |
| Send **private note** | same messages endpoint | `content` + `private: true` (§4a) | agent (`CHATWOOT_AGENT_TOKEN`) — verify per §2.4 |
| Send attachment | `POST` (multipart/form-data) to the same messages endpoint | file(s) as the `attachments[]` field; optional `content` caption | bot |
| Typing on/off | `POST /api/v1/accounts/{account_id}/conversations/{conversation_id}/toggle_typing_status` | on when working, off when done | bot |
| Inbound events | Chatwoot POSTs to your webhook URL | validated by optional `?token=` | — |

Every request sends its token in the **`api_access_token`** request header. The
conversation identifier in these paths is the **display id** — see §3.2.

### Representative inbound `message_created` payload

Only the fields the logic depends on are highlighted; real payloads carry much
more.

```json
{
  "event": "message_created",
  "message_type": "incoming",     // may also be integer 0 (incoming) / 1 (outgoing)
  "private": false,
  "content": "hello there",
  "conversation": {
    "id": 42,
    "display_id": 42,
    "status": "pending"           // open/pending/resolved/snoozed; "open" ⇒ human handoff, do not reply
  },
  "account": { "id": 1 },
  "sender": { "id": 88, "name": "Jane Doe", "email": "jane@example.com" },
  "attachments": [
    { "file_type": "image", "data_url": "https://…/photo.png" }
  ]
}
```

| Field | Used for |
|---|---|
| `event` | Must be the new-message event. |
| `message_type` | Direction filter (accept string **or** integer incoming). |
| `private` | Skip internal notes. |
| `conversation.status` | `open`/`pending`/`resolved`/`snoozed`; `open` ⇒ human handoff ⇒ don't reply. |
| `conversation.id` / `display_id` | Conversation half of the chat id (mind §3.2). |
| `account.id` | Account half of the chat id (or configured default). |
| `sender.id` | Authorization key + inbound user id. |
| `sender.name` / `email` | Display name. |
| `content` | Message text. |
| `attachments[]` | Cache images/audio; pass others as documents. |

---

## 9. Testing strategy

Cover behavior and invariants, not incidental data. At minimum:

- **Config / connection.** "Connected" requires base URL + token; env/config
  overrides land in the adapter; account id is optional.
- **Chat id round-trip.** Format and parse of `account:conversation`, including
  the bare-conversation fallback to a configured account.
- **Inbound filtering** — the highest-value tests:
  - incoming as string **and** as integer enum → accepted;
  - outgoing as string **and** as integer enum → ignored;
  - private note → ignored;
  - `status: "open"` → ignored; `status: "pending"` → accepted;
  - `display_id` fallback resolves the conversation id;
  - empty body → ignored;
  - attachment caching produces the right media type.
- **Webhook HTTP handler** — not-running→404, oversized body→413, bad
  secret→403, bad JSON→400, valid incoming→accepted + dispatched,
  duplicate message id→acknowledged + **not** re-dispatched (idempotency),
  non-actionable→ignored (and **not** dispatched).
- **Authorization** — allow-all authorizes; allowlist matches and blocks. On the
  plugin path assert the entry registers `allowed_users_env`/`allow_all_env` (if
  those are missing, authorization silently no-ops — §7).
- **Private-note trace (§4a)** — `send(..., private=True)` sets `private: true` on
  the payload; private notes use the agent token when `CHATWOOT_AGENT_TOKEN` is
  set; when the trace is enabled but no agent token is configured, the note is
  **skipped with a warning** and the customer reply still sends; with the trace
  **off**, reasoning/tool progress does **not** post private notes.
- **Scheduled delivery (§7a)** — the registered entry exposes **both**
  `cron_deliver_env_var` and `standalone_sender_fn`; `CHATWOOT_HOME_CHANNEL`
  resolves as an `account:conversation` target.
- **Wiring guards** — the platform is registered in the plugin registry
  (`platform_registry.is_registered("chatwoot")`), so toolset + send-routing
  resolve; the prompt hint is present. (No `hermes-chatwoot` toolset to assert —
  it's inherited via registry membership, §7.)

Put tests in `tests/gateway/test_chatwoot.py`. Assert **behavior and invariants**
— avoid change-detector tests (e.g. don't snapshot the status enum or the model
list); test relationships (incoming accepted / outgoing ignored, `pending`
answered / `open` silent) instead.

---

## 10. Behavior summary

| Behavior | Rule |
|---|---|
| Echo prevention | Only new inbound messages from the contact; skip the bot's own outgoing and private notes. |
| Direction robustness | Accept `message_type` as string **or** integer enum. |
| Human handoff | `conversation.status == "open"` ⇒ stay silent. |
| Session isolation | One session per conversation via the composite chat id + direct chat type. |
| Reply routing | Reply by conversation **display id**; parse account+conversation from the chat id. |
| Chunking | Split very long replies. |
| Attachments | Cache inbound images/audio for the agent; upload outbound files as attachments. |
| Fast ack | Acknowledge webhooks immediately; process the agent turn asynchronously. |
| Token split | Bot token (`api_access_token`) for customer replies; user token (`CHATWOOT_AGENT_TOKEN`) for private notes. |
| Private-note trace | With `CHATWOOT_PRIVATE_NOTE_TRACE` on, reasoning + tool/skill activity post as `private: true` notes; off by default; never blocks the customer reply. |
| Scheduled delivery | `deliver=chatwoot` needs both `cron_deliver_env_var` and `standalone_sender_fn`; target is `CHATWOOT_HOME_CHANNEL` = `account:conversation`. |

---

## 11. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Webhook returns 404 | Adapter not running (integration disabled) or creds missing — check the adapter's own listener started on the expected host/port/path. |
| Webhook returns 403 | Shared-secret mismatch between the URL's `?token=` and config. |
| Events arrive but always "ignored" | Outgoing/private message, empty body, or conversation status is `open` (handoff). |
| Inbound silently dropped on some versions | `message_type` sent as integer and only the string was matched — accept both. |
| Messages received but never answered, no error | Authorization: allowlist unconfigured and defaulting to deny. |
| Reply fails with 401/403 | Bad/expired bot token (or a user token used where the bot token is required). |
| Reply fails with 404 | Wrong conversation id — try `display_id` vs `id`. |
| Reply goes to the wrong conversation | Chat id not encoded/parsed as `account:conversation`, or wrong default account. |
| Private note fails 401/403 (reply works) | Bot token can't post `private: true` — set `CHATWOOT_AGENT_TOKEN` (user token) for private notes (§2.4/§4a). |
| Thinking/tool trace leaks to the customer | Private-note trace posting as customer-visible — ensure `send(private=True)` and `CHATWOOT_PRIVATE_NOTE_TRACE` gating (§4a). |
| `No live adapter for platform 'chatwoot'` on cron | Only `cron_deliver_env_var` registered; add `standalone_sender_fn` (§7a). |
| `deliver=chatwoot` rejected as unknown target | `cron_deliver_env_var` not registered on the plugin entry (§7a). |

---

## 12. Implementation order (plugin path, suggested)

Scaffold `plugins/platforms/chatwoot/` (`plugin.yaml`, `adapter.py`, `__init__.py`)
modeled on `plugins/platforms/teams/`. **Templates to copy from:**
`gateway/platforms/msgraph_webhook.py` — the closest inbound-webhook adapter: own
aiohttp server, `hmac.compare_digest` secret check, body-size guard,
`max_seen_receipts` idempotency, `is_network_accessible()` warning. And
`plugins/platforms/teams/adapter.py:712` `connect()`/`disconnect()` — the
`web.Application → AppRunner → TCPSite` server lifecycle + `self.handle_message(event)`
dispatch.

1. **Adapter skeleton** — config parsing, connect/disconnect (start/stop the own
   aiohttp listener + open/close the HTTP client), the chat-id helpers.
2. **Outbound send** — post a reply to a known conversation with the **bot** token;
   verify auth and chunking against the **live instance** (§2.4 step 1).
3. **Inbound converter** — the §3 filters and event building, with unit tests.
4. **Webhook handler** — serve the inbound endpoint on the adapter's own server,
   with body-size limit, `?token=` check, idempotency dedup, and async dispatch
   via `handle_message` (model on `msgraph_webhook.py`).
5. **`register(ctx)` + `register_platform()`** — wire the §7 kwargs; no core edits.
   Confirm "connected" logic (base URL + token).
6. **Authorization** — pass `allowed_users_env`/`allow_all_env` (§5/§7); verify a
   blocked vs allowed sender.
7. **Scheduled delivery (§7a)** — register `cron_deliver_env_var` **and**
   `standalone_sender_fn`; test a `deliver=chatwoot` cron job end-to-end.
8. **Private-note trace (§4a)** — add the `private` send flag, `CHATWOOT_AGENT_TOKEN`,
   and `CHATWOOT_PRIVATE_NOTE_TRACE` gating; **verify on the live instance** whether
   the bot token can post `private: true` (§2.4 step 2) and wire the token choice.
9. **Setup/status + docs** — setup flow prints the exact webhook URL; write the user guide.
10. **Tests** per §9, then a live end-to-end pass: a real inbox message, a deliberate
    `open`-status message (confirm handoff silence), and a private-note trace check.
