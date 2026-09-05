<div align="center">

<br/>

<img src="assets/github-logo.jpg" alt="MAX Hermes Plugin" width="720">

### MAX — a Russian messenger, plugged into Hermes Agent.

**<span style="color:#f59e0b">Message your bot in MAX, get your own Hermes back. Memory, tools, and no public IP needed.</span>**

<br/>

[![Version](https://img.shields.io/badge/version-0.22.0-blue.svg)](https://github.com/FraN-arti/max-hermes-plugin)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Hermes](https://img.shields.io/badge/Hermes-0.20+-7C3AED.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MAX](https://img.shields.io/badge/MAX-API-FF6B00.svg)](https://dev.max.ru)

[English](README.en.md) · [Русский](README.md)

</div>

---

<br/>

## What is this, in plain English

**MAX** (max.ru) is a Russian messenger by VK. This plugin connects it to
[Hermes Agent](https://hermes-agent.nousresearch.com) as a real messaging
channel — just like Telegram, WhatsApp or Slack.

You write to a bot in MAX. Your Hermes answers. The same agent that lives on
your machine — with its memory, its tools, its scheduled jobs. Not an echo bot,
not a chatbot. *Your* agent, just in your messenger.

<br/>

## What it can do

The full surface area of the plugin, as of v0.21.0. Every item is implemented
and tested; nothing here is aspirational.

### Messaging

- **Direct messages** (1:1) with full Hermes Agent — memory, tools, cron delivery
- **Group chats** with role-based access (owner / admin / member)
- **Channels** — works as a participant in MAX channels too
- **Long Polling** transport — no public IP, no HTTPS cert, no domain, no port
  forwarding. Works behind NAT, on Russian ISPs, on mobile
- **Marker persistence** — `max/marker.json` so gateway restarts don't replay
  old messages
- **Smart 4000-char splitting** — long messages are split at line/word
  boundaries, never mid-word
- **Rate limiting** — 2 msg/sec per chat, respected automatically

### Attachments

- **Images** (JPEG, PNG, GIF, WebP) — downloaded to local cache so the vision
  tool can read them
- **Audio files** (MP3, OGG, WAV, M4A, FLAC, OPUS, AAC) — STT pipeline picks up
  automatically when extension suggests audio
- **Video** — downloaded for the agent's tools
- **Documents** — downloaded to cache
- **Voice messages (hold-to-talk)** — *not* delivered via Long Polling
  (empty update). Regular audio files work fine.

### Group chats — addressing & awareness

- **Bot responds only when addressed** — by `@username`, by display name
  (`Kain, what's the build?`), or by any alias you configure in
  `MAX_BOT_ALIASES`. The word "бот" does *not* count.
- **Sender context** — every message carries `user_id` + name, so the agent
  always knows who's talking
- **Group session modes:**
  - per-user (default) — each member has their own session, isolated memory
  - shared (`MAX_GROUP_SESSIONS_PER_USER=false`) — one session for the whole
    group, recommended for party / raid bots
- **Per-group mini-prompt (`channel_prompt`)** — bot identity (name, @username,
  description from MAX) is auto-injected at the start of every group session.
  Custom per-group instructions can be layered via `channel_prompts` config.

### Group chats — security & roles

- **Bot owner** (`MAX_OWNER_USER_ID`) — full access: terminal, files, anything.
  Only the owner can run `/approve` and `/deny`.
- **Group owner / admin** — can request moderation, can run safe slash
  commands (`/new`, `/reset`, `/compress`, `/status`, `/help`).
- **Regular members** — chat + questions + web search. No slash commands.
- **Approval for new groups** — when the bot is added to an unapproved group,
  it stays silent and DMs the owner with `/approve <chat_id>` / `/deny <chat_id>`.
  No group reaches the agent without your say-so.
- **Allowlist** — `MAX_GROUP_ALLOWED_CHATS` pins the bot to specific groups
- **Member-role cache** — `GET /chats/{id}/members` cached for `MAX_MEMBERS_TTL`
  seconds (default 300). Force-refreshed before any moderation action so a
  freshly promoted admin doesn't get refused.
- **Freshness timeout** — member fetch is bounded by 3s, so a slow MAX API
  never delays the bot's reply; stale cache is used as fallback.

### Group chats — moderation

- **Delete messages** — `DELETE /messages` (requires bot to be group admin)
- **Kick / ban members** — `DELETE /chats/{id}/members/{uid}`
- **Resolve by name or @username** — the bot knows everyone in the group's
  cached member list, so `kain, ban Vasya` resolves the name to a user_id
- **Permission check** — only the bot owner or a group owner/admin can ask;
  the bot itself must be group admin to actually do it
- **Refuses politely** — non-admin askers get a one-line explanation

### Slash commands

- **Owner-only:** `/approve <chat_id>`, `/deny <chat_id>`
- **Group admins:** `/new`, `/reset`, `/compress`, `/status`, `/help`
- **Members:** nothing (silently dropped)
- **DM:** all standard Hermes commands pass through to the agent

### Setup & operations

- **`hermes setup gateway` wizard** — interactive flow: token → DM allowlist →
  groups → session mode → owner → pre-approved groups
- **`check_requirements()`** — diagnostic for the gateway health check
- **`validate_config(config)`** — pre-flight validation
- **`is_connected(config)`** — runtime liveness probe
- **Standalone send** (`_standalone_send`) — for cron fallbacks and
  `send_message_tool`
- **Auto-download of Russian Trusted Root CA** — on first run, from
  [gu-st.ru](https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt),
  10s timeout, PEM-validated. Override with `MAX_CA_CERT_PATH`.
- **Reconnect with exponential backoff + jitter** — survives transient
  network blips
- **Typing indicator** — `POST /chats/{id}/actions` while the agent thinks
- **Health tracking** — `_last_poll_at`, `_last_poll_error`, `_last_error_at`
  for observability

### What's *not* in scope (yet)

- Webhook transport (requires public HTTPS endpoint)
- Hold-to-talk voice messages via Long Polling
- Multi-bot orchestration (one MAX account = one bot per Hermes profile)
- Reactions, polls, inline keyboards

<br/>

## Why MAX?

- 🇷🇺 **Works in Russia** — no VPN, no blocks, data stays in RF
- 📡 **No public IP needed** — your machine just polls MAX. Works behind NAT,
  on Rostelecom, on mobile. Nothing to forward, no HTTPS cert, no domain.
- 🔐 **Allowlist by default** — only people you approve can talk to the bot
- 🪄 **Zero-config TLS** — the Russian Trusted Root CA is downloaded on first run
- 👥 **First-class group support** — with role-based access, approval for new
  groups, and moderation (delete messages, kick spammers)

<br/>

## Quick start (5 minutes)

### 1. Create a bot in MAX

You need a verified partner profile (legal entity / sole proprietor /
self-employed). [Connection guide](https://dev.max.ru/docs/maxbusiness/connection).

1. Create a bot: [business.max.ru](https://business.max.ru/self) → **Chat bots**
2. Wait for moderation
3. Grab the token: **Chat bots** → **Advanced settings** → **Configure**

### 2. Install

```bash
# Linux / macOS
cp -r plugins/platforms/max ~/.hermes/plugins/platforms/

# Windows: drop plugins/platforms/max into
#          %LOCALAPPDATA%\hermes\plugins\platforms\

# Enable
hermes plugins enable max
```

### 3. Configure

```bash
# The wizard (recommended):
hermes setup gateway      # pick MAX → paste token

# Or manually — put this in ~/.hermes/.env:
MAX_BOT_TOKEN=your_token_here
MAX_ALLOWED_USERS=your_user_id
```

> **How do I find my user_id?** Send any message to the bot and look at the
> gateway log — `Message from <name>` — the user_id is right there. Or set
> `MAX_ALLOW_ALL_USERS=true` temporarily, send a message, then disable it.

### 4. Run

```bash
hermes gateway restart
```

Done. Message the bot in MAX. 🎉

<br/>

## How it works

```
 You (MAX)  →  MAX API (platform-api2.max.ru)  ←─ Long Polling ←─  Hermes Gateway
                                                                      ↓
 You (MAX)  ←  POST /messages  ←────────────────────────────  Hermes (reply)
```

- **Inbound** uses Long Polling (`GET /updates` with a marker cursor). Your
  machine polls MAX — no inbound access required, nothing to expose.
- **The marker persists** to `max/marker.json`, so a gateway restart won't
  replay old messages.
- **Outbound** is `POST /messages` — `user_id` for DMs, `chat_id` for groups.
  MAX's 2 msg/sec rate limit is respected automatically.
- **Formatting:** every reply goes out as MAX HTML: code blocks become a
  `<pre>` inside a blockquote frame, inline code gets a `<mark>` highlight.
  MAX Markdown supports neither multi-line code nor inline styling (raw
  asterisks stay literal), so the conversion is always applied.
- **Reasoning is hidden:** the 💭 thinking block is stripped from replies by
  default (even with `display.show_reasoning: true`). Bring it back with
  `MAX_SHOW_REASONING=true`.
- **Typing indicator** works through `POST /chats/{id}/actions`.
- **TLS** uses the Russian Trusted Root CA, auto-downloaded from
  [gu-st.ru](https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt)
  on first run. You can override the path with `MAX_CA_CERT_PATH`.

<br/>

## Group chats — the interesting part

This is what makes the plugin more than just another DM bridge.

### The bot answers only when you call it

In a group, the bot ignores everything except direct mentions: `@kain_bot`,
`Kain, what's the build?`, or any alias you configure in `MAX_BOT_ALIASES`.
Regular party chatter — *ignored*. No spam, no noise. By default,
`bot` / `bot` style words do **not** count as addressing — you set the bot's
name and that's what people use.

### You can see who's talking

Every message carries the sender's `user_id` and display name, so the agent
always knows who it's talking to. Useful when several people are discussing
with the bot at once.

### Approval for new groups (security)

If someone adds your bot to a group that isn't pre-approved, the bot
**stays silent there** and DMs the owner with two options:

```
/approve <chat_id>   ← let it work in this group
/deny    <chat_id>   ← refuse (bot stays but stays quiet)
```

Nothing reaches your agent without your say-so. `MAX_OWNER_USER_ID` is the
only user that can run `/approve` and `/deny` — no one else.

### Roles & moderation

The bot fetches the group's member list (`GET /chats/{id}/members`) and
understands roles:

- **You** (bot owner) — full access: terminal, files, anything.
- **Group owner / admin** — can ask the bot to moderate (delete messages,
  kick members). Only if the bot itself is a group admin.
- **Regular member** — can chat with the bot, can ask questions, can search
  the web. No slash commands. No moderation.

Role checks happen *before* the message reaches the agent. Members of a group
can change over time — the bot refreshes its member cache every
`MAX_MEMBERS_TTL` seconds (default: 300), and force-refreshes before any
moderation action so a freshly promoted admin doesn't get refused.

### Slash commands by role

- **Bot owner:** everything.
- **Group owner / admin:** `/new`, `/reset`, `/compress`, `/status`, `/help`
  only. Stuff like `/platform pause` is refused.
- **Regular members:** no slash commands at all.

### Shared group context (party bots)

By default, every group member has their own private session with the bot.
For a party bot — where everyone sees the same conversation — set
`MAX_GROUP_SESSIONS_PER_USER=false`. Then the whole group shares one
session, and the bot remembers the full party chat.

### Per-group mini-prompt

The bot's identity (name, @username, description you set in MAX) is
automatically injected into the start of every group session, so the bot
always knows who it is and how it's addressed. You can layer extra
instructions per group via `channel_prompts` in the plugin config.

<br/>

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `MAX_BOT_TOKEN` | ✅ | Bot token from business.max.ru |
| `MAX_ALLOWED_USERS` | ❌ | Allowed user_ids (comma-separated) |
| `MAX_ALLOW_ALL_USERS` | ❌ | `true` = allow everyone (dev only!) |
| `MAX_HOME_CHANNEL` | ❌ | chat_id for cron delivery to a group |
| `MAX_HOME_USER_ID` | ❌ | user_id for cron delivery to your DM |
| `MAX_CA_CERT_PATH` | ❌ | Custom path to the Russian Trusted Root CA |
| `MAX_GROUP_ALLOWED_CHATS` | ❌ | Allowed chat_ids for groups (comma-separated) |
| `MAX_GROUP_SESSIONS_PER_USER` | ❌ | `false` = shared session for the whole group (party bots); `true` = per-user (default) |
| `MAX_OWNER_USER_ID` | ❌ | Your MAX user_id. Full access; only owner can run `/approve` and `/deny` |
| `MAX_APPROVED_CHATS` | ❌ | Pre-approved chat_ids. Empty = bot asks owner when added to a new group |
| `MAX_MEMBERS_TTL` | ❌ | Seconds between member-role refreshes (default 300) |
| `MAX_SHOW_REASONING` | ❌ | `true` = show the 💭 Reasoning block in bot replies (hidden by default) |
| `MAX_BOT_ALIASES` | ❌ | Extra names the bot answers to, e.g. `kain,kai` |

<br/>

## Structure

```
plugins/platforms/max/
├── __init__.py      # plugin registration
├── adapter.py       # Long Polling + send + auto-cert + roles + moderation
├── plugin.yaml      # metadata for `hermes setup gateway`
├── README.md        # Russian version
└── README.en.md     # this file
```

<br/>

## Limitations

- MAX recommends **webhooks** for production, but they require HTTPS + a
  public URL. Long Polling works everywhere — even behind NAT — and is
  perfect for personal use.
- **Voice messages (hold-to-talk)** aren't delivered via Long Polling (the
  update arrives empty). Regular audio files (MP3) work fine.
  See [#6](https://github.com/FraN-arti/max-hermes-plugin/issues/6).
- Messages over 4000 chars are **split into multiple** (smart split at
  line/word boundaries).

<br/>

## Roadmap

- ✅ **Group chats & @mentions** ([#2](https://github.com/FraN-arti/max-hermes-plugin/issues/2)) — done in 0.20
- ✅ **Attachments** ([#1](https://github.com/FraN-arti/max-hermes-plugin/issues/1)) — done
- 🔜 **Webhook mode** for production ([#3](https://github.com/FraN-arti/max-hermes-plugin/issues/3))
- 🔜 **CI: automated tests & linter** ([#4](https://github.com/FraN-arti/max-hermes-plugin/issues/4))

Full list — in [Issues](https://github.com/FraN-arti/max-hermes-plugin/issues).

<br/>

## License

[MIT](LICENSE)

<br/>

> ⭐ **Find this useful?** A star helps the project grow.
> Questions or ideas — drop them in [Issues](https://github.com/FraN-arti/max-hermes-plugin/issues).

---

<div align="center">

Built for the Russian Hermes community 🇷🇺 · [MAX for developers](https://dev.max.ru)

</div>