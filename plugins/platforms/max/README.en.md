<div align="center">

<br/>

<img src="assets/github-logo.jpg" alt="MAX Hermes Plugin" width="720">

### MAX — Russian messenger as a channel for Hermes Agent.

**<span style="color:#f59e0b">Message a bot in MAX — your Hermes answers. With memory, tools and no public IP needed.</span>**

<br/>

[![Version](https://img.shields.io/badge/version-0.20.0-blue.svg)](https://github.com/FraN-arti/max-hermes-plugin)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Hermes](https://img.shields.io/badge/Hermes-0.20+-7C3AED.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MAX](https://img.shields.io/badge/MAX-API-FF6B00.svg)](https://dev.max.ru)

[English](README.en.md) · [Русский](README.md)

</div>

---

<br/>

## What is this

**MAX** (max.ru) is a Russian messenger by VK. This plugin connects it to
[Hermes Agent](https://hermes-agent.nousresearch.com) as a full messaging channel —
just like Telegram, WhatsApp or Slack.

You message a bot in MAX → **your** Hermes answers: with shared memory, tools,
cron jobs and everything it can do. Not an echo stub — the same agent that lives
on your machine.

<br/>

## Why MAX

- 🇷🇺 **Russian messenger** — works without VPN, data stays in RF
- 📡 **Long Polling** — no public IP, domain, port forwarding or HTTPS cert needed.
  Works behind NAT (typical RU ISPs are fine)
- 🔐 **Allowlist** — only approved users can talk to the bot
- ⏰ **Cron delivery** — reminders and notifications from Hermes arrive in MAX
- 🪄 **Zero-config TLS** — Russian Trusted Root CA is auto-downloaded on first run

<br/>

## Quick start (5 minutes)

### 1. Create a bot in MAX

You need a verified partner profile (legal entity / sole proprietor / self-employed):
[connection guide](https://dev.max.ru/docs/maxbusiness/connection)

1. Create a bot: [business.max.ru](https://business.max.ru/self) → **Chat bots** → create
2. Wait for moderation
3. Get the token: **Chat bots** → **Advanced settings** → **Configure**

### 2. Install the plugin

```bash
# Copy the plugin folder into Hermes
# Linux/macOS:
cp -r plugins/platforms/max ~/.hermes/plugins/platforms/
# Windows:
#   put plugins/platforms/max into %LOCALAPPDATA%\hermes\plugins\platforms\

# Enable the plugin
hermes plugins enable max
```

### 3. Configure

```bash
# Via wizard (recommended):
hermes setup gateway        # → pick MAX → paste token

# Or manually — add to .env:
MAX_BOT_TOKEN=your_bot_token
MAX_ALLOWED_USERS=your_user_id
```

> **How to find your user_id?** Send any message to the bot and check the gateway
> log: `Message from <name>` — the user_id is right there. Or temporarily set
> `MAX_ALLOW_ALL_USERS=true`, find the ID, then remove it.

### 4. Start

```bash
hermes gateway restart
```

Done! Message the bot in MAX — it will answer. 🎉

<br/>

## How it works

```
 You (MAX)  →  MAX API (platform-api2.max.ru)  ←─ Long Polling ←─  Hermes Gateway
                                                                      ↓
 You (MAX)  ←  POST /messages  ←──────────────  Hermes Gateway (reply)
```

- **Inbound:** Long Polling `GET /updates` with marker cursor. Your machine polls
  MAX servers — no inbound access needed.
- **Marker is persisted** to `max/marker.json` — no update replay after gateway restart.
- **Outbound:** `POST /messages` with `user_id` (DM) or `chat_id` (groups/channels).
  MAX rate limit (2 msg/sec per chat) is respected automatically.
- **Typing indicator:** via `POST /chats/{id}/actions` (`typing_on`).
- **TLS:** MAX uses Russian Trusted Root CA. The plugin auto-downloads it from the
  official source [gu-st.ru](https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt)
  on first run (10s timeout, PEM validation), or uses `MAX_CA_CERT_PATH`.

<br/>

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `MAX_BOT_TOKEN` | ✅ | Bot token from business.max.ru |
| `MAX_ALLOWED_USERS` | ❌ | Allowed user_ids (comma-separated) |
| `MAX_ALLOW_ALL_USERS` | ❌ | `true` = allow everyone (dev only!) |
| `MAX_HOME_CHANNEL` | ❌ | chat_id for cron delivery |
| `MAX_HOME_USER_ID` | ❌ | user_id for cron DM delivery |
| `MAX_CA_CERT_PATH` | ❌ | Custom path to the Russian Trusted Root CA |

<br/>

## Structure

```
plugins/platforms/max/
├── __init__.py      # plugin registration
├── adapter.py       # Long Polling + send + auto-cert
├── plugin.yaml      # metadata for hermes setup gateway
└── README.md        # this file
```

<br/>

## Limitations

- MAX recommends **webhooks** for production, but they require HTTPS + public URL.
  Long Polling works everywhere (even behind NAT) — ideal for personal use.
- **Attachments** (images, files) are not sent yet — text only. Planned.
- Messages over 4000 chars are **split into multiple** (smart split at line/word boundaries).

<br/>

## License

[MIT](LICENSE)

---

<div align="center">

Made for the Russian Hermes community 🇷🇺 · [MAX for developers](https://dev.max.ru)

</div>
