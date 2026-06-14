# VK Messenger Platform Adapter for Hermes Agent

A community-contributed plugin adapter that connects Hermes Agent to **VK Messenger** (VKontakte), the most popular messaging platform in Russia.

## Features

| Feature | Status |
|---------|--------|
| Text messages (send/receive) | ✅ |
| Typing indicator (`setActivity`) | ✅ |
| Inline keyboards (approve/deny/confirm/clarify) | ✅ |
| Image upload (native VK photo attachment) | ✅ |
| Document upload (files, PDFs, etc.) | ✅ |
| Voice messages (send/receive) | ✅ |
| Receive images/docs/audio from users | ✅ |
| Message splitting (4096 char limit) | ✅ |
| HTML → .txt rename (VK blocks .html) | ✅ |
| Standalone send (cron delivery) | ✅ |
| Markdown stripping (VK bots don't render it) | ✅ |
| REST API polling (no public URL needed) | ✅ |
| Group chat support | ✅ |

## Architecture

```
VK User → VK Servers → REST polling (5s) → VKAdapter → Hermes Agent
                     ← messages.send       ←
```

The adapter uses **two-step REST API polling**:
1. `messages.getConversations` — detects active conversations
2. `messages.getHistory` — fetches ALL new messages since last processed ID

This approach works reliably from Russian networks where `lp.vk.com` (Long Poll) has DNS issues, and ensures follow-up messages (`/approve`, `/deny`) are never missed even when the bot has sent responses in between.

Each message is dispatched as an independent `asyncio.Task` so the poll loop continues to run — preventing deadlocks during dangerous-command approval flows.

## Installation

### 1. Place files in Hermes plugins directory

```bash
mkdir -p ~/.hermes/plugins/vk/
cp adapter.py plugin.yaml __init__.py ~/.hermes/plugins/vk/
```

### 2. Create a VK Community

- Go to **vk.com/groups** → Create a community (any type, can be private/closed)
- **Manage** → **Messages** → Enable community messages
- **Manage** → **API** → Create token → enable `Messages` permission

### 3. Configure Hermes

Add to `~/.hermes/.env`:

```bash
VK_TOKEN=vk1.a.....your_token.....
VK_GROUP_ID=123456789
```

Enable in config:

```bash
hermes config set gateway.platforms.vk.enabled true
```

### 4. Restart Gateway

```bash
systemctl --user restart hermes-gateway
```

## Configuration Reference

| Env Var | Required | Description |
|---------|----------|-------------|
| `VK_TOKEN` | ✅ | VK community API token |
| `VK_GROUP_ID` | ✅ | Numeric community ID |
| `VK_API_VERSION` | ❌ | API version (default: 5.199) |
| `VK_ALLOWED_USERS` | ❌ | Comma-separated user IDs allowed to use the bot |
| `VK_ALLOW_ALL_USERS` | ❌ | Allow all users (true/false) |
| `VK_HOME_CHANNEL` | ❌ | Peer ID for cron/notification delivery |

## VK-Specific Behaviour

- **Inline buttons in group chats**: VK prepends an `@mention` of the bot to button labels. The adapter handles this transparently.
- **Formatting**: VK community bots don't render Markdown. The adapter strips `**bold**`, `*italic*`, `` `code` ``, etc.
- **Message limit**: VK enforces 4096 chars per message. Longer messages are auto-split with ` (1/N)` suffixes.
- **HTML files**: VK blocks `.html` uploads — they're renamed to `.txt` transparently.

## Code Overview

| File | Lines | Purpose |
|------|-------|---------|
| `plugin.yaml` | 35 | Plugin metadata, env var declarations |
| `__init__.py` | 3 | Exports `register()` |
| `adapter.py` | ~1150 | Full adapter implementation |

---

**Author:** Igor Borges  
**License:** MIT (same as Hermes Agent)  
**Status:** Production-tested since June 2026
