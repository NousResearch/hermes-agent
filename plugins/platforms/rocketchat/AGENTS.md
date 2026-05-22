# Rocket.Chat Platform Plugin — AI Agent Guide

Reference for AI coding assistants working on this plugin.

## Overview

A Hermes gateway platform adapter for self-hosted Rocket.Chat instances.
~1,460 lines, single-file (`adapter.py`), built on `aiohttp` (zero new deps).

**Architecture:** REST API v1 for outbound writes, DDP WebSocket for inbound receive.

## File Map

| File | Purpose |
|------|---------|
| `adapter.py` | Full adapter — transport, parsing, media, reactions, cron sender |
| `plugin.yaml` | Plugin manifest — env vars, discovery metadata |
| `__init__.py` | Exports `register()` for Hermes plugin discovery |
| `README.md` | User-facing setup guide (German) |
| `AGENTS.md` | This file — AI agent development reference |

## Critical Design Decisions

### 1. Slash Command: Position 0 Only

The adapter scans both `raw_msg` (with @mention prefix) and `message_text` (mention stripped)
for `/` — but **only at position 0**. Mid-sentence `/status` is NOT a command.

```python
# CORRECT — only position 0
slash_pos = candidate_text.find("/")
if slash_pos == 0:
    # ... parse command
```

Historical note: the original PR#14869 matched `slash_pos >= 0 and (slash_pos == 0 or candidate_text[slash_pos - 1] in (" ", "\t", "\n"))` which caused false positives. Fixed in `433b7a15d`.

### 2. Dual-Text Scanning for DMs

In DMs, the @mention is sometimes NOT stripped from `raw_msg` (varies by RC version).
The adapter checks both `raw_msg` and `message_text` — one of them will have `/` at
position 0 for a real command.

### 3. Room Type Detection

Uses `GET /api/v1/rooms.info` with a per-room cache. Returns `c`, `p`, `d` (channel,
private group, direct message). Cache is lazy — populated on first message from a room.

### 4. TTS Audio Pipeline

Voice messages arrive as WebM/OGG attachments via RC. The adapter:
1. Downloads the attachment via `_download_attachments()`
2. Converts to MP3 via `ffmpeg` (`_convert_audio_to_mp3()`)
3. Delivers the MP3 path to Hermes for STT processing

RC's `rooms.media` has no direct audio transcoding, so ffmpeg is required.

### 5. DDP Protocol

- Connect: WebSocket to `wss://<server>/websocket`
- Auth: `{"msg": "connect", "version": "1", "support": ["1", "pre2", "pre1"]}`
  → `resume` with PAT token
- Subscribe: `{"msg": "sub", "name": "stream-room-messages", "params": ["__my_messages__", {}]}`
- System messages filtered by `"t"` field (join/leave/role changes, etc.)
- Reconnect: exponential backoff 2s–60s

### 6. Bidirectional Topic Sync

Hermes session titles sync back to RC room topics via `dm.setTopic` (DMs) or
`groups.setTopic`/`channels.setTopic` for group rooms. In `_sync_title_to_rc_topic()`.

Power-on self-topic: On connect, the adapter sets the room topic to
"🤖 Hermes Agent — connected at <timestamp>" to confirm connectivity.

### 7. Async Standalone Sender (Cron)

`_standalone_send()` is a REST-only sender used by Hermes cron delivery — no
WebSocket dependency. Instantiates its own `aiohttp.ClientSession`, sends via
`chat.postMessage`, cleans up. No adapter lifecycle needed.

### 8. RC Admin: Unerkannte Slash Commands Weiterleiten

Rocket.Chat Desktop-Client (Browser) fängt unbekannte `/`-Befehle client-seitig ab —
die Nachricht erreicht Hermes gar nicht. Mobile Clients sind nicht betroffen.

**Fix:** `Message_AllowUnrecognizedSlashCommand = true` in RC Admin
(Administration → Workspace → Settings → Message)

**Env-Var-Alternative:** `OVERWRITE_SETTING_Message_AllowUnrecognizedSlashCommand=true`

Nur RC-Admins mit `edit-privileged-setting` Permission können das setzen.

## Known Pitfalls

| Pitfall | Detail | Mitigation |
|---------|--------|------------|
| `totp-required` | PAT without "Ignore Two Factor" generates TOTP challenge | User must re-create PAT with checkbox |
| DDP subscription lost on reconnect | RC does NOT resume DDP subs across reconnects | Full re-login + re-sub in `_ws_loop()` |
| Image URLs truncated | RC has a ~2KB URL limit in messages | `_send_url_as_file()` uploads as file attachment |
| Room type ambiguity | `rooms.info` can fail for archived rooms | `_resolve_room_type()` falls back to `'c'` |
| ffmpeg not installed | Audio processing breaks silently | `_convert_audio_to_mp3()` returns None, logs warning |
| Nginx close WS on 60s idle | Default proxy timeout kills long connections | Set `proxy_read_timeout 600s` |
| `Message_AllowUnrecognizedSlashCommand` | Desktop browser shows "invalid command" error | RC admin setting required (not an adapter fix) |

## Tools & Functions Reference

**Transport:**
- `connect()`, `disconnect()`, `_ws_loop()`, `_ws_connect_and_listen()`

**Send:**
- `send(chat_id, text, msg_id)`, `send_image()`, `send_image_file()`, `send_document()`,
  `send_voice()`, `send_video()`, `send_typing()`, `stop_typing()`

**Receive:**
- `_handle_message(post)` — main inbound dispatch (1,160 lines)
- `_handle_ddp_frame(event)` — DDP frame routing

**Media:**
- `_download_attachments()`, `_upload_file()`, `_convert_audio_to_mp3()`

**Reactions:**
- `_add_reaction(msg_id, emoji)`, `_remove_reaction(msg_id, emoji)` — 👀✅❌

**Meta:**
- `edit_message(chat_id, msg_id, text)`, `get_chat_info()`
- `_sync_title_to_rc_topic()`, `_resolve_room_type()`
- `format_message(content)` — RC-specific markdown (single `*` for bold)
- `check_requirements()`, `validate_config()`, `_env_enablement()`

## PR History

This plugin is a refactor of **PR #14869** (`@cyb0rgk1tty`, `gateway/platforms/` core adapter)
into the modern Hermes plugin format (`plugins/platforms/`, `kind: platform`).
Parallel independent work: **PR #4637** (`@meron1122`, same plugin structure).

**Key commits (local):**
| SHA | Change |
|-----|--------|
| `ce4852bb3` | Initial port from PR#14869 → plugin format |
| `7103c75ea` | TTS audio pipeline (ffmpeg MP3 conversion) |
| `84ddeb401` | RC-native slash command routing |
| Various | Debug logging, reaction fixes, topic sync |
| `433b7a15d` | **/status mid-sentence fix** (position 0 only) |

## Testing

```bash
# Unit tests exist in tests/gateway/test_rocketchat.py (from PR#14869)
# Run:
pytest tests/gateway/test_rocketchat.py -q
```

Live test: DM the bot or @mention in a channel after config.
