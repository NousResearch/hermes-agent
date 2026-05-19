---
name: giphy-multi
description: >-
  Search and send reaction GIFs from Giphy on Telegram, Discord, WhatsApp,
  Signal, Slack, and more. Auto-configures per-platform conversion (MP4,
  GIF optimized, text-only fallback).
version: 1.0.0
author: chdlc
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [python3, ffmpeg, curl]
  env_vars: [GIPHY_API_KEY]
metadata:
  hermes:
    tags: [GIF, Giphy, media, multi-platform, reaction]
    related_skills: []
---

# Giphy Multi — Cross-platform GIF skill

Search Giphy and convert animated GIFs optimized for your current messaging platform. Each platform gets its own conversion profile (MP4 for Telegram, baseline MP4 for WhatsApp, optimized GIF for Slack, etc.).

## When to Use

- **Send a reaction GIF** in any conversation — Telegram, Discord, WhatsApp, etc.
- **User says** "send a GIF of…", "reaction gif", "manda un gif de…"
- **Spontaneous reactions** in `natural` mode — the agent judges when a GIF fits
- **First-time setup** on a new platform — run discovery

## Setup

### 1. Get a Giphy API Key

1. Go to <https://developers.giphy.com> → "Create an App"
2. Select **API** (free tier: 1,000 requests/day)
3. Copy your API Key

### 2. Configure the key

Add to `~/.hermes/.env`:

```
GIPHY_API_KEY=your_key_here
```

### 3. Verify

```bash
python3 ~/.hermes/skills/media/giphy-multi/scripts/giphy_multi.py --check
```

Expected output:
```
✅ GIPHY_API_KEY: key_…_xxxx
✅ python3
✅ ffmpeg
✅ curl
ℹ️  Run --discover to detect active channels
```

### 4. Discover platforms

The agent calls `send_message(action='list')`, parses the available platforms, then configures the skill:

```bash
python3 ~/.hermes/skills/media/giphy-multi/scripts/giphy_multi.py \
  --discover --platforms telegram,discord
```

Example output:
```json
{"ok": true, "channels": ["telegram", "discord"], "mode": "natural", "config_path": "…/config.json"}
```

This only needs to run once per platform — the config persists in `<skill_dir>/config.json`.

## Daily Workflow

### 1. Determine channel and target

The agent determines the **platform type** from session context (e.g. `telegram`, `discord`).
For delivery, call `send_message(action='list')` to see all available targets and pick the appropriate one. **Do not assume topics/threads exist** — they depend on user configuration.

```
Available:
  telegram:Christian (dm)              ← DM sin topic
  telegram:Christian / General (dm)    ← DM con topic General
  telegram:Christian / System (dm)     ← DM con topic System
  discord:#general                     ← Canal de servidor
  ...
```

The agent selects the target that matches where the conversation is happening.

### 2. Search and convert

```bash
python3 ~/.hermes/skills/media/giphy-multi/scripts/giphy_multi.py \
  "<query>" --channel <platform>
```

Returns JSON with the converted file path:

```json
{
  "gif_url": "https://media4.giphy.com/…",
  "gif_id": "abc123",
  "title": "excited cat GIF",
  "channel": "telegram",
  "format": "mp4",
  "path": "/home/…/.giphy_cache/giphy_telegram_12345.mp4"
}
```

### 3. Send via Hermes

The agent sends a text message with the GIF as media attachment using `MEDIA:<path>`.
Use `send_message(action='list')` to find the correct target for the current conversation:

```python
# Get available targets first
send_message(action='list')
# → "Available: telegram:Christian (dm), telegram:Christian / General (dm), discord:#general ..."

# Then send to the matching one
send_message(
  action='send',
  target='telegram:Christian / System (dm)',  # or whichever matches
  message="Caption aquí 🐱 MEDIA:/path/to/file.mp4"
)
```

**Do not hardcode chat_id/thread_id.** Topics are optional per-platform and per-user — let the `send_message(action='list')` output guide the target selection.

Hermes delivers the text and the file as native media on the platform (text + file arrive as separate messages).

### 4. Cleanup

The cache under `~/.hermes/.giphy_cache/` auto-purges files older than 10 minutes on each search.

## Usage Modes

The config's `"mode"` field controls when GIFs are sent:

- **`natural`** (default) — spontaneous, like emoji reactions.
- **`on_request`** — only when the user explicitly asks ("send a gif of…", "reaction gif").

Change mode:

```bash
python3 ~/.hermes/skills/media/giphy-multi/scripts/giphy_multi.py --mode on_request
```

The user can also say it in conversation:
- "deja de mandar gifs sin preguntar" → switch to `on_request`
- "manda gifs cuando quieras" → switch to `natural`

## Platform Profiles

| Platform | Format | Max size | Notes |
|---|---|---|---|
| Telegram | MP4 (H.264) | 50 MB | Standard, works in topics/DMs |
| Discord | MP4 (H.264) | 25 MB | Within Nitro limits |
| WhatsApp | MP4 baseline | 16 MB | Baseline profile for compatibility |
| Signal | MP4 (H.264) | 50 MB | |
| iMessage | MP4 (H.264) | 50 MB | |
| Slack | GIF optimized | 5 MB | Scaled to 320px, 10fps |
| Matrix | MP4 (H.264) | 50 MB | |
| IRC/Nostr/Twitch | text_only | — | Returns a link instead |

## Rating

Default: `g`. Override with `--rating pg`, `--pg-13`, or `--r`:

```bash
python3 ~/.hermes/skills/media/giphy-multi/scripts/giphy_multi.py \
  "funny fail" --channel telegram --rating pg-13
```

## Common Pitfalls

1. **GIPHY_API_KEY not set.** The only symptom is the script printing help text in JSON `"error"` field. Run `--check` first.

2. **ffmpeg not installed.** Required for conversion. Install with `sudo apt install ffmpeg` or equivalent.

3. **Omitting `--channel` when multiple platforms are configured.** The script will error. Always pass `--channel` from the current session context.

4. **Sending to the wrong topic on Telegram.** The `send_message` target must include the thread_id for topic chats. Check `send_message(action='list')` for available targets.

5. **Cache files accumulating.** Auto-purge runs on every search (files >10 min removed). Run `rm -rf ~/.hermes/.giphy_cache/` to force-clean.

6. **Giphy rate limit (1,000/day).** If hit, the API returns an error. Wait until the next day or upgrade to a paid plan.

## Reference files

- `references/hermes-media-delivery.md` — how MEDIA: flows through the gateway and why caption doesn't carry through with `send_message`. Research from 18-May-2026.

## Verification Checklist

- [ ] `GIPHY_API_KEY` is set in `~/.hermes/.env`
- [ ] `python3 giphy_multi.py --check` shows all ✅
- [ ] `--discover --platforms` configured the skill for your platforms
- [ ] A test search + MEDIA: send works on the target platform
- [ ] The GIF plays correctly in the destination chat

## Agent Workflow (Internal)

1. **Setup (first time):**
   - Call `send_message(action='list')` to see available platforms and targets
   - Parse the output — extract platform names (e.g. `telegram`, `discord`)
   - Run `giphy_multi.py --discover --platforms telegram,discord`
   - Confirm with user if desired

2. **Sending (each time):**
   - Determine current platform from session context
   - Check mode: `natural` (send spontaneously) or `on_request` (wait for explicit ask)
   - **Do not hardcode targets** — call `send_message(action='list')` to discover available targets if unsure
   - Pick a query based on the reaction needed
   - Run the script: `giphy_multi.py "<query>" --channel <platform>`
   - Send via Hermes: `send_message(message="text MEDIA:<path>", target="<target from list>")`

3. **Mode changes:**
   - User says "stop sending without asking" → `giphy_multi.py --mode on_request`
   - User says "feel free to send GIFs naturally" → `giphy_multi.py --mode natural`
