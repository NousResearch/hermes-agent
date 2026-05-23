---
name: gif-multi
description: >-
  Search and send reaction GIFs from Giphy on Telegram, Discord, WhatsApp,
  Signal, Slack, and more. Auto-configures per-platform conversion (MP4,
  GIF optimized, text-only fallback).
version: 1.1.0
author: chdlc
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [python3, ffmpeg, curl]
  env_vars: [GIPHY_API_KEY]
metadata:
  hermes:
    tags: [GIF, media, Giphy, multi-platform, reaction]
    related_skills: []
---

# GIF Multi — Cross-platform GIF skill

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
python3 ~/.hermes/skills/media/gif-multi/scripts/gif_multi.py --check
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
python3 ~/.hermes/skills/media/gif-multi/scripts/gif_multi.py \
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
  telegram:user (dm)                         ← DM root (no topic)
  telegram:user / General (dm)               ← DM with a topic
  telegram:user / topic 12345 (dm)           ← Auto-created topic by Telegram
  discord:#general                           ← Server channel
  ...
```

The agent selects the target that matches where the conversation is happening.

- **No topics configured:** only `telegram:user (dm)` appears.
- **Topics active:** each topic appears as `telegram:user / <topic-name> (dm)`.
- **Bare platform** (e.g. `telegram`) sends to the home channel, not the current topic.

### 2. Search and convert

```bash
python3 ~/.hermes/skills/media/gif-multi/scripts/gif_multi.py \
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
  "path": "/home/…/.gif_cache/gif_telegram_12345.mp4"
}
```

### 3. Investigate the platform

Before choosing how to send, check how the target platform handles animated content:

- **Telegram:** Use `sendAnimation` (Bot API) — sends MP4 as an auto-looping GIF. The `sendVideo` method displays it as a regular video with controls. Do NOT use Hermes MEDIA delivery for GIFs on Telegram; use the Bot API `sendAnimation` workaround instead (see below).
- **Discord:** `send_video` with attachments works as a GIF if the file is MP4/WebM.
- **WhatsApp / Signal / others:** Check the platform's documentation or the Hermes platform adapter in `gateway/platforms/` to find the correct sending method.

Do not assume `MEDIA:` delivery behaves identically across platforms. When in doubt, inspect the adapter code or this skill's reference files.

### 4. Send by platform

**Telegram — Bot API directly (recommended for GIFs):**

```bash
BOT_TOKEN=$(grep TELEGRAM_BOT_TOKEN ~/.hermes/.env | grep -v "^#" | cut -d= -f2-)
curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendAnimation" \
  -F chat_id=<CHAT_ID> \
  -F message_thread_id=<THREAD_ID> \
  -F animation=@/path/to/gif.mp4 \
  -F caption="optional caption"
```

See the `references/telegram-sendAnimation-workaround.md` reference file for full details.

**Other platforms — MEDIA delivery as fallback:**

Use `MEDIA:<path>` in `send_message` as a generic fallback. Text and media arrive as separate messages on most platforms.

```python
send_message(
  target='platform:target',
  message="Caption 🐱 MEDIA:/path/to/file.mp4"
)
```

**Do not hardcode chat_id/thread_id.** Topics are optional per-platform and per-user.

> **Why separate messages?** The gateway's `_deliver_media_from_response()` extracts media paths and sends them via `send_video()` without a caption, even though platform adapters support captions.

### 5. Cleanup

The cache under `~/.hermes/.gif_cache/` auto-purges files older than 10 minutes on each search.

## Usage Modes

The config's `"mode"` field controls when GIFs are sent:

- **`natural`** (default) — spontaneous, like emoji reactions.
- **`on_request`** — only when the user explicitly asks ("send a gif of…", "reaction gif").

Change mode:

```bash
python3 ~/.hermes/skills/media/gif-multi/scripts/gif_multi.py --mode on_request
```

The user can also say it in conversation:
- "stop sending without asking" → switch to `on_request`
- "feel free to send GIFs naturally" → switch to `natural`

## Platform Profiles

| Platform | Format | Max size | Notes |
|---|---|---|---|
| Telegram | MP4 (H.264) | 50 MB | Send via sendAnimation, not MEDIA |
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
python3 ~/.hermes/skills/media/gif-multi/scripts/gif_multi.py \
  "funny fail" --channel telegram --rating pg-13
```

## Common Pitfalls

1. **GIPHY_API_KEY not set.** The only symptom is the script printing help text in JSON `"error"` field. Run `--check` first.

2. **ffmpeg not installed.** Required for conversion. Install with `sudo apt install ffmpeg` or equivalent.

3. **Omitting `--channel` when multiple platforms are configured.** The script will error. Always pass `--channel` from the current session context.

4. **Sending to the wrong topic on Telegram.** The `send_message` target must include the thread_id for topic chats. Check `send_message(action='list')` for available targets.

5. **Cache files accumulating.** Auto-purge runs on every search (files >10 min removed). Run `rm -rf ~/.hermes/.gif_cache/` to force-clean.

6. **API rate limit (Giphy: 1,000/day).** If hit, the API returns an error. Wait until the next day or upgrade to a paid plan.

7. **Forgot to send after generating.** Running `gif_multi.py` saves the file to cache — it does NOT deliver it. You must call the sending method (Bot API or `send_message`) after generating. The script only creates the file.

8. **MEDIA needs text alongside it (when used as fallback).** `send_message(message="MEDIA:/path/to/file.mp4")` fails with "No deliverable text or media remained". Include at least some text: `message="🎉 MEDIA:/path/to/file.mp4"`.

9. **MEDIA on Telegram sends as video, not GIF.** Hermes' MEDIA delivery uses `send_video()` which displays the MP4 as a regular video with controls. For proper auto-looping GIFs on Telegram, use the Bot API `sendAnimation` workaround (see step 4 and the reference file).

10. **Assuming MEDIA works the same across platforms.** Always investigate how the target platform handles animated content before sending. What works on Discord may not work on Telegram, Signal, or others.

## Reference files

- `references/telegram-sendAnimation-workaround.md` — Bot API curl command for proper GIF delivery on Telegram
- `references/hermes-media-delivery.md` — how MEDIA: flows through the gateway

## Verification Checklist

- [ ] `GIPHY_API_KEY` is set in `~/.hermes/.env`
- [ ] `python3 gif_multi.py --check` shows all ✅
- [ ] `--discover --platforms` configured the skill for your platforms
- [ ] A test search + send works on the target platform
- [ ] The GIF plays correctly as an animation (not a video/sticker) in the destination chat

## Agent Workflow (Internal)

1. **Setup (first time):**
   - Call `send_message(action='list')` to see available platforms and targets
   - Parse the output — extract platform names (e.g. `telegram`, `discord`)
   - Run `gif_multi.py --discover --platforms telegram,discord`
   - Confirm with user if desired

2. **Sending (each time):**
   - Determine current platform from session context
   - Check mode: `natural` (send spontaneously) or `on_request` (wait for explicit ask)
   - **Do not hardcode targets** — call `send_message(action='list')` to discover available targets if unsure
   - Pick a query based on the reaction needed
   - Run the script: `gif_multi.py "<query>" --channel <platform>`
   - **Investigate the platform first** — check how it handles animated content. See step 3 in Daily Workflow.
   - **Telegram:** use Bot API `sendAnimation` via curl (see reference file). Do NOT use MEDIA for Telegram GIFs.
   - **Other platforms:** use `send_message` with `MEDIA:<path>` as generic fallback. Text and media arrive as separate messages.
   - **Critical:** ensure you actually call the sending method after generating — the script only creates the file, it doesn't deliver it.

3. **Mode changes:**
   - User says "stop sending without asking" → `gif_multi.py --mode on_request`
   - User says "feel free to send GIFs naturally" → `gif_multi.py --mode natural`
