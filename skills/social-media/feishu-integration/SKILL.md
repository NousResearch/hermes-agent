---
name: feishu-integration
description: Enable Feishu/Lark voice bubbles, TTS routing, and platform-specific configuration for Hermes Agent.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [feishu, lark, voice-bubble, tts, opus, audio]
    related_skills: []
---

# Feishu (Lark) Integration

Platform-specific integration guide for Hermes Agent on the Feishu/Lark messaging platform. Covers voice bubble support, TTS routing, auto-TTS configuration, and common pitfalls.

## Overview

Feishu's Bot API supports sending audio messages via `msg_type="audio"` with opus-encoded `.ogg` files. While it does **not** support true voice bubbles (like WeChat's round voice button), the audio player renders as a playable media element in chat.

This skill documents:
- Enabling opus output for Feishu TTS
- Configuring auto-TTS on all text replies
- Gateway restart procedures specific to Feishu
- Debugging TTS mismatches and silent failures

## When to Use

- You need voice/TTS output on Feishu/Lark
- Audio messages are playing as files instead of audio players
- TTS produces no audio or wrong language/voice
- Auto-TTS is not firing on text replies
- Gateway restarts cause Feishu lock contention errors

## Voice Bubble Setup (opus format)

### Step 1: Enable opus output in TTS tool

In `tools/tts_tool.py` line ~1588, ensure `"feishu"` is in the platform tuple:

```python
want_opus = platform in ("telegram", "feishu")
```

This ensures TTS providers output `.ogg` (opus) format.

### Step 2: Verify routing

The `_resolve_outbound_file_routing()` method in `gateway/platforms/feishu.py` line ~4438 returns `("opus", "audio")` for `.ogg` files, which sends as `msg_type="audio"` via the Feishu Bot API.

### Step 3: Configure gateway

Add `feishu` to your Gateway's `platforms` list in `config.yaml`. The default only includes `api_server`.

## Platform Comparison

| Platform | Opus Supported | Message Type |
|----------|---------------|--------------|
| Telegram | Yes | voice_file |
| Feishu/Lark | Yes | audio (opus) |
| Discord | Partial* | voice_file (ogg/opus) |

*\*Discord uses a different message type for ogg/opus.*

## TTS Provider Notes

- **Native opus** — OpenAI, ElevenLabs, Mistral, Gemini: output `.ogg` when `want_opus=True`
- **Requires ffmpeg transcoding** — Edge TTS, NeuTTS, Minimax, XAI, KittenTTS, Piper: output `.mp3`, auto-transcoded to opus via ffmpeg (lines ~1759-1762 in `hermes_gateway.py`)
- **Command provider**: needs `voice_compatible: true` to trigger transcoding

## Auto-TTS on All Text Replies

The gateway has **two** independent auto-TTS code paths:

1. `gateway/run.py` — `_send_voice_reply()` triggered after agent response via `_should_send_voice_reply()`
2. `gateway/platforms/base.py` — auto-TTS block triggered on inbound messages

To make ALL text replies become voice bubbles:
1. Set `voice.auto_tts: true` in config.yaml
2. Remove the `MessageType.VOICE` restriction in the auto-TTS block of `base.py`

See `references/auto-tts-all-replies.md` for full procedure.

## Adding a New Platform for Voice Bubbles

When adding support for a new platform that supports voice bubbles:

1. Add the platform name to the `want_opus` tuple in `tools/tts_tool.py` (~line 1588)
2. Verify `_resolve_outbound_file_routing()` returns the correct `(format, message_type)` tuple
3. Ensure the platform's file extension mapping recognizes the TTS output format (`.ogg` for opus, `.mp3` for mp3)

## Common Pitfalls

1. **Gateway restart required after code changes**: After modifying `tools/tts_tool.py` or any platform adapter, restart: `launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway.plist && sleep 2 && launchctl load ~/Library/LaunchAgents/ai.hermes.gateway.plist`. Verify with `hermes gateway status` that PID changed.

2. **ffmpeg required**: Edge TTS and other mp3-output providers rely on ffmpeg for transcoding to opus. Without it, Feishu receives unplayable `.mp3` files.

3. **Command provider no auto-transcode**: By default does NOT transcode. Must explicitly set `voice_compatible: true` in config.

4. **Feishu websocket lock contention on restart**: Before restarting the Gateway after any feishu-related changes, kill ALL hermes gateway processes first (including those from other profiles). An old process holds the feishu `app_id` lock and causes: `Another local Hermes gateway is already using this Feishu app_id`. Fix: `kill -9 <all-hermes-pids>` then unload/load launchd plist. See `references/feishu-lock-contention.md`.

5. **PlatformConfig is NOT a dict**: `self.config` in platform adapters is a `PlatformConfig` dataclass — NOT a Python dict. Use pre-pushed config fields like `self._auto_tts_default` instead of `self.config.get('key')`. See `references/auto-tts-all-replies.md`.

6. **TTS text mismatch with AI reply**: Check `extract_images()` in `base.py` — it may strip image markdown including alt-text from content passed to TTS. Also check `_strip_markdown_for_tts()` in `tts_tool.py`. See `references/tts-text-mismatch.md`.

7. **Edge TTS voice mismatch silent failure**: When using Edge TTS with a non-native voice (e.g., `en-US-AriaNeural` for Chinese text), returns `NoAudioReceived` silently — no audio, no error. For Chinese+English mix, use `zh-CN-XiaoxiaoNeural`. See `references/tts-voice-language-mismatch.md`.

8. **Auto-TTS strips emojis**: Emoji characters and `:smile:`-style alt-text get synthesized as speech. Strip them before calling TTS. See `references/tts-emoji-stripping.md`.

## Voice Bubble Limitation

Feishu's Bot API message types are limited to: text, share_chat, image, file, media, audio, post, interactive, sticker, system. The `audio` type is a file-based player, not a voice bubble. True voice bubbles require the user to send via the Feishu client (not bot API).

See `references/voice-bubble-limitation.md` for investigation details.

## References

- `references/auto-tts-all-replies.md` — configuring auto-TTS on all text replies
- `references/feishu-lock-contention.md` — gateway restart procedure with feishu lock handling
- `references/tts-text-mismatch.md` — debugging TTS output that doesn't match AI reply
- `references/tts-voice-language-mismatch.md` — Edge TTS voice/language compatibility guide
- `references/tts-debugging-guide.md` — step-by-step TTS troubleshooting flow
- `references/tts-emoji-stripping.md` — regex patterns for emoji removal before TTS
- `references/voice-bubble-limitation.md` — Feishu API limitation investigation
