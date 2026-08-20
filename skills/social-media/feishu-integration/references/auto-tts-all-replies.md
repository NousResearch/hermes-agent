# Auto-TTS for All Text Replies

## Problem

By default, Hermes Gateway only auto-TTS when the user sends a **voice message** (MessageType.VOICE). For text message replies, the LLM does NOT automatically call the TTS tool — it just returns plain text.

## Solution: Enable auto-TTS on ALL text replies

### Step 1: Set `voice.auto_tts: true` in config

```yaml
# ~/.hermes/profiles/<profile>/config.yaml
voice:
  auto_tts: true
```

This alone is NOT sufficient — it only enables the auto-TTS *trigger*, but the code still checks for voice messages.

### Step 2: Modify base.py to remove MessageType.VOICE restriction

In `gateway/platforms/base.py`, find the auto-TTS block (around line 2776-2795):

```python
# BEFORE: only triggers on voice messages
if (self._should_auto_tts_for_chat(event.source.chat_id)
        and event.message_type == MessageType.VOICE
        and text_content
        and not media_files):
    ...

# AFTER: also triggers for all text replies when auto_tts is enabled
if (_should_auto_tts_for_chat := self._should_auto_tts_for_chat(event.source.chat_id)) \
        or self._auto_tts_default:
    if text_content and not media_files:
        ...
```

> **⚠️ Pitfall — PlatformConfig is NOT a dict:** `self.config` is a `PlatformConfig` dataclass, NOT a Python dict. You CANNOT use `self.config.get('voice', {})`. The correct way to access the global auto_tts setting is via `self._auto_tts_default`, which GatewayRunner pushes from config.yaml on connect. Using `self.config.get()` will raise `AttributeError: 'PlatformConfig' object has no attribute 'get'`.

### Step 3: Restart Gateway

```bash
hermes gateway restart --profile <profile>
```

## Verification

1. Send a text message on the platform
2. Bot should reply with: (a) voice bubble first, then (b) text content
3. Check logs for TTS generation entries

## Caveats

- **Media files bypass TTS**: If the LLM returns media_files (images, videos), auto-TTS is skipped to avoid redundant output.
- **TTS failures are silent**: If TTS fails, only a warning log is emitted; the text reply is still sent.
- **Config per-profile**: Each profile has its own config.yaml. Make sure you edit the right one.
