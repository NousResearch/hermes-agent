# Feishu Voice Bubble Investigation

## Session: 2026-05-07 (Honey)

### Finding

Feishu's Bot API **does not support true voice bubbles** (like WeChat's round voice button with time indicator). 

The current implementation sends `msg_type="audio"` which renders as a **playable audio player** in the chat. This is the closest Feishu API provides for bot-sent audio.

### Code Path

1. TTS tool (`tools/tts_tool.py` line ~1588): `want_opus = platform in ("telegram", "feishu")`
2. Output: `.ogg` (opus) format when want_opus=True and provider supports native opus
3. Feishu send_voice(): calls `_send_uploaded_file_message()` with `outbound_message_type="audio"`
4. `_resolve_outbound_file_routing()`: returns `("opus", "audio")` for `.ogg`/`.opus` files
5. API call: `im.v1.message.create` with `msg_type="audio"`, payload `{"file_key": "..."}`

### Why Not a Voice Bubble?

Feishu's Bot API message types are limited to: text, share_chat, image, file, media, audio, post, interactive, sticker, system. The `audio` type is a file-based player, not a voice bubble. True voice bubbles require the user to send via the Feishu client (not bot API).

### Workaround Options

- Accept the audio player as-is (current behavior)
- Use Feishu's custom message type if available in newer API versions
- Client-side: not feasible for bot-sent messages
