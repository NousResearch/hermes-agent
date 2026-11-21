# TTS Output Routing Chain

Complete chain of how TTS output flows from provider to platform message delivery.

## 1. TTS Provider Output

Provider determines output format based on `want_opus` flag:

| Provider | want_opus=True | want_opus=False |
|----------|---------------|-----------------|
| OpenAI | .ogg (opus) | .mp3 |
| ElevenLabs | .ogg (opus) | .mp3 |
| Mistral | .ogg (opus) | .mp3 |
| Gemini | .ogg (opus) | .mp3 |
| Edge TTS | .mp3 (always) | .mp3 |
| NeuTTS | .mp3 (always) | .mp3 |
| Minimax | .mp3 (always) | .mp3 |
| XAI | .mp3 (always) | .mp3 |
| KittenTTS | .mp3 (always) | .mp3 |
| Piper | .wav (always) | .wav |

## 2. Transcoding Layer (hermes_gateway.py lines ~1759-1762)

If `want_opus=True` AND output is NOT already opus (.ogg):
- ffmpeg converts to .ogg (opus) automatically
- Applies to: Edge TTS, NeuTTS, Minimax, XAI, KittenTTS, Piper

## 3. File Routing (gateway/platforms/feishu.py _resolve_outbound_file_routing)

```python
def _resolve_outbound_file_routing(self, file_path):
    ext = Path(file_path).suffix.lower()
    if ext in {".ogg", ".opus"}:  # _FEISHU_OPUS_UPLOAD_EXTENSIONS
        return ("opus", "audio")   # voice bubble on Feishu
    elif ext in _FEISHU_MEDIA_UPLOAD_EXTENSIONS:
        return ("mp4", "media")
    ...
```

> **Note:** Routing is now per-platform. Each platform's adapter has its own `_resolve_outbound_file_routing` method with platform-specific extensions and message types.

## 4. Platform Rendering

- Telegram: `voice_file` → voice message bubble
- Feishu: `audio` with opus → playable audio player in chat
- Other platforms: `file` → file attachment
