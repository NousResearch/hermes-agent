# Streaming TTS

Hermes can stream TTS audio as it arrives from the provider, instead of waiting for the full audio before playing. This is used by voice mode (live conversation) and any caller that wants low time-to-first-audio.

## Architecture

The streaming pipeline has three parts:

1. **Producer** — the LLM emits text deltas as it generates a response
2. **Sentence buffer** — accumulates deltas, flushes complete sentences
3. **TTS provider** — turns each sentence into PCM audio chunks
4. **Audio output** — `sounddevice.OutputStream` plays chunks as they arrive

The producer/sentence-buffer/stream-playback scaffolding lives in `agent/voice_mode.py` and `tools/tts_tool.py::stream_tts_to_speaker`. The TTS provider layer lives in `tools/tts_streaming.py` and is provider-agnostic.

## How to pick a provider

Set `tts.streaming.provider` in your `config.yaml` to one of:

- `elevenlabs` — chunked HTTP, PCM 24kHz, requires `ELEVENLABS_API_KEY`
- `gemini` — SSE via `streamGenerateContent`, PCM 24kHz, requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `openai` — chunked HTTP via `with_streaming_response`, PCM 24kHz, requires `OPENAI_API_KEY`
- `xai` — WebSocket, requires `XAI_API_KEY`
- `edge` — Microsoft Edge TTS (free, no key); uses a tempfile MVP, not true real-time streaming

Example:

```yaml
tts:
  provider: gemini
  streaming:
    provider: gemini
  gemini:
    model: gemini-3.1-flash-tts-preview
    voice: Aoede
```

If `tts.streaming.provider` is unset, the dispatcher falls back through the priority list: `elevenlabs → gemini → openai → xai → edge`, picking the first one whose required env vars are set. Edge needs no env var, so it always wins if nothing else is configured.

## Adding a new streaming provider

1. Subclass `StreamingTTSProvider` in `tools/tts_streaming.py`
2. Set `sample_rate`, `channels`, `sample_width` class attrs
3. Implement `stream(self, text: str) -> Iterator[bytes]` to yield raw PCM chunks
4. Decorate with `@register("yourname")` so the dispatcher can find it
5. Add a test in `tests/tools/test_tts_streaming_providers.py`

The ABC enforces the contract; the registry makes the provider discoverable; the dispatcher handles audio device + sentence buffer + stop events.

## Capability matrix

| Provider | Streaming | Mechanism | Latency-to-first-audio |
|---|---|---|---|
| ElevenLabs | ✅ | chunked HTTP | ~300ms |
| Gemini | ✅ | SSE | ~500ms |
| OpenAI | ✅ | chunked HTTP | ~400ms |
| xAI | ✅ | WebSocket | ~300ms |
| Edge | ✅ (MVP) | tempfile accumulate | ~1-2s |
| Piper, NeuTTS, KittenTTS, Mistral | ❌ | sync only | n/a |
