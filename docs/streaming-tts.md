# Streaming TTS

Hermes can stream TTS audio as it arrives from the provider, instead of waiting
for the full audio before playing. This is used by voice mode (CLI/TUI live
conversation), the dashboard speak-stream WebSocket, and — via the gateway
`StreamingTTSConsumer` — any platform adapter that opts into streaming audio.
Voice replies start speaking after the first clause instead of after full
generation + synthesis.

## Architecture

The streaming pipeline has four parts:

1. **Producer** — the LLM emits text deltas as it generates a response
2. **Sentence chunker** — `tools.tts_streaming.SentenceChunker` accumulates
   deltas, strips `<think>` blocks (even split across deltas), and flushes
   complete sentences
3. **TTS provider** — a registered `StreamingTTSProvider` turns each sentence
   into raw PCM chunks (int16 mono at the provider's declared `sample_rate`)
4. **Audio sink** — `sounddevice.OutputStream` for local playback
   (`tools.tts_tool_speaker.stream_tts_to_speaker`), or a gateway platform adapter's
   `write_streaming_tts` seam (`gateway/streaming_tts_consumer.py`)

Providers with no chunked API still get per-*sentence* playback via the proven
sync `text_to_speech_tool` path, so edge (the default) is conversational too.
All spoken text is cleaned by `tools.tts_text_normalize.prepare_spoken_text`
(one cleaner, all paths).

## How to pick a provider

By default the dispatcher streams with the provider you already configured
(`tts.provider`) when that provider has a chunked API — it never silently
swaps your voice for a different provider just to get streaming.

To override, set `tts.streaming.provider` in your `config.yaml`:

- a provider name (`elevenlabs`, `gemini`, `openai`, `xai`, `voicevox`) pins that streamer
- `auto` walks the priority list `elevenlabs → gemini → openai → xai` and uses
  the first one whose credentials resolve — an explicit opt-in to "best
  chunked voice available"

```yaml
tts:
  provider: gemini
  streaming:
    provider: gemini      # or "auto"
  gemini:
    model: gemini-2.5-flash-preview-tts
    voice: Kore
```

## Capability matrix

| Provider    | Transport                             | Chunked PCM | Credentials |
|-------------|---------------------------------------|-------------|-------------|
| elevenlabs  | chunked HTTP (`pcm_24000`)            | yes         | `ELEVENLABS_API_KEY` / `tts.elevenlabs` |
| openai      | chunked HTTP (`with_streaming_response`, `pcm`) | yes | `tts.openai.api_key` → env → managed gateway |
| gemini      | SSE (`streamGenerateContent?alt=sse`) | yes         | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| xai         | WebSocket (`wss://api.x.ai/v1/tts`)   | yes         | xAI OAuth or `XAI_API_KEY` |
| voicevox    | HTTP `/audio_query` + `/synthesis` | per-sentence WAV → PCM | running VOICEVOX Engine |
| edge, piper, kitten, neutts, mistral, minimax, deepinfra, … | — | no (per-sentence sync fallback) | as usual |

All credential lookups go through `resolve_provider_secret()`
(config > env/.env > credential pool) — never bare env reads. Streamed bodies
are capped at 16 MiB per sentence, mirroring the sync providers' bounded
upstream-body invariant.

## Japanese speech with VOICEVOX

Start [VOICEVOX Engine](https://github.com/VOICEVOX/voicevox_engine) separately,
then merge these keys into the active Hermes profile's `config.yaml`:

```yaml
tts:
  # Keep your existing whole-file provider for fallback and voice attachments.
  streaming:
    provider: voicevox
  voicevox:
    base_url: http://127.0.0.1:50021
    speaker: 3  # ずんだもん・ノーマル; check the engine's /speakers endpoint
```

The streaming override affects streaming speech only. If it fails before any
PCM arrives, Desktop uses the existing `tts.provider` for whole-text playback;
that provider may have a different voice. This PR does not add VOICEVOX to
whole-file attachment generation. Existing `tts.providers.voicevox` command
setups can continue using `tts.provider: voicevox`; their `voice`, `host` and
`port` settings remain supported, with `tts.voicevox` taking precedence.

Enable **Read replies aloud** in Desktop, or start a voice conversation.
New text is sent while the assistant is writing; `うん。` is enough to begin
synthesis. Opening a conversation does not read its existing history. Stop
abandons the remaining speech, including a pending fallback. Each session's
cancel operation affects only that session, so late completion cannot stop the
next reply. Voice selection remains per profile.

The engine returns a complete WAV for each sentence. The integration pipelines
sentence synthesis and playback while the LLM continues; it does not stream
partial acoustic generation inside VOICEVOX. It requests and verifies 24 kHz
mono int16 PCM before playback. Long sentences retain their tail rather than
being silently truncated. Stop closes the WebSocket promptly and drops later
results; an already running engine HTTP request may finish in the background.

Test with a short acknowledgement and a longer reply, then stop during speech
and submit again. Server logs report `first_text_ms` and `first_pcm_ms` relative
to WebSocket startup, plus sentence/byte counts, without recording spoken text.
These are transport measurements, not user-to-audible-speech latency. No fixed
speed or improvement over another PR is guaranteed.

For published examples using speaker 3, credit **VOICEVOX:ずんだもん** and follow
the engine and character terms. Engine and voice assets are not bundled.

## Adding a new streaming provider

1. Subclass `StreamingTTSProvider` in `tools/tts_streaming.py`
2. Set `sample_rate` (and `channels` / `sample_width` if not int16 mono)
3. Implement `available()` (a pure probe — never install anything) and
   `stream(self, text) -> Iterator[bytes]` yielding raw PCM chunks
4. Decorate with `@register("yourname")`
5. Add tests in `tests/tools/test_tts_streaming.py`

The ABC enforces the contract; the registry makes the provider discoverable;
the dispatcher (`stream_tts_to_speaker`) and the gateway consumer handle the
sentence buffer, stop events, and audio sink for free.

## Gateway streaming (platform adapters)

`gateway/streaming_tts_consumer.py` bridges agent deltas to an adapter's
streaming-audio seam. Adapters opt in by overriding, on
`BasePlatformAdapter`:

- `supports_streaming_tts(chat_id, audio_format) -> bool`
- `begin_streaming_tts / write_streaming_tts / finish_streaming_tts /
  abort_streaming_tts`

All default to unsupported/no-op, so existing adapters are untouched. When a
turn's streaming audio completes, the whole-file auto-TTS reply for that turn
is suppressed (no double playback); when streaming fails before any audio was
audible, the gateway falls back to the legacy whole-file voice reply.
