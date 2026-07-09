# Generic Streaming TTS — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Refactor the ElevenLabs-hardcoded `stream_tts_to_speaker` pipeline into a provider-agnostic dispatcher, then add real streaming support for every TTS provider whose API exposes chunked output (Gemini, OpenAI, xAI, edge, with Piper staying sync).

**Architecture:** Introduce a `StreamingTTSProvider` ABC that each provider implements to yield PCM chunks. The existing sentence-buffer + sounddevice scaffolding moves to a generic dispatcher that picks the right provider at runtime. ElevenLabs is the reference port; all other providers are added behind the same interface.

**Tech Stack:** `httpx` (SSE for Gemini), `openai` SDK (streaming for OpenAI + xAI REST), `websockets` (xAI TTS WebSocket fallback), `edge-tts`'s `Communicate` (already a dependency), `sounddevice` (existing), `numpy` (existing).

---

## Streaming-Capability Matrix (verified)

| Provider | Streaming? | Mechanism | Status |
|---|---|---|---|
| ElevenLabs | ✅ | chunked HTTP, `text_to_speech.convert()` iterator | **Reference impl** — port to ABC |
| Gemini | ✅ | `streamGenerateContent` SSE → L16 PCM chunks at 24kHz | Add |
| OpenAI | ✅ | `gpt-4o-mini-tts` with `stream=True` (PCM/mp3/opus) | Add |
| xAI TTS | ✅ | WebSocket-only (`wss://api.x.ai/v1/tts`) | Add (harder) |
| edge-tts | ✅ | `edge_tts.Communicate.stream()` yields bytes | Add |
| Piper (local) | ❌ | VITS produces whole utterances; can't stream inside a sentence | **Stay sync** — works in pipeline as sentence consumer |
| NeuTTS, KittenTTS, Mistral Voxtral | ❌ | Sync REST only | **Stay sync** |

---

## Files to Create

1. `tools/tts_streaming.py` (~400 lines) — `StreamingTTSProvider` ABC + dispatcher + all 5 implementations
2. `tests/tools/test_tts_streaming_providers.py` (~250 lines) — one mocked test per provider verifying chunk iteration, format, and stop handling

## Files to Modify

1. `tools/tts_tool.py`:
   - Replace the ElevenLabs-specific body of `stream_tts_to_speaker` (lines 2475-2600) with a call to the new dispatcher
   - Keep the public function signature, threading, queue protocol, and `display_callback` semantics intact
   - Add a config-driven `tts.streaming.provider` default (falls back to the active sync provider)
2. `agent/voice_mode.py` — no changes expected; relies on `stream_tts_to_speaker` public contract
3. `docs/streaming-tts.md` — short design doc + capability matrix (used as the reference for tests + future providers)

---

## Design Notes

### ABC contract

```python
class StreamingTTSProvider(ABC):
    sample_rate: int
    channels: int
    sample_width: int   # bytes per sample (2 for int16)

    @abstractmethod
    def stream(self, text: str) -> Iterator[bytes]:
        """Yield raw PCM chunks. Caller handles device/playback."""
```

The dispatcher opens the `sounddevice.OutputStream` **once** (configurable from any provider's spec — they all use 24kHz/mono/int16 in practice) and writes each chunk via `output_stream.write(np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1))`. Stop event is checked between chunks.

### Provider selection

- Config key: `tts.streaming.provider` (string, default = `tts.provider`)
- If the configured provider has no streaming implementation → log a warning, fall back to the first available streaming provider in priority order: `elevenlabs → gemini → openai → xai → edge`
- This is the safety net so enabling "voice mode" doesn't break for users with a non-streaming default provider

### Audio tag handling

- ElevenLabs: no audio tags (provider doesn't support)
- Gemini: aux-model rewrite step (existing) happens **before** the streaming call returns to the dispatcher
- OpenAI: accepts `instructions` parameter for tone; no inline tags
- xAI: Speech Tags (`[laugh]`, `[whisper]`, etc.) pass through as plain text
- edge: SSML support, but we don't pre-process — keep simple

### Per-provider PCM format

All providers can be coerced to **L16 mono 24kHz**:
- ElevenLabs: ask for `pcm_24000` (existing)
- Gemini: hardcoded by the API (L16 24kHz)
- OpenAI: ask for `pcm` format with sample rate param
- xAI: server returns PCM; verify on first impl, possibly resample
- edge: `edge_tts.Communicate` yields mp3 chunks → must decode via ffmpeg/pydub before passing to sounddevice

Edge's mp3-in / pcm-out is the only implementation that's a real deviation. For MVP, the edge streaming impl can write to a temp WAV and play that (matching the existing `_play_via_tempfile` fallback). A future task can swap in real-time pcm decode.

---

## Task Structure (bite-sized, 2-5 min each)

### Task 1: Create `StreamingTTSProvider` ABC
- New file: `tools/tts_streaming.py`
- Define class with `sample_rate`, `channels`, `sample_width`, abstract `stream()`
- Add module-level `_PROVIDERS` registry dict and `register(name)` / `get(name)` helpers

### Task 2: Test ABC contract with a fake provider
- New file: `tests/tools/test_tts_streaming_providers.py`
- `test_abc_requires_stream_method` — can't instantiate without `stream()`
- `test_abc_requires_audio_format_attrs` — concrete subclass missing attrs raises
- `test_get_returns_registered_provider` — register + get round-trip
- `test_get_raises_for_unknown` — unknown name → `KeyError` (or custom exception)

### Task 3: Implement ElevenLabs streaming provider
- Port the existing `client.text_to_speech.convert(...)` call from `tts_tool.py:2553-2566` into a class
- `sample_rate=24000`, `channels=1`, `sample_width=2`
- `stream()` yields bytes; respects a `stop_event` passed in via a small wrapper to avoid coupling the ABC to threading
- Test: mock ElevenLabs client, assert two chunks yielded, audio format correct

### Task 4: Implement Gemini streaming provider
- Use `httpx` (already a dep) to POST to `streamGenerateContent` endpoint
- Parse SSE: each event is JSON; the audio lives at `candidates[0].content.parts[0].inlineData.data` (base64-encoded L16 PCM)
- Yield decoded bytes; abort cleanly on stop event
- Test: mock `httpx.post` to return canned SSE; assert bytes match decoded base64 chunks

### Task 5: Implement OpenAI streaming provider
- Use the existing `openai` SDK with `stream=True`
- Request format `pcm`; sample rate passed via `extra_body` (verify on impl)
- Test: mock the OpenAI client, assert iterator chunks and audio format

### Task 6: Implement xAI streaming provider
- WebSocket client via `websockets` (check if it's already a dep; add to pyproject if not)
- Per-sentence: open WS, send text+voice config, receive PCM frames, yield
- Test: mock `websockets.connect`, assert handshake + text send + frame receive + close

### Task 7: Implement edge streaming provider (MVP: tempfile fallback)
- Wrap the existing `_play_via_tempfile` logic as a streaming provider
- Yields the entire mp3 as a single "chunk" via the tempfile path
- Test: mock `edge_tts.Communicate`, assert a WAV is written and a player is invoked

### Task 8: Implement `stream_tts_to_speaker` dispatcher
- Keep the **public signature** of `stream_tts_to_speaker(text_queue, stop_event, tts_done_event, display_callback)` unchanged
- Resolve the active streaming provider from config (with priority fallback)
- Open `sounddevice.OutputStream` using the provider's sample rate/channels/width
- Per sentence: call `provider.stream(sentence)`, write each chunk to the output stream, check `stop_event` between chunks
- `display_callback` fires before TTS, just like today
- Reuse the existing think-block stripping and sentence-boundary regex verbatim

### Task 9: Add config knob `tts.streaming.provider`
- Defaults to `None` → dispatcher picks the active sync provider, then falls back through the priority list
- Set explicit value via `hermes config set tts.streaming.provider gemini`
- Document in `docs/streaming-tts.md`

### Task 10: Test full dispatcher flow
- Mock a fake streaming provider that yields 3 chunks of synthetic PCM
- Feed a text queue with 2 sentences + a `None` sentinel
- Assert both sentences get spoken, the right number of chunks written, `tts_done_event` set
- Test stop-event: set `stop_event` mid-stream, assert no further writes + event is set

### Task 11: Replace ElevenLabs-specific code in `tts_tool.py`
- Remove the ElevenLabs imports/setup inside `stream_tts_to_speaker`
- Replace the body with a call to the new dispatcher
- Keep the queue/stop_event/tts_done_event protocol identical
- Run existing voice-mode tests to confirm no regression

### Task 12: E2E test against the real ElevenLabs API
- Gated on `ELEVENLABS_API_KEY` env var
- Generate a short sentence, assert non-empty audio bytes returned
- Assert sample rate / channels match the contract

### Task 13: E2E test against the real Gemini API
- Gated on `GEMINI_API_KEY` / `GOOGLE_API_KEY`
- Generate a short Spanish sentence (the user's actual use case), assert non-empty audio

### Task 14: Wire config + run existing TTS test suite
- `pytest tests/tools/test_tts_*` should still pass — the sync path is untouched
- Add a regression test: `test_tts_provider_param_still_works` for the recent `feat/tts-provider-parameter` work

### Task 15: Update `AGENTS.md` streaming-TTS section
- Add a short paragraph under "Tooling principles" describing the ABC + dispatcher pattern
- Note that "any provider that supports streaming gets it for free"

### Task 16: Open PR with checklist
- Title: `feat(tts): generic streaming dispatcher for all chunked providers`
- Description: link to this plan + capability matrix, call out the xAI WebSocket complexity, list the 5 providers now streaming-capable
- Checklist items: tests pass, no sync-path regression, doc updated, E2E gated tests documented

---

## Verification

- `pytest tests/tools/test_tts_streaming_providers.py -v` — all unit tests pass
- `pytest tests/tools/test_tts_*.py -v` — sync path regression-clean
- Manual: enable voice mode, ask "tell me a short joke", confirm audio starts ~1s after first sentence vs the current full-LLM-then-TTS delay
- Manual: switch `tts.streaming.provider` between `elevenlabs` / `gemini` / `openai` / `xai` / `edge` and confirm each plays

## Estimated scope

~600 lines new (streaming module + tests) + ~80 lines removed from `tts_tool.py` + ~50 lines docs = ~700 net. Spread across 16 small tasks, no single commit larger than ~150 lines.

## Out of scope (YAGNI)

- Real-time mp3→PCM decode for edge-tts (tempfile fallback is good enough for MVP)
- Adaptive buffering / jitter handling (sounddevice's natural buffering is sufficient)
- Multi-speaker / SSML — punted to a follow-up
- Streaming for NeuTTS / KittenTTS / Mistral / Piper (none of them support it; document why)
