# SimpleX Streaming Voice — Slice 4: Real Streaming STT (Local) (Design)

> Status: DESIGN. Architecture **A — ports own the loop**. Delivers a real `SpeechToTextPort` backed by **local faster-whisper** (segment STT, transcribe-on-finalize). Cloud Deepgram Flux (true streaming) is a **later** slice. `FakeSTT` stays the default; the real adapter is opt-in. Cloud is never the only path; the default path here is fully local.

## 1. Goal

`LocalWhisperSTT`, a real `SpeechToTextPort` that buffers pushed audio frames and, on `finalize()` (called by the session at the turn endpoint), transcribes the buffered turn with local faster-whisper and returns one `FINAL` `TranscriptEvent`. No cloud, no API key. Delivered behind the port with mocked unit tests (run in CI) + a real-ASR contract test (runs locally, skips where faster-whisper is absent). No session/engine wiring beyond an opt-in selector; `FakeSTT` remains the default.

## 2. Validated facts & constraints

- `SpeechToTextPort` (ports.py): `async start(ctx)`, `async push(frame)`, `events() -> AsyncIterator[TranscriptEvent]`, `async finalize() -> TranscriptEvent | None`, `async cancel()`, `async close()`.
- `TranscriptEvent(call_id, kind, text, start_ms=0, end_ms=0, stability=1.0, words=(), provider="")`; `TranscriptKind`: PARTIAL, FINAL.
- `faster-whisper==1.2.1` lives in the `voice` extra (heavy: ctranslate2 + onnxruntime), NOT installed in the dev venv nor in CI's `.[all,dev,simplex-streaming]`. It is **segment/batch** STT (no native streaming partials). `tools/transcription_tools.py` already constructs `WhisperModel(model, device=..., compute_type=...)` and calls `model.transcribe(...)`.
- ast-grep `no-walltime` bans `time.*`/`asyncio.sleep` in `streaming/**` (except `clock.py`) — the adapter buffers bytes and runs once on `finalize()`, so it needs no wall-clock. `start_ms/end_ms` derive from frame timestamps.
- Slice-3 committed fixture reused: `tests/gateway/streaming/fixtures/turn_detection/speech_16k_mono.wav` (16k mono, phrase "Hello Hermes, what is the weather forecast for today?").

## 3. Decisions

- **D1 — Backend & scope:** a **local faster-whisper** adapter is the Slice-4 deliverable; **defer Deepgram Flux** (cloud, true-streaming) to a later slice. This is the only path giving real ASR coverage without a cloud key, and it makes the default streaming STT fully local (cloud never the only path).
- **D2 — Streaming semantics:** **buffer-on-push, transcribe-on-`finalize()`, emit one `FINAL`, no `PARTIAL`s.** `push(frame)` appends `frame.pcm16` to an in-memory buffer (tracking first/last frame `timestamp_ms` for `start_ms`/`end_ms`). `finalize()` converts the buffered int16 PCM to a float32 numpy array in `[-1, 1]` (`np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0`) and passes it **directly** to `WhisperModel.transcribe(audio_array, language="en")` (verified: faster-whisper accepts an in-memory numpy array — **no temp WAV**). numpy is available without the `voice` extra (it rides in via `pipecat-ai` / the STT stack), so `import numpy` is safe. Returns one `FINAL` `TranscriptEvent(provider="faster-whisper")`; `None` if no audio buffered. `events()` returns an empty async iterator (no partials this slice — verified safe: `session.py` reads the FINAL from the `finalize()` return value, and consumes `events()` only to update `_latest_partial`, which tolerates `None`). `cancel()`/`close()` clear the buffer and release the model reference.
- **D3 — Model loading & gating:** lazy-import `faster_whisper` inside a factory `build_local_whisper_stt(*, model="distil-small.en", device="cpu", compute_type="int8")`; raise a clear `RuntimeError` naming the extra when faster-whisper is absent. The adapter accepts an **injected** `transcribe` callable / model object so unit tests mock it (no real model in CI). Keep the streaming package importable without the extra (no top-level `import faster_whisper`).
- **D4 — Packaging:** new extra `simplex-streaming-local-stt = ["faster-whisper==1.2.1"]` (reuse the existing pin; not folded into `simplex-streaming` to keep pipecat-only installs lean; not `voice` to avoid dragging the `sounddevice` audio-capture dep — numpy is unaffected, it arrives via pipecat). **Not** added to CI install (avoids ~150MB model download + ctranslate2 weight + HF flakiness on the private fork). **No config flag this slice** — a `stt.backend` selector has no consumer until the session is wired to choose a real STT (a later slice); adding it now would be an orphaned, untested key (YAGNI).
- **D5 — Test strategy:** (mocked, run in CI) inject a fake transcriber to test buffering, finalize→FINAL, empty-buffer→None, events()-empty, cancel/close clearing, and the absent-extra factory error. (real ASR, local-only) a contract test `skipif(not _faster_whisper_available())` that loads the committed WAV, pushes it as 20ms frames, calls `finalize()`, and asserts **normalized keyword containment** — `"weather"` and `"today"` and (`"hermes"` or `"forecast"`) — never an exact string. Not `@integration`.
- **D6 — Scope:** deliver adapter + factory + tests only. No engine/session wiring beyond the (optional) config flag. `FakeSTT` stays default.

## 4. BDD scenarios

1. **Buffer + finalize → one FINAL** (mocked transcriber): given frames pushed then `finalize()`, the injected transcriber receives the concatenated PCM and the result is one `FINAL` `TranscriptEvent` with `provider="faster-whisper"`, `text` == the mocked transcript, `start_ms`/`end_ms` from first/last frame.
2. **Empty buffer → None**: `finalize()` with nothing pushed returns `None`.
3. **events() is empty**: iterating `events()` yields nothing (no partials this slice).
4. **cancel()/close() clear state**: after `push`+`cancel()`, the buffer is empty and a subsequent `finalize()` returns `None`; `close()` releases the model.
5. **Absent extra → clear error**: `build_local_whisper_stt()` with faster-whisper unavailable raises `RuntimeError` naming `simplex-streaming-local-stt`.
6. **Real-ASR contract** (skipif faster-whisper absent; local): pushing the committed WAV as 20ms frames then `finalize()` yields a `FINAL` whose normalized text contains `"weather"` + `"today"` + (`"hermes"` | `"forecast"`).
7. **16k-mono guard** (mirrors Slice 3): `push()` validates the frame's own `frame.media` — a non-16k or non-mono frame raises a clear `ValueError` naming the 16k-mono requirement.

## 5. Files

- **Create:** `gateway/calls/native/streaming/local_whisper_stt.py` — `LocalWhisperSTT(SpeechToTextPort)` + `build_local_whisper_stt(...)`; lazy faster-whisper import; injected transcriber for testing.
- **Create:** `tests/gateway/streaming/test_local_whisper_stt.py` — mocked unit tests (1–5, 7) + real-ASR contract (6, skipif-guarded).
- **Modify:** `pyproject.toml` — add `simplex-streaming-local-stt` extra; `uv.lock` regenerate.
- **Do NOT modify:** `.github/workflows/tests.yml` (no CI install change; real test skips in CI). `streaming/__init__.py` (no eager export forcing the import).

## 6. Acceptance criteria

- All scenarios pass; mocked tests run in CI; real-ASR test runs locally (faster-whisper installed) and skips cleanly in CI.
- Package imports without the `simplex-streaming-local-stt` extra (lazy import).
- ast-grep no-walltime / ruff / ty clean over the new file. `FakeSTT` stays the default; no session/engine behavior change.
- `uv lock --check` consistent; CI shards green except the documented pre-existing `test_setup*` + private-fork `osv-scan`.

## 7. Out of scope (later slices)

- **Deepgram Flux** streaming STT (cloud, gated, real partials + native end-of-turn) — a later slice.
- PARTIAL transcript emission / eager response.
- Real Cartesia TTS (Slice 5), aiortc transport (Slice 6), E2E (Slice 7), live call (Slice 8).
