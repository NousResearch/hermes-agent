# SimpleX Streaming Voice — Slice 6: Real aiortc AudioTransportPort Bridge (Design)

> Status: DESIGN (revised after spec review — **scope split**). Architecture **A — ports own the loop**.
>
> **Slice 6 (this spec, CI-verifiable):** the pure-asyncio core — `AiortcStreamingTransport(AudioTransportPort)` with an *injected* outbound sink + a `process_pcm16`-shaped `StreamingPipeline` running the session with **fake** cognitive ports, and the engine STREAMING branch **constructing** it (un-defers Slice 2). No aiortc import in the streaming core → fully testable in CI (aiortc/av are NOT installed in CI). Real-port wiring is Slice 7.
> **Slice 6b (split out, aiortc-only, validated live):** the real `MediaStreamTrack` (16k↔48k), the `_DirectFeedAccumulator` bypass in `AiortcAudioPeer.start()`, and the cross-object wiring that attaches the peer-created track as the transport's `outbound_sink`. These skip in CI (aiortc absent) and get their real validation at the live call (Slice 8), so they land adjacent to it.
>
> Turn-based path unchanged; fakes stay default for non-streaming.

## 1. Goal

Bridge the proven aiortc media handling into the streaming reflex:
- `AiortcStreamingTransport(AudioTransportPort)` — pure-asyncio, **no aiortc import** in its core: inbound queue (fed externally) + outbound via an injected sink; `flush_outbound` drops pending (barge-in).
- `StreamingPipeline` — implements the live engine's pipeline contract `async process_pcm16(*, call_id, pcm16, sample_rate)`; lazily starts `StreamingCallSession.run()` as a background task, turns each inbound PCM chunk into a `push_inbound(AudioFrame)` (resampled to 16k), returns non-blocking; `aclose()` closes the transport (inbound sentinel) and awaits the session task.
- The engine STREAMING branch constructs a `StreamingPipeline` (guarded by aiortc availability) **instead of raising** `PipecatIntegrationDeferred`. A `cognitive="fake"|"real"` selector defaults to `fake` this slice.
- A live outbound PCM `MediaStreamTrack` (16k→48k resample, 20ms pacing, `drop_pending`) + a direct-feed shim that bypasses the turn-based RMS accumulator in streaming mode — both live in `aiortc_engine.py`/`webrtc_media.py` (**outside `streaming/**`** so wall-clock pacing is allowed).

## 2. Validated facts (from the live code)

- `AudioTransportPort`: `media` (property), `async push_inbound(frame)`, `inbound() -> AsyncIterator[AudioFrame]`, `async emit_outbound(frame)`, `async flush_outbound(reason) -> FlushResult`, `async close()`. `FakeAudioTransport` uses an `asyncio.Queue` + `None` sentinel for inbound and tolerates `FlushResult(..., last_sent_mark=None)`.
- Live engine: `build_native_pipeline()` returns a pipeline fed by `await pipeline.process_pcm16(call_id=, pcm16=, sample_rate=)`. Turn-based `HermesVoiceTurnPipeline` implements it. The relay loop (`aiortc_engine.py:_relay_audio`) does `track.recv()` → `_audio_frame_to_pcm16` (PyAV resampler at the frame's rate) → `AudioUtteranceAccumulator.accept_pcm16()` (RMS-VAD utterance batching) → `pipeline.process_pcm16(...)` per utterance.
- **Rate:** `SimplexAiortcConfig.sample_rate` default **48000**; the streaming ports require strict **16000 mono** (they raise otherwise). So `process_pcm16` must resample 48k→16k inbound; the outbound track must resample 16k→48k.
- Outbound (turn-based): `QueuedFileAudioTrack.recv()` pulls **file paths**, transcodes via ffmpeg, paces 20ms, silence when empty. No PCM-frame ingress, no flush — a **new** PCM-frame track is needed for streaming.
- `StreamingCallSession.run()` drains `transport.inbound()` to EOS, emits TTS `AudioFrame`s via `emit_outbound`, calls `flush_outbound` on barge-in. The **session is the turn detector** — the accumulator's RMS-VAD must be bypassed in streaming mode.
- aiortc 1.14 / av 16.1 in `simplex-native-calls`; session/ports in `simplex-streaming`. ast-grep no-walltime scopes `streaming/**` only.

## 3. Decisions

- **D1 — Transport core (no aiortc):** `AiortcStreamingTransport(media, *, clock, outbound_sink: Callable[[AudioFrame], Awaitable[None]])`. Inbound: `_inbound: asyncio.Queue[AudioFrame|None]`; `push_inbound` puts; `inbound()` async-gen yields until `None`; `close()` puts the sentinel (idempotent). Outbound: `emit_outbound` appends to `_pending` and `await`s `outbound_sink(frame)`; `flush_outbound(reason)` clears `_pending`, computes `dropped_frames=len(pending)`, `dropped_ms=Σ frame.duration_ms`, requests the sink/track to drop its un-played queue (via a `drop` hook on the sink), returns `FlushResult(dropped_frames, dropped_ms, last_sent_mark=None)` (mark tracking stays in the ledger — matches `FakeAudioTransport`). No aiortc/av import here.
- **D2 — `StreamingPipeline` (the process_pcm16 adapter):** builds the transport + a `StreamingCallSession` (with chosen ports + injected `Clock`); the `StreamingCallContext` it builds uses a **16k `MediaFormat`** (the ports require 16k; `session.run()` does not read `transport.media`, but TTS frames use `ctx.media`). `is_streaming = True` (marker the engine reads). `process_pcm16(*, call_id, pcm16, sample_rate)`: lazily start `self._session_task = asyncio.create_task(session.run())` on first call; resample `pcm16` `sample_rate`→16k mono via stdlib `audioop.ratecv` (lazy import — `audioop` is stdlib on the CI Python 3.11; the no-aiortc-in-streaming constraint forces `audioop` over `av.AudioResampler` here); `await transport.push_inbound(AudioFrame(pcm16_16k, media=16k, timestamp_ms=clock.now_ms(), seq=next))`; return a lightweight ack object. **The ack is consumed only by the Slice-6b `_DirectFeedAccumulator`, which discards it** (no `.ok`/`.audio_path` read; outbound flows via the transport's sink, not `queue_file`). `process_audio_file` (the CLI debug consumer that expects a `VoiceTurnResult`) is **out of scope for streaming mode** — not reached when `is_streaming`. `aclose()`: `await transport.close()` then `await self._session_task` (cancel-safe).
- **D3 — Scope = FAKE cognitive ports (Q2=a):** Slice 6 runs the session with `FakeTurnDetection`/`FakeSTT`/`FakeTTS`/`FakeBrain` (realistic 16k/20ms frame shapes) to prove the media seam end-to-end in CI **without** onnx/whisper/piper. `build_native_pipeline(..., cognitive="fake"|"real")` gains the selector (default `fake`); Slice 7 flips to `real`. The fakes are production-wired (not test doubles) — they make the streaming engine path runnable/observable now.
- **D4 — Engine wiring (Slice 6):** the STREAMING branch constructs `StreamingPipeline(...)` guarded by `_aiortc_available()` (+ the engine flag), **replacing** the `PipecatIntegrationDeferred` raise. If aiortc absent → clear error (like Slice 2's `StreamingExtraNotInstalled`). The engine constructs the pipeline with an **injected outbound sink**: in Slice 6 a buffering/observable sink (proves construction + the media-seam logic in CI); the real peer-owned track is attached in Slice 6b. Turn-based default unchanged. The engine-construct test asserts the **aiortc-absent → clear-error** path (always runs in CI) and, `importorskip`-guarded, the aiortc-present → returns-`StreamingPipeline` path.
- **D5 — [SLICE 6b — deferred, aiortc-only, validated live]:** add `_create_pcm_streaming_track(target_rate)` in `aiortc_engine.py` — a `MediaStreamTrack` with an `asyncio.Queue[AudioFrame]`; `recv()` pops a 16k frame, resamples 16k→`config.sample_rate` via `av.AudioResampler`, paces 20ms, silence when empty; plus `enqueue(frame)` (becomes the transport's `outbound_sink`) and `drop_pending()` (for flush). In `AiortcAudioPeer.start()`, when `getattr(pipeline, "is_streaming", False)`: build the **streaming** output track (not `_create_queued_audio_track`), **attach it as the pipeline's transport `outbound_sink`/drop hook** (the track is created in `start()` *after* the pipeline is built, so `start()` reads `pipeline.transport` and wires the sink — the one cross-object seam), and feed the relay via a `_DirectFeedAccumulator(pipeline)` (`async accept_pcm16(self, pcm16, *, now=None)` → `await pipeline.process_pcm16(...)`, **return discarded**) instead of the RMS `AudioUtteranceAccumulator`. Wall-clock pacing is allowed here (outside `streaming/**`). Tested `importorskip`-guarded (skips in CI); real validation at Slice 8.
- **D6 — Resampling:** inbound 48k→16k in `process_pcm16` (stdlib `audioop.ratecv`, lazy import — add `audioop-lts` marker dep where the transport extra is declared, or rely on `simplex-native-calls` deps; confirm). Outbound 16k→48k in the live track via `av.AudioResampler`. Ports stay strict-16k.
- **D7 — Packaging:** reuse `simplex-native-calls` (aiortc/av) + `simplex-streaming` (session). No new extra. Lazy aiortc/av imports (track factory + resampler only); the transport core and `StreamingPipeline` import no aiortc, so the streaming package stays importable without the extra.

## 4. BDD scenarios

1. **Transport inbound queue**: `push_inbound(f)` then `inbound()` yields `f`; after `close()`, `inbound()` terminates (sentinel). Deterministic, no aiortc.
2. **emit_outbound → sink**: `emit_outbound(f)` calls the injected async sink with `f` and records it pending.
3. **flush_outbound drops + reports**: after N `emit_outbound`, `flush_outbound("barge")` clears pending, invokes the sink's `drop` hook, returns `FlushResult(dropped_frames=N, dropped_ms=20*N, last_sent_mark=None)`.
4. **StreamingPipeline.process_pcm16 feeds the session**: feeding synthetic 48k PCM resamples to 16k and pushes an `AudioFrame(media=16k, seq monotonic, timestamp from injected clock)` to a spy transport/session; returns promptly (non-blocking).
5. **Full fake-port media seam**: a `StreamingPipeline` with fake ports + an observable outbound sink — feed a scripted inbound utterance via `process_pcm16`; assert the fake brain/TTS produce outbound `AudioFrame`s that reach the sink, ending in playback; `aclose()` cleanly stops the session task.
6. **Barge-in through the seam**: while the fake TTS is emitting, a fake turn event triggers the session's `flush_outbound`; assert the sink's drop hook fired and a `FlushResult` with dropped frames was produced.
7a. **Engine STREAMING, aiortc absent → clear error (ALWAYS runs in CI)**: with `_aiortc_available()` patched/False, `build_native_pipeline(streaming-config, cognitive="fake", turn_based_factory=...)` raises a clear error naming the requirement — this is the always-on CI coverage of the new engine branch. Must NOT import `StreamingPipeline` in a way that transitively imports aiortc (the streaming core imports no aiortc, so the raise path collects fine).
7b. **Engine STREAMING, aiortc present → returns StreamingPipeline** (`importorskip("aiortc")`, SKIPS in CI): returns an object with `process_pcm16` + `is_streaming=True`; turn_based default still returns the legacy pipeline.
8. **[SLICE 6b] Live track contract** (skipif `importorskip("aiortc","av")`): `_create_pcm_streaming_track(48000)` accepts enqueued 16k frames, `recv()` returns a resampled 48k `av.AudioFrame`, `drop_pending()` empties the queue; the `_DirectFeedAccumulator`+`start()` wiring attaches the track as the transport sink. (Lives outside streaming/**; tested under tests/gateway/; skips in CI.)

## 5. Files

**Slice 6 (this slice):**
- **Create:** `gateway/calls/native/streaming/aiortc_transport.py` — `AiortcStreamingTransport` + `StreamingPipeline` (**no** top-level aiortc/av import; lazy `audioop` for resample). Despite the filename, the module imports no aiortc — the name reflects its role.
- **Create:** `tests/gateway/streaming/test_aiortc_transport.py` — scenarios 1–6 (pure asyncio, fakes/observable sink) + 7a (engine construct, aiortc-absent raise; always CI) + 7b (aiortc-present, importorskip).
- **Modify:** `gateway/calls/native/streaming/engine.py` — STREAMING branch constructs `StreamingPipeline` (aiortc-guarded, injected buffering sink), add `cognitive` selector; replace the deferral raise.
- **Do NOT:** add a top-level export to `streaming/__init__.py` that forces aiortc import; change the turn-based path; add cloud.

**Slice 6b (deferred):**
- **Modify:** `gateway/calls/native/aiortc_engine.py` — `_create_pcm_streaming_track` + `_DirectFeedAccumulator` + streaming branch in `AiortcAudioPeer.start()` (attach track as transport sink). **Create/extend:** a test under `tests/gateway/` — scenario 8 (importorskip-guarded).

## 6. Acceptance criteria

- **Slice 6:** scenarios 1–6 + **7a run in CI** (pure asyncio; aiortc NOT installed in CI — confirmed `[all]` excludes `simplex-native-calls`); 7b is `importorskip`-guarded (skips in CI). The new engine STREAMING branch has always-on CI coverage via the 7a aiortc-absent raise.
- The engine STREAMING path **constructs** (no `PipecatIntegrationDeferred`) with `cognitive="fake"`; with aiortc available the live consumer gets an object with `process_pcm16` + `is_streaming=True`.
- `streaming/` package imports without aiortc (transport core + `StreamingPipeline` import no aiortc at module top — `audioop` lazy inside `process_pcm16`).
- ast-grep no-walltime clean over `streaming/**` (the new transport uses injected `Clock`; no `time.*`/`asyncio.sleep`). ruff + ty clean. Turn-based path + `Fake*` defaults unchanged.
- CI shards green except documented pre-existing `test_setup*` + private-fork `osv-scan`.
- **Slice 6b** acceptance is validated locally/at the live call (scenario 8 skips in CI).

## 7. Out of scope (later slices)

- **Slice 6b:** the live aiortc PCM track (16k↔48k), `_DirectFeedAccumulator` bypass, and the cross-object sink wiring in `AiortcAudioPeer.start()` (D5). aiortc-only, skips in CI, validated at the live call — lands adjacent to Slice 8.
- Wiring the **real** cognitive ports (Silero/Smart-Turn, faster-whisper, Piper, Hermes brain) into the streaming engine — **Slice 7** (flip `cognitive="real"`, full simulated E2E).
- Live iPhone round trip + gap fixes — **Slice 8** (human-gated).
- Cloud STT/TTS (Deepgram/Cartesia).
