# SimpleX Streaming Voice — Slice 6b: Live aiortc PCM Track + Bypass (Design)

> Status: DESIGN. Architecture **A — ports own the loop**. The aiortc-only half of the transport bridge that Slice 6 split out. Touches `gateway/calls/native/aiortc_engine.py` only (plus an importorskip-guarded test). Skips in CI (aiortc/av not installed there); real validation happens at the live call (Slice 8). Builds on merged Slice 6 core (`AiortcStreamingTransport` + `StreamingPipeline` in `streaming/aiortc_transport.py`).

## 1. Goal

Make the constructed `StreamingPipeline` actually carry live audio:
1. A real outbound `MediaStreamTrack` that plays the session's 16k TTS `AudioFrame`s out the aiortc peer at the SimpleX rate (48k), with flush/drop for barge-in.
2. A direct-feed inbound path so the relay loop feeds `process_pcm16` per frame (the session is the turn detector — bypass the turn-based RMS `AudioUtteranceAccumulator`).
3. The cross-object wiring that attaches the peer-created track as the pipeline transport's `outbound_sink`/drop hook.
4. The robustness items carried over from the Slice 6 code review (back-pressure, `aclose` semantics, resampler state, typed sink contract).

## 2. Components

- **`_create_pcm_streaming_track(target_rate)`** (aiortc_engine.py, outside `streaming/**` so wall-clock pacing is allowed):
  - a `MediaStreamTrack` (kind="audio") with a **bounded** `asyncio.Queue[AudioFrame]` (M-I2 back-pressure: bounded, drop-oldest on overflow with a `log()` so truncation isn't silent).
  - `recv()`: pop a 16k frame, resample 16k→`target_rate` via a **persistent** `av.AudioResampler` (M1: keep the resampler instance across calls so phase is continuous, not reset per frame), pace 20ms (`time`/`asyncio.sleep` OK here), emit silence when the queue is empty.
  - `async enqueue(frame)` — the transport's `outbound_sink`.
  - `drop_pending()` — empties the queue (barge-in flush); becomes the sink's `.drop` hook. Reports count for the FlushResult path.
- **`_DirectFeedAccumulator(pipeline)`** (aiortc_engine.py): `async accept_pcm16(self, pcm16, *, now=None)` → `await pipeline.process_pcm16(call_id=…, pcm16=pcm16, sample_rate=<native>)`; **discards the ack** (no `.ok`/`.audio_path`; outbound goes via the sink). Signature matches the relay's `await accumulator.accept_pcm16(pcm16)`.
- **`AiortcAudioPeer.start()` streaming branch**: when `getattr(pipeline, "is_streaming", False)`:
  - build the streaming output track via `_create_pcm_streaming_track(config.sample_rate)` (instead of `_create_queued_audio_track`);
  - **attach the track as the pipeline transport's sink**: `pipeline.transport.set_outbound_sink(track.enqueue, drop=track.drop_pending)` — requires the **B1** + **B2** + **B3** changes below;
  - use `_DirectFeedAccumulator(pipeline)` for the relay instead of `AudioUtteranceAccumulator`;
  - **retain the pipeline** on the peer (`self._pipeline = pipeline`) so teardown can close it.
  - **Teardown wiring (A5 gap):** today `_close_session`/peer `close()` does not touch the pipeline (the session dict holds it separately and never calls it). Fix: the peer stores `self._pipeline` in `start()`, and peer `close()` / `_close_session` calls `await pipeline.aclose()` (guarded `if getattr(pipeline,"is_streaming",False)`), so the session task is drained on call end.
- **Slice-6 core changes required first (B1/B2/B3 — small edits to `streaming/aiortc_transport.py`, CI-tested there):**
  - **B1 — expose the transport:** `StreamingPipeline` gains a public `@property def transport(self) -> AiortcStreamingTransport: return self._transport` (currently private/unexposed). `start()`'s `pipeline.transport.set_outbound_sink(...)` depends on this.
  - **B2 — resolve the drop contract (currently broken-if-used):** Slice-6's `flush_outbound` reads `.drop` as an *attribute on the sink callable*, but a bound method `track.enqueue` has no `.drop`. Decision: `set_outbound_sink(sink, *, drop=None)` stores `self._outbound_sink = sink` **and** `self._outbound_drop = drop` in a **separate field**; `flush_outbound` calls `self._outbound_drop` if set (awaiting if it returns a coroutine), falling back to `getattr(sink, "drop", None)` for the `__init__`-sink case. Without this, barge-in flush silently no-ops. Update the typed `OutboundSink` (M4) so `drop` is an *optional separate hook*, not a required sink attribute.
  - **B3 — no throwaway sink on the live path:** `build_streaming_pipeline(..., sink=None)` — when `sink is None` (the engine/live path), construct the transport with a **no-op** sink (frames buffer harmlessly until `start()` calls `set_outbound_sink`) OR defer transport construction; the recording sink is used only by the in-CI seam tests that pass an explicit sink. This removes the race where the session emits into a throwaway recording sink before `start()` swaps in the live track.
  - **I1**: comment the no-await invariant on `_ensure_started`.
  - **I3**: `aclose(self, *, abort: bool=False)` clears `_task` in a `finally` (idempotent); `abort=True` `cancel()`s the session task then awaits-with-suppression (graceful-drain default; abort for forced teardown).
  - **M4**: a typed `OutboundSink` Protocol (`async __call__(frame)`) + the `set_outbound_sink(sink, *, drop=None)` setter (drop is a separate optional hook per B2).
  - **I2** back-pressure: the track's outbound queue is bounded (drop-oldest + `log`); the transport's *inbound* queue stays unbounded but is paced by RTP arrival (~50 fps) and drained per-frame — revisit if Slice 7's real STT/brain can't keep up.

## 3. BDD scenarios (all `importorskip("aiortc","av")` — skip in CI; run locally/at Slice 8)

1. **Track plays enqueued audio**: `t=_create_pcm_streaming_track(48000)`; `await t.enqueue(AudioFrame(16k,640B))`; `await t.recv()` returns an `av.AudioFrame` at 48000 Hz with samples (resampled).
2. **Silence when empty**: `recv()` on an empty queue returns a silence frame (no hang) within ~one frame interval.
3. **drop_pending flush**: enqueue N, `drop_pending()` → queue empty; reports N.
4. **Resampler continuity (M1)**: feeding two consecutive 16k frames yields 48k output whose total sample count ≈ 3× input across the pair (state preserved; no per-frame truncation).
5. **Direct-feed bypass**: `_DirectFeedAccumulator(spy_pipeline).accept_pcm16(pcm)` calls `pipeline.process_pcm16(...)` once and ignores the return.
6. **start() wiring** (split for CI coverage, A7): a **CI-runnable** portion drives the streaming branch of `start()` with a **fully faked peer** (no aiortc — stub the RTCPeerConnection/track-add bits) and a real `StreamingPipeline`, asserting `pipeline.transport`'s sink was replaced via `set_outbound_sink` (a spy) and that `_DirectFeedAccumulator` was selected (not `AudioUtteranceAccumulator`); plus an **importorskip** portion exercising the real track. Without the CI-runnable portion, regressions in the `start()` streaming branch wouldn't be caught.

CI-side carry-in tests (in `streaming/aiortc_transport.py`, **run in CI** — no aiortc needed): the `transport` property (B1); `set_outbound_sink` + separate `drop` field + `flush_outbound` calling it (B2 — assert barge-in flush actually invokes the drop hook); `build_streaming_pipeline(sink=None)` uses a no-op sink (B3); `aclose` idempotency + abort-cancel path (I3); typed `OutboundSink` (M4). These get unit tests in `test_aiortc_transport.py`.

## 4. Files

- **Modify:** `gateway/calls/native/aiortc_engine.py` — `_create_pcm_streaming_track`, `_DirectFeedAccumulator`, the `start()` streaming branch.
- **Modify:** `gateway/calls/native/streaming/aiortc_transport.py` — `set_outbound_sink` setter, typed `OutboundSink`, `aclose(abort=…)` idempotent, `_ensure_started` invariant comment.
- **Create:** `tests/gateway/test_native_streaming_track.py` (importorskip-guarded, aiortc) — scenarios 1–6.
- **Extend:** `tests/gateway/streaming/test_aiortc_transport.py` — CI-side carry-in tests (aclose idempotency/abort, set_outbound_sink).

## 5. Acceptance criteria

- CI-side carry-in tests (aclose idempotent + abort, set_outbound_sink, typed sink) pass in CI; aiortc track tests skip in CI, pass locally.
- The streaming `start()` branch builds the PCM track, wires it as the transport sink, and uses the direct-feed accumulator — verified by the local/wiring test.
- Turn-based path unchanged (non-streaming pipelines still use `_create_queued_audio_track` + `AudioUtteranceAccumulator`).
- ast-grep no-walltime stays clean over `streaming/**` (the track + pacing are in `aiortc_engine.py`, outside the glob). ruff + ty clean.
- Sets up Slice 8: with `cognitive="real"` (Slice 7) + this live wiring, a real SimpleX call can flow audio both ways.

## 6. Out of scope

- Real cognitive ports in the live session — **Slice 7** (`cognitive="real"`).
- The live iPhone call itself — **Slice 8** (human-gated).
- Cloud STT/TTS.
