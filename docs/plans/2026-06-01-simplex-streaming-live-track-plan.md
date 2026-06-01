# Slice 6b: Live aiortc PCM Track + Bypass — Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD. Design: `docs/plans/2026-06-01-simplex-streaming-live-track-design.md`. Builds on merged Slice 6 core.

**Goal:** the aiortc-only half of the transport bridge — a live outbound PCM `MediaStreamTrack` + direct-feed inbound + the cross-object sink wiring in `start()`, plus the CI-testable Slice-6 carry-in fixes (B1/B2/B3/I3/M4). aiortc track tests skip in CI; carry-in fixes run in CI; the seam validates at the live call (Slice 8).

## Task 1: Slice-6 core carry-in fixes (CI-tested) — `streaming/aiortc_transport.py`

**Files:** Modify `gateway/calls/native/streaming/aiortc_transport.py`; extend `tests/gateway/streaming/test_aiortc_transport.py`.

Read first: the current `AiortcStreamingTransport` (__init__ outbound_sink, emit_outbound, flush_outbound's `.drop` lookup), `StreamingPipeline` (_transport, _ensure_started, aclose), `build_streaming_pipeline` (the recording sink).

- [ ] **Step 1 (TDD):** failing tests:
  - `test_pipeline_exposes_transport` — `pipe.transport is pipe._transport` (B1).
  - `test_set_outbound_sink_replaces_and_drop_hook_fires` — construct transport with a no-op sink; `set_outbound_sink(new_sink, drop=drop_spy)`; `emit_outbound` routes to new_sink; `flush_outbound("x")` awaits/calls `drop_spy` and returns the right `FlushResult` (B2 — the drop hook is a SEPARATE field, not a sink attribute).
  - `test_flush_drop_fallback_to_sink_attribute` — if `set_outbound_sink` not used and the ctor sink has a `.drop` attr, flush still calls it (back-compat).
  - `test_build_streaming_pipeline_sink_none_uses_noop` — `build_streaming_pipeline(config, cognitive="fake", sink=None)` constructs without a recording sink; emitting before a sink is set doesn't crash (no-op sink) (B3).
  - `test_aclose_idempotent_and_abort` — second `aclose()` is a no-op (no re-raise); `aclose(abort=True)` cancels a still-running session task and returns without raising (I3).
- [ ] **Step 2 → red.**
- [ ] **Step 3:** implement:
  - `@property def transport(self): return self._transport` (B1).
  - typed `OutboundSink` Protocol (`async def __call__(self, frame: AudioFrame) -> None`); `set_outbound_sink(self, sink, *, drop=None)` sets `self._outbound_sink=sink; self._outbound_drop=drop` (M4/B2). `__init__` keeps accepting an initial sink; default `_outbound_drop=None`.
  - `flush_outbound`: `drop = self._outbound_drop or getattr(self._outbound_sink, "drop", None)`; if drop: `r=drop(); if iscoroutine(r): await r` (B2).
  - `build_streaming_pipeline(..., sink=None)`: if `sink is None`, use a module-level `async def _noop_sink(frame): return None` (B3); the recording sink is only used when a test passes one explicitly. (Keep the existing recording-sink helper for tests, or move it to the test file.)
  - `aclose(self, *, abort=False)`: `try: ... finally: self._task=None`; if `abort and self._task and not self._task.done(): self._task.cancel()` then await-suppress `CancelledError` (I3). Comment the no-await invariant on `_ensure_started` (I1).
- [ ] **Step 4 → green.** Commit — `fix(streaming): transport carry-ins (transport property, set_outbound_sink, drop hook, aclose abort)`.

## Task 2: Live PCM track + direct-feed + start() wiring — `aiortc_engine.py`

**Files:** Modify `gateway/calls/native/aiortc_engine.py`; Create `tests/gateway/test_native_streaming_track.py` (importorskip aiortc/av) + a CI-runnable faked-peer wiring test.

Read first: `_create_queued_audio_track`/`QueuedFileAudioTrack` (MediaStreamTrack base, recv(), `_pace`, `_frame_from_pcm16`, pts/time_base), `_audio_frame_to_pcm16` (av.AudioResampler API), `AiortcAudioPeer.start()` (track + accumulator creation, the relay call `await accumulator.accept_pcm16(pcm16)`), `_close_session`/peer `close()`.

- [ ] **Step 1:** `_create_pcm_streaming_track(target_rate)` — a `MediaStreamTrack` (kind="audio"); bounded `asyncio.Queue[AudioFrame]` (drop-oldest + `logger.warning` on overflow); a **persistent** `av.AudioResampler(format="s16", layout="mono", rate=target_rate)`; `recv()` pops a 16k frame, resamples (handle 0-frame returns; re-stamp `pts += samples`, `time_base=Fraction(1,target_rate)` like `_frame_from_pcm16`), paces 20ms, silence when empty; `async enqueue(frame)`; `drop_pending()` (empty queue, return count). `_DirectFeedAccumulator(pipeline)`: `async accept_pcm16(self, pcm16, *, now=None): await pipeline.process_pcm16(call_id=self._call_id, pcm16=pcm16, sample_rate=self._native_rate)` (discard ack).
- [ ] **Step 2:** `start()` streaming branch: `if getattr(pipeline, "is_streaming", False):` build `_create_pcm_streaming_track(config.sample_rate)`, add it to the pc, `pipeline.transport.set_outbound_sink(track.enqueue, drop=track.drop_pending)`, set `self._accumulator = _DirectFeedAccumulator(pipeline, call_id, native_rate)`, `self._pipeline = pipeline`. Else the existing path. In peer `close()`/`_close_session`: `if getattr(self._pipeline, "is_streaming", False): await self._pipeline.aclose()`.
- [ ] **Step 3:** tests:
  - `test_native_streaming_track.py` (importorskip aiortc, av): scenarios 1–4 (enqueue→recv resampled 48k; silence when empty; drop_pending; resampler continuity); scenario 5 direct-feed (spy pipeline). 
  - CI-runnable wiring test (faked peer, no aiortc): drive the `start()` streaming branch with a fake pc + real `StreamingPipeline`; assert `set_outbound_sink` called + `_DirectFeedAccumulator` selected + pipeline retained for close.
- [ ] **Step 4:** Commit — `feat(calls): live streaming PCM track + direct-feed bypass + start() wiring`.

## Task 3: Gates
- [ ] ast-grep no-walltime over `streaming/**` clean (the track/pacing is in aiortc_engine.py, outside the glob). ruff + ty clean on both modules.
- [ ] Package imports without aiortc (transport carry-ins don't add aiortc imports).
- [ ] `uv run --no-sync python -m pytest tests/gateway/streaming/test_aiortc_transport.py tests/gateway/test_native_streaming_track.py -q` → carry-ins pass; track tests pass locally (aiortc installed).

## Done criteria
Carry-in fixes (B1/B2/B3/I3/M4) pass in CI; live track + wiring pass locally (skip in CI); turn-based path unchanged; gates clean. Then PR → CI → fix obvious → merge on green (track tests skip in CI by design; modulo pre-existing `test_setup*` + `osv-scan`). Sets up Slice 7 (cognitive="real") then Slice 8 (live, human-gated).
