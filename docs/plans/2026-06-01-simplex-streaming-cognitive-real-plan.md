# Slice 7: `cognitive="real"` + Simulated E2E — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD. Frequent commits. Design: `docs/plans/2026-06-01-simplex-streaming-cognitive-real-design.md`. Builds on merged Slices 1–6b.

**Goal:** Flip `build_streaming_pipeline(cognitive="real")` from `NotImplementedError` to a real-port wiring (turn detection + STT + TTS + brain) via an injected-backend seam; add a creds-free simulated end-to-end harness + tests; land three carry-in fixes (#1 resampler drain, #2 inbound frame-rate, #3 bounded inbound queue).

**Architecture:** Ports own the loop. `cognitive="real"` resolves each port from an injectable param (default = real builder) on a `MonotonicClock`. The E2E sim swaps only the brain for a deterministic `FakeBrain`, keeping real STT/TTS, so the audio reflex path runs reproducibly without LLM creds. Real-dep tests `skipif`/`importorskip` (skip in CI, run locally); wiring + queue tests run in CI.

**Tech Stack:** Python asyncio, pipecat (Silero VAD / Smart Turn v3), faster-whisper, piper-tts, aiortc/av (engine track), pytest.

---

## Task 1: `cognitive="real"` factory branch + injectable ports (CI-tested wiring)

**Files:**
- Modify: `gateway/calls/native/streaming/aiortc_transport.py` (`build_streaming_pipeline`)
- Test: `tests/gateway/streaming/test_aiortc_transport.py`

Read first: `build_streaming_pipeline` (lines ~288-355, the `cognitive="fake"` branch for the construction shape), `_noop_sink`, `StreamingPipeline` ctor; `session.py` ctor + `BrainFactory = Callable[[], HermesBrainPort]`; `brain.py` `HermesSyncBrain` + `build_call_agent_factory`; `clock.py` `MonotonicClock`.

- [ ] **Step 1 (TDD): failing CI tests** (no real deps — inject fakes):
  - `test_build_streaming_pipeline_real_wiring`: call `build_streaming_pipeline(config, cognitive="real", turn_detector=FakeTurnDetection([]), stt=FakeSTT(), tts=FakeTTS(VirtualClock()), brain_factory=lambda: FakeBrain(VirtualClock(), text="hi"), clock=VirtualClock())`. Assert the returned `StreamingPipeline.is_streaming is True`, and that the underlying `session` has exactly the injected `turns`/`stt`/`tts` objects and the `clock` is the injected `VirtualClock`. (Access via `pipe._session` — `StreamingPipeline` stores `self._session` at aiortc_transport.py:197, and the session stores `self.turns`/`self.stt`/`self.tts`/`self.clock`/`self.brain_factory` at session.py:91-98.)
  - `test_real_brain_factory_default_resolves_without_invoking`: call `build_streaming_pipeline(config, cognitive="real", turn_detector=fake, stt=fake, tts=fake, clock=VirtualClock())` with `brain_factory` omitted; capture `pipe._session.brain_factory`; assert calling it returns a `HermesSyncBrain` instance (this constructs `HermesSyncBrain(build_call_agent_factory())` but does NOT call `.respond`, so no LLM/network). Assert `isinstance(brain, HermesSyncBrain)`.
  - **Remove** the existing `test_build_streaming_pipeline_real_not_implemented` (~lines 420-422).
- [ ] **Step 2 → red** (`NotImplementedError` still raised / new params not accepted).
- [ ] **Step 3: implement.** Add params `brain_factory=None, turn_detector=None, stt=None, tts=None` to `build_streaming_pipeline`. Replace the `if cognitive == "real": raise NotImplementedError(...)` block with a real branch (mirror the fake branch's construction, but):
  - `from .clock import MonotonicClock`; `from .brain import HermesSyncBrain, build_call_agent_factory`; reuse the same imports for `InterruptionPolicy`, `StreamingCallSession`, `StreamingCallTracer`, `StreamingCallContext`, `MediaFormat`, `AiortcStreamingTransport`.
  - `clk = clock or MonotonicClock()`; `media = MediaFormat(16000, 1, 20)`; `call_id = "streaming-real"`.
  - Resolve ports (lazy real builders raise clear RuntimeErrors when extras absent — do NOT pre-guard):
    - `turns = turn_detector or build_local_turn_detector(media, call_id=call_id)`
    - `stt_port = stt or build_local_whisper_stt(media, call_id=call_id)`
    - `tts_port = tts or build_piper_tts(media, clock=clk, call_id=call_id)`
    - `bf = brain_factory or (lambda: HermesSyncBrain(build_call_agent_factory()))`
    - Import the three builders lazily inside the real branch (so the module still imports without pipecat/whisper/piper): `from .local_turn_detection import build_local_turn_detector` etc.
  - `outbound_sink = _noop_sink if sink is None else sink`; build transport, ctx, session (`brain_factory=bf`), tracer; `return StreamingPipeline(media=media, session=session, transport=transport, clock=clk)`.
  - Keep `cognitive` not in {fake, real} → `ValueError`.
- [ ] **Step 4 → green.** `uv run --no-sync python -m pytest tests/gateway/streaming/test_aiortc_transport.py -q`.
- [ ] **Step 5: commit** — `feat(streaming): build_streaming_pipeline cognitive="real" via injected-backend seam`.

---

## Task 2: `build_real_stream_simulation` + real E2E test (skipif real deps)

**Files:**
- Modify: `gateway/calls/native/streaming/simulate.py`
- Create: `tests/gateway/streaming/test_real_e2e_simulation.py`

Read first: `simulate.py` in full (`build_stream_simulation`, `StreamSimulation`, `_MEDIA`, `_make_frame`); `fakes.py` `FakeAudioTransport` (`.sent`/`.flushes`/`push_inbound`/`end_inbound`), `FakeTurnDetection`, `FakeBrain`; an existing real-port test that chunks the wav (e.g. `tests/gateway/streaming/test_local_whisper_stt.py`) for the `wave.open`→640-byte-chunk pattern; the fixture `tests/gateway/streaming/fixtures/turn_detection/speech_16k_mono.wav`.

- [ ] **Step 1: implement `build_real_stream_simulation`** in `simulate.py` (pure synchronous construction; **no `asyncio.sleep`** — no-walltime rule):
  ```python
  def build_real_stream_simulation(*, call_id="real-stream-sim", contact_id="sim-contact",
      response_text="It's sunny today.", barge_in=False,
      turn_detector=None, stt=None, tts=None, clock=None) -> StreamSimulation:
  ```
  - `from .clock import MonotonicClock`; `clk = clock or MonotonicClock()`; `media = _MEDIA`.
  - Lazy-import the real builders inside the function (module must import without extras): `build_local_turn_detector`, `build_local_whisper_stt`, `build_piper_tts`.
  - **Normal case (`barge_in=False`):** real turn detector + real STT + real TTS.
    - `turns = turn_detector or build_local_turn_detector(media, call_id=call_id)`
    - `stt_port = stt or build_local_whisper_stt(media, call_id=call_id)`
    - `tts_port = tts or build_piper_tts(media, clock=clk, call_id=call_id)`
  - **Barge-in case (`barge_in=True`):** per design §3.3 (revised), inject BOTH a scripted `FakeTurnDetection` AND a deterministic `FakeSTT` (real whisper can't deterministically finalize on a scripted endpoint → empty transcript → no turn to barge into). Keep **real TTS**.
    - `turns = turn_detector or FakeTurnDetection([(0, ENDPOINT_DETECTED), (1, USER_SPEECH_STARTED), (2, USER_SPEECH_STOPPED)])` (mirror `build_stream_simulation`'s barge-in script, simulate.py:157-162).
    - `stt_port = stt or FakeSTT(partials=[_partial_transcript(call_id, "hold on")], final=_final_transcript(call_id, "tell me a story"))` (the `"hold on"` partial satisfies the `min_words=2` interruption policy; reuse the `_partial_transcript`/`_final_transcript` helpers already in simulate.py).
    - `tts_port = tts or build_piper_tts(media, clock=clk, call_id=call_id)`.
  - `transport = FakeAudioTransport(media)`; `def brain_factory(): return FakeBrain(clk, text=response_text, delay_ms=0)`. For barge-in, the response_text must be long enough that real Piper TTS is still emitting when USER_SPEECH_STARTED arrives — use a multi-word `response_text` and push the interrupt-trigger frames promptly after the assistant turn launches.
  - `ctx = StreamingCallContext(..., interruption=InterruptionParams(min_speech_ms=40, min_words=2))` (mirror the fake sim so barge-in policy fires).
  - Build `StreamingCallSession(...)`; return the `StreamSimulation` dataclass (same fields as the fake builder).
  - **N5:** add `"build_real_stream_simulation"` to the module `__all__` (simulate.py:39).
- [ ] **Step 2: write the E2E test** `test_real_e2e_simulation.py`:
  - Module-level skip guard: `pytest.mark.skipif(not (pipecat_available() and find_spec("faster_whisper") and find_spec("piper")), reason="real streaming deps not installed")` (import `pipecat_available` from `pipecat_runtime`, `find_spec` from importlib.util).
  - A `_drive(sim, wav_path)` async helper (in the TEST file — outside `streaming/**`, so real `asyncio.sleep` is allowed): start `asyncio.create_task(sim.session.run())`; `wave.open` the fixture, `readframes`, slice into 640-byte chunks, for each `await sim.transport.push_inbound(AudioFrame(pcm16=chunk, media=sim.media, timestamp_ms=seq*20, seq=seq))` then `await asyncio.sleep(0.02)`. **Then append a MANDATORY trailing-silence tail** (≈0.6 s = 30 frames of `b"\x00"*640`) so real Smart-Turn v3 (`stop_secs=0.5`) reliably emits `ENDPOINT_DETECTED` for the normal case — without it the test hangs to the timeout. Then `sim.transport.end_inbound()`; `await asyncio.wait_for(run_task, 60)`.
  - `test_real_e2e_normal_turn`: drive with `barge_in=False`; assert `summary()["outbound_audio_frames"] >= 1`; assert at least one committed record with `heard_chars > 0` and not interrupted; assert the STT transcript (read from `sim.session.records[...]` user_transcript or via a tracer hook) is non-empty. (Confirm a keyword actually present in the fixture during impl — N4; if unsure, assert non-empty rather than substring.)
  - `test_real_e2e_barge_in`: `barge_in=True`; assert `summary()["flushes"]` non-empty and a record with `interrupted is True` (BARGED_IN) and `abandoned_chars > 0`.
- [ ] **Step 3: run locally** `uv run --no-sync python -m pytest tests/gateway/streaming/test_real_e2e_simulation.py -q` → passes locally (deps installed); confirm it SKIPS when a dep is force-absent (sanity: `-p no:cacheprovider` not needed; just confirm skip logic by reading).
- [ ] **Step 4: commit** — `feat(streaming): build_real_stream_simulation + simulated E2E (real STT/TTS, stub brain)`.

---

## Task 3: Carry-in #3 — bounded inbound queue (CI-tested)

**Files:**
- Modify: `gateway/calls/native/streaming/aiortc_transport.py` (`AiortcStreamingTransport`)
- Test: `tests/gateway/streaming/test_aiortc_transport.py`

Read first: `AiortcStreamingTransport.__init__` (the `_inbound = asyncio.Queue()` line ~92), `push_inbound`, `inbound()`/`_inbound_gen`, `close()` (the `None`-sentinel enqueue). The outbound track back-pressure in `aiortc_engine.py::_create_pcm_streaming_track` (~lines 1661-1672) for the drop-oldest+warn idiom to mirror.

- [ ] **Step 1 (TDD): failing tests**:
  - `test_inbound_queue_bounded_drops_oldest`: construct transport with a small bound (add an optional `inbound_maxsize` ctor param, default e.g. 100; test passes a tiny value like 2); `push_inbound` more than the bound rapidly without draining; assert the oldest frames are dropped (queue holds ≤ bound) and a warning was logged (caplog).
  - `test_close_sentinel_never_dropped`: fill the queue to the bound, then `await close()`; assert `inbound()` still terminates (the `None` sentinel was enqueued, not dropped).
- [ ] **Step 2 → red.**
- [ ] **Step 3: implement.** Add `inbound_maxsize: int = <default>` to `__init__`; build `self._inbound = asyncio.Queue(maxsize=inbound_maxsize)`. In `push_inbound`: if full, drop-oldest (`self._inbound.get_nowait()` in a guarded try) + `logger.warning("inbound overflow; dropped oldest frame")`, then `put_nowait(frame)`. In `close()`: ensure the sentinel is enqueued unconditionally — if the queue is full, drop one oldest frame to make room, then `put_nowait(None)` (never drop the sentinel). Keep `close()` idempotent.
- [ ] **Step 4 → green.** `uv run --no-sync python -m pytest tests/gateway/streaming/test_aiortc_transport.py -q`.
- [ ] **Step 5: commit** — `fix(streaming): bound transport inbound queue (drop-oldest + watermark); never drop close sentinel`.

---

## Task 4: Carry-in #1 — resampler multi-frame drain (local; importorskip av)

**Files:**
- Modify: `gateway/calls/native/aiortc_engine.py` (`_create_pcm_streaming_track`)
- Test: `tests/gateway/test_native_streaming_track.py`

Read first: `_create_pcm_streaming_track` (the track class, `recv()` ~lines 1684-1700, `self._resampler.resample(frame)`, the pacing `_pace()`, `enqueue`, `drop_pending`).

- [ ] **Step 1 (TDD): failing test** (`importorskip("av")`, `importorskip("aiortc")`):
  - `test_recv_drains_multiple_resampler_frames`: build the track at 48000; enqueue a single **large** 16k frame — concretely several thousand samples (e.g. 4000 samples = 8000 bytes, ~250 ms), well above the 320-sample/20 ms frame size, so the 48k resample (≈12000 samples) spans more than one output AudioFrame and `av.AudioResampler.resample()` returns >1 frame. (A 320-sample input resamples to a single ~960-sample 48k frame and would NOT exercise the drain — under-sizing makes the test pass trivially.) Call `recv()` repeatedly and assert ALL resampled output frames are returned in order (none discarded) before the queue falls back to silence; assert total returned samples ≈ 3× input samples across the drained frames.
- [ ] **Step 2 → red** (second frame currently discarded).
- [ ] **Step 3: implement.** Add `self._pending: collections.deque[av.AudioFrame] = deque()` to the track. In `recv()`: if `self._pending`, popleft and return it (after pacing). Else pull the next queued input frame, `frames = self._resampler.resample(input_frame)` (a list), pts/time_base-stamp each, extend `self._pending` with all of them, then return `self._pending.popleft()`. Empty queue → silence frame as before. Preserve the persistent-resampler continuity (M1).
- [ ] **Step 4 → green (locally).** `uv run --no-sync python -m pytest tests/gateway/test_native_streaming_track.py -q`.
- [ ] **Step 5: commit** — `fix(calls): drain secondary av.AudioResampler frames in PCM streaming track (deque)`.

---

## Task 5: Carry-in #2 — inbound frame-rate threading (CI-tested)

**Files:**
- Modify: `gateway/calls/native/aiortc_engine.py` (`_DirectFeedAccumulator.accept_pcm16` + the relay call site)
- Test: `tests/gateway/test_native_streaming_track.py` (or the accumulator test file)

Read first: `_DirectFeedAccumulator` (`__init__`, `accept_pcm16(self, pcm16, *, now=None)` ~line 1737), the relay call site (`pcm16 = _audio_frame_to_pcm16(frame)` then `await self._accumulator.accept_pcm16(pcm16)` ~lines 759-761), `_audio_frame_to_pcm16` (resamples to `frame.sample_rate`).

- [ ] **Step 1 (TDD): failing tests** (CI — spy pipeline, no aiortc needed for the accumulator unit):
  - `test_accept_pcm16_threads_frame_rate`: build `_DirectFeedAccumulator(spy_pipeline, call_id, native_rate=48000)`; `await acc.accept_pcm16(b"...", sample_rate=16000)`; assert `spy_pipeline.process_pcm16` was called with `sample_rate=16000`.
  - `test_accept_pcm16_defaults_to_native_rate`: `await acc.accept_pcm16(b"...")` (no `sample_rate`); assert called with `sample_rate=48000` (back-compat for the existing positional call).
- [ ] **Step 2 → red.**
- [ ] **Step 3: implement.** Change signature to `async def accept_pcm16(self, pcm16, *, now=None, sample_rate=None)`; forward `sample_rate=sample_rate or self._native_rate` to `process_pcm16`. At the relay call site, pass `sample_rate=int(getattr(frame, "sample_rate", 0) or self.config.sample_rate)`. Keep `now` accepted (unused) for the turn-based call shape.
- [ ] **Step 4 → green.** `uv run --no-sync python -m pytest tests/gateway/test_native_streaming_track.py -q` (and any accumulator test).
- [ ] **Step 5: commit** — `fix(calls): thread inbound frame sample-rate to process_pcm16 (avoid pitch-shift on non-48k)`.

---

## Task 6: Gates

- [ ] `uv run ast-grep scan` (or the project's invocation) → no-walltime clean over `streaming/**` (the E2E drive loop lives in tests; `simulate.py` has no `asyncio.sleep`).
- [ ] `ruff check` + `ty check` clean on: `aiortc_transport.py`, `simulate.py`, `aiortc_engine.py`, and the new/modified test files.
- [ ] Package imports without extras: `uv run --no-sync python -c "import gateway.calls.native.streaming.aiortc_transport, gateway.calls.native.streaming.simulate; print('ok')"` → ok (real builders are lazy-imported inside the real branch / function).
- [ ] CI-representative run (no real deps): `uv run --no-sync python -m pytest tests/gateway/streaming/ tests/gateway/test_native_streaming_track.py -q` → wiring + queue + #2 tests pass; real E2E (#Task 2) and #1 test SKIP when deps absent (verify the skip markers fire).
- [ ] Full local streaming validation (deps installed): real E2E + #1 pass.

---

## Done criteria

`build_streaming_pipeline(cognitive="real")` wires real ports (default) or injected ports; CI wiring + brain-default tests green; bounded inbound queue (#3) + frame-rate threading (#2) tested in CI; resampler drain (#1) + real E2E (normal + barge-in) pass locally and skip in CI; gates clean; turn-based + `cognitive="fake"` default unchanged. Then PR → CI → fix obvious → merge on green (real-dep tests skip in CI by design; modulo pre-existing `test_setup*` + `osv-scan` infra reds). Sets up **Slice 8** (live iPhone, human-gated): with `cognitive="real"` + the Slice 6b live track, a real SimpleX call can flow audio both ways.
