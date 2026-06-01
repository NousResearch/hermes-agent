# SimpleX Streaming Voice — Slice 7: `cognitive="real"` + Simulated E2E (Design)

> Status: DESIGN. Architecture **A — ports own the loop**. Flips `build_streaming_pipeline(cognitive="real")` from `NotImplementedError` to a real-port wiring, adds a creds-free **simulated end-to-end** harness (recorded audio → real turn detection → real STT → deterministic brain → real TTS → outbound audio, incl. barge-in), and lands the three Slice-6/6b carry-in fixes. Builds on merged Slices 1–6b. Real-dep tests skip in CI (pipecat/faster-whisper/piper/aiortc absent there); real validation runs locally; the live iPhone call is **Slice 8** (human-gated).

## 1. Goal

1. **Wire `cognitive="real"`** in `build_streaming_pipeline` so the real cognitive ports run together in a live `StreamingPipeline`: `build_local_turn_detector` + `build_local_whisper_stt` + `build_piper_tts` + `HermesSyncBrain(build_call_agent_factory())`, all on a `MonotonicClock`. This is the exact wiring the live call (Slice 8) consumes.
2. **Simulated E2E**: drive the *real* audio reflex path end-to-end from a recorded wav, reproducibly and without LLM credentials, by injecting only a deterministic stub brain. Cover the normal turn and a barge-in variant.
3. **Carry-in fixes** from the Slice 6/6b reviews (#1 resampler multi-frame drain, #2 inbound frame-rate robustness, #3 bounded inbound queue).

## 2. Key decisions (subagent-evaluated)

- **Decision A — injected-backend seam (chosen over "all-real, sim-skips" and "separate harness").** `build_streaming_pipeline` gains `brain_factory/turn_detector/stt/tts` injectable params (plus the existing `clock`), each defaulting to its real builder when `None`. This is the same injected-backend pattern Slices 4 (`LocalWhisperSTT(transcribe=…)`) and 5 (`StreamingTTS(synthesize_pcm=…)`) established one level down — lifted to the pipeline factory. It is the only option where the E2E test drives the **actual production factory** (one port swapped) rather than a parallel wiring, and where the audio path is deterministic & creds-free (real audio ports, stub brain). The live call uses the all-real defaults.
- **Decision B — `build_real_stream_simulation` beside `build_stream_simulation` in `streaming/simulate.py`.** Same "pure synchronous construction returns a drivable `StreamSimulation`" shape; reuses the `StreamSimulation` dataclass + `summary()`. The drive loop (real-time pacing) lives in the **test/CLI layer**, never in `streaming/**`, because the `no-walltime` ast-grep rule bans `asyncio.sleep` there. Reuses the existing `tests/gateway/streaming/fixtures/turn_detection/speech_16k_mono.wav`.
- **Decision C — `MonotonicClock` for the real pipeline.** Confirmed present (`clock.py:15`), already the production default in `build_piper_tts`, and the *sanctioned* wall-clock seam (`no-walltime.yml` explicitly `ignores: clock.py`). The real branch uses an injected `clock` if given (lets a CI wiring test pass a `VirtualClock`), else `MonotonicClock()`.

## 3. Components

### 3.1 `build_streaming_pipeline` — real branch (`streaming/aiortc_transport.py`)

New signature (additive; fake branch unchanged):
```python
def build_streaming_pipeline(
    config, *, cognitive="fake", clock=None, sink=None,
    brain_factory=None, turn_detector=None, stt=None, tts=None,
) -> StreamingPipeline
```
`cognitive="real"` branch:
- `media = MediaFormat(16000, 1, 20)`; `clk = clock or MonotonicClock()`.
- Resolve each port (injected wins, else build real, lazily — each builder already raises a clear `RuntimeError` naming its extra when absent):
  - `turns = turn_detector or build_local_turn_detector(media, call_id=call_id)`
  - `stt = stt or build_local_whisper_stt(media, call_id=call_id)`
  - `tts = tts or build_piper_tts(media, clock=clk, call_id=call_id)`
  - `bf = brain_factory or (lambda: HermesSyncBrain(build_call_agent_factory()))` (the session's `brain_factory` returns a `HermesBrainPort`; `HermesSyncBrain` IS that port).
- Build `AiortcStreamingTransport(media, clock=clk, outbound_sink=sink or _noop_sink)`, `StreamingCallContext`, `StreamingCallSession(...)`, `StreamingCallTracer(call_id)` exactly as the fake branch does; return `StreamingPipeline(media=media, session=session, transport=transport, clock=clk)`.
- Remove the `NotImplementedError`. `cognitive` other than `fake`/`real` still raises `ValueError`.

**Brain-factory shape (important):** `build_call_agent_factory()` returns an `AgentFactory` (`ctx → AIAgent`), NOT a `HermesBrainPort`. The session needs `brain_factory: () -> HermesBrainPort`. So the default brain_factory is `lambda: HermesSyncBrain(build_call_agent_factory())`. The injectable `brain_factory` param is the `() -> HermesBrainPort` callable directly (the stub brain factory the sim passes).

### 3.2 Deterministic stub brain (test seam)

The sim needs a `HermesBrainPort` that yields a fixed `BrainEvent(FINAL_TEXT, text=response_text)` with no LLM/network. The existing `FakeBrain` (fakes.py) already does exactly this on a `Clock`. **Reuse `FakeBrain`** — no new type. The real-sim brain_factory is `lambda: FakeBrain(clock, text=response_text, delay_ms=0)`. (This keeps the stub a single, already-tested implementation; only the audio ports are "real" in the E2E.)

### 3.3 `build_real_stream_simulation` (`streaming/simulate.py`)

```python
def build_real_stream_simulation(
    *, call_id="real-stream-sim", contact_id="sim-contact",
    response_text="It's sunny today.", barge_in=False,
    turn_detector=None, stt=None, tts=None, clock=None,
) -> StreamSimulation
```
- `clk = clock or MonotonicClock()`; `media = _MEDIA` (16k/1/20).
- Real ports by default (injected wins, for the CI wiring test): `build_local_turn_detector(media, call_id=…)`, `build_local_whisper_stt(media, call_id=…)`, `build_piper_tts(media, clock=clk, call_id=…)`.
- **Transport:** the simulation uses `FakeAudioTransport` (it records `.sent`/`.flushes` for assertions and exposes `push_inbound`/`end_inbound`) — the same transport the fake sim uses. (The real `AiortcStreamingTransport` is exercised separately by the engine wiring tests; the E2E sim's job is the cognitive path, observed via the fake transport's recorders.)
- Brain: deterministic `FakeBrain(clk, text=response_text)`.
- For `barge_in=True`: real turn detection on a single fixed wav cannot deterministically produce a second mid-TTS onset, AND real whisper STT can't deterministically finalize a transcript on a scripted-endpoint schedule (an empty buffer → empty transcript → no assistant turn → nothing to barge into). So the **barge-in variant injects BOTH a scripted `FakeTurnDetection`** (ENDPOINT → USER_SPEECH_STARTED → USER_SPEECH_STOPPED) **and a deterministic `FakeSTT`** (preconfigured final + a `"hold on"` partial to satisfy the `min_words` policy), while keeping **real TTS**. What barge-in validates is the real-TTS flush/abandon path under interruption — transcript content is irrelevant to that, and the **normal-turn test already covers real STT end-to-end**. Documented as an explicit, intentional narrowing.
- Returns the existing `StreamSimulation` dataclass; `summary()` reused unchanged. **No `asyncio.sleep` in this module** (no-walltime); the caller paces.

### 3.4 Drive loop (test layer — outside `streaming/**`)

A test helper (in `tests/gateway/streaming/`) chunks the wav exactly as the existing real-port tests do:
`wave.open` → `readframes` → slice into 640-byte (20 ms @ 16k) chunks → `AudioFrame(pcm16=chunk, media, timestamp_ms=seq*20, seq=seq)` → `await transport.push_inbound(frame)`, pacing with `await asyncio.sleep(0.02)` between frames (real time, because real Silero VAD / Smart-Turn / whisper run on `MonotonicClock`). Then `transport.end_inbound()`; `await` the `session.run()` task; read `simulation.summary()`.

### 3.5 Carry-in fixes (`aiortc_engine.py` + `streaming/aiortc_transport.py`)

- **#1 — resampler multi-frame drain** (`aiortc_engine.py::_create_pcm_streaming_track.recv`): `av.AudioResampler.resample()` can return >1 output frame for one input (reachable now that real Piper TTS emits variable-size chunks). Today only the first non-empty frame is returned and the rest are discarded. Fix: hold a `collections.deque` of pending resampled frames on the track; `recv()` drains the deque first, only pulling+resampling a new input frame when the deque is empty. Add a multi-frame-output test (importorskip av).
- **#2 — inbound frame-rate robustness** (`_DirectFeedAccumulator.accept_pcm16`): today passes a hardcoded `self._native_rate` (config 48k) to `process_pcm16`. A non-48k remote frame would pitch-shift (process_pcm16 resamples *from* the passed rate). The relay decodes the raw `av.AudioFrame` (in scope at the `accept_pcm16` call site) and `_audio_frame_to_pcm16` resamples to *that frame's own rate*, so the bytes are at `frame.sample_rate` while the accumulator forwards 48k — a real mismatch. **Plumbing:** add a `sample_rate: int | None = None` kwarg to `accept_pcm16` (keep the existing `*, now=None` for back-compat with the turn-based `AudioUtteranceAccumulator` call shape; keep `sample_rate` optional so the existing positional `accept_pcm16(pcm16)` test call still works); the relay passes `int(getattr(frame, "sample_rate", 0) or native_rate)`; `accept_pcm16` forwards `sample_rate or self._native_rate` to `process_pcm16`. Add a test that a 16k inbound frame is fed at 16k (no resample) and a 48k frame at 48k.
- **#3 — bounded inbound queue** (`AiortcStreamingTransport.__init__`): the inbound `asyncio.Queue()` is unbounded. Under live load with a slow STT, this could grow without limit. Fix: bound it (`maxsize`, e.g. matching the outbound `_PCM_STREAMING_QUEUE_MAXSIZE` magnitude) with drop-oldest + a `logger.warning` watermark, mirroring the outbound track's back-pressure. The `None` close-sentinel path must still always enqueue (close must not be dropped). Add a test for overflow drop + that close still terminates `inbound()`.

## 4. CI vs local coverage

- **Runs in CI (no real deps):**
  - `build_streaming_pipeline(cognitive="real", turn_detector=fake, stt=fake, tts=fake, brain_factory=fake, clock=VirtualClock())` returns a wired `StreamingPipeline` (session has the injected ports; `is_streaming True`).
  - The `brain_factory=None` default resolves to a `HermesSyncBrain`-producing factory **without invoking it** (assert type/identity; no LLM call).
  - Carry-in #2 (frame-rate) and #3 (bounded queue) unit tests — pure asyncio, no aiortc.
- **Skips in CI / runs locally (`skipif` on `pipecat_available()` + `find_spec("faster_whisper")` + `find_spec("piper")`):**
  - The full real E2E (normal + barge-in) via `build_real_stream_simulation` + the wav driver.
  - Carry-in #1 (resampler drain) test (`importorskip("av")`).

## 5. Files

- **Modify:** `gateway/calls/native/streaming/aiortc_transport.py` — real branch of `build_streaming_pipeline` (injectable ports), bounded inbound queue (#3).
- **Modify:** `gateway/calls/native/streaming/simulate.py` — `build_real_stream_simulation`.
- **Modify:** `gateway/calls/native/aiortc_engine.py` — `_create_pcm_streaming_track.recv` deque drain (#1); `_DirectFeedAccumulator` frame-rate threading (#2).
- **Create:** `tests/gateway/streaming/test_real_e2e_simulation.py` — skipif real-deps; normal + barge-in E2E; the wav driver helper.
- **Extend:** `tests/gateway/streaming/test_aiortc_transport.py` — cognitive="real" wiring (injected fakes, CI); brain default resolution; bounded inbound queue (#3). **Remove/replace** the existing `test_build_streaming_pipeline_real_not_implemented` (~lines 420-422) which asserts `cognitive="real"` raises `NotImplementedError` — it will fail once the real branch lands; replace it with the new wiring assertion.
- **Extend:** `tests/gateway/test_native_streaming_track.py` — resampler multi-frame drain (#1, importorskip av).
- **Extend:** `tests/gateway/test_native_aiortc_engine.py` (or the accumulator test) — frame-rate threading (#2).

## 6. Acceptance criteria

- `build_streaming_pipeline(cognitive="real")` returns a `StreamingPipeline` wired with the real ports by default and accepts injected ports; CI wiring test (injected fakes) green; default brain_factory resolves to `HermesSyncBrain` without invocation.
- Local real E2E: normal turn → STT transcript non-empty (keyword-substring of the spoken fixture, not exact) + ≥1 outbound TTS audio frame + an un-interrupted committed record with non-empty heard text. Barge-in variant → `flushes` non-empty + a `BARGED_IN`/`interrupted` record with non-empty abandoned text.
- Carry-in #1/#2/#3 fixed with tests.
- `ast-grep` no-walltime clean over `streaming/**` (drive loop lives in tests); ruff + ty clean; package imports without aiortc/pipecat/whisper/piper.
- Turn-based path + `cognitive="fake"` default unchanged.

## 7. Out of scope

- The live iPhone call — **Slice 8** (human-gated).
- Cloud STT/TTS.
- Streaming/partial brain tokens (the brain yields a single FINAL_TEXT today; incremental token streaming is a later enhancement).
