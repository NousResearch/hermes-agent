# Slice 6: aiortc AudioTransportPort Bridge — Core (Implementation Plan)

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD. Design: `docs/plans/2026-06-01-simplex-streaming-transport-design.md`.
> Scope (post-review split): the **CI-verifiable pure-asyncio core only**. Live aiortc track + accumulator bypass = Slice 6b.

**Goal:** `AiortcStreamingTransport(AudioTransportPort)` (no aiortc import; injected outbound sink) + `StreamingPipeline` exposing `process_pcm16` (lazy session task, non-blocking push, `aclose()`) + engine STREAMING branch constructs it (un-defers Slice 2) with a `cognitive="fake"|"real"` selector (default `fake`). Turn-based path + `Fake*` defaults unchanged.

**Validated facts:**
- `AudioTransportPort`: `media` prop, `async push_inbound`, `inbound()->AsyncIterator`, `async emit_outbound`, `async flush_outbound(reason)->FlushResult`, `async close()`. `FlushResult(dropped_frames, dropped_ms, last_sent_mark=None)`. `FakeAudioTransport` uses `asyncio.Queue`+`None` sentinel; flush returns `last_sent_mark=None`.
- `StreamingCallSession.run()` drains `inbound()` to the `None` sentinel; does NOT read `transport.media`; emits via `emit_outbound`; barge-in → `flush_outbound`. `close()`-sentinel → `run()` returns (cancel-safe). No deadlock starting `run()` as a task before the first `push_inbound`.
- Engine `build_native_pipeline(config, *, turn_based_factory)` (Slice 2): streaming branch raises `StreamingExtraNotInstalled` (pipecat absent) else `PipecatIntegrationDeferred`. We replace the deferral with `StreamingPipeline` construction, aiortc-guarded.
- SimpleX media is 48kHz; ports require 16k mono → `process_pcm16` resamples via stdlib `audioop.ratecv` (stdlib on CI py3.11; lazy import). `audioop` lives only inside `process_pcm16`.
- ast-grep no-walltime scopes `streaming/**`: use injected `Clock.now_ms()`, no `time.*`/`asyncio.sleep`. aiortc NOT installed in CI (`[all]` excludes `simplex-native-calls`).

---

## Task 1: `AiortcStreamingTransport` + `StreamingPipeline` (TDD, pure asyncio)

**Files:** Create `gateway/calls/native/streaming/aiortc_transport.py`; Create `tests/gateway/streaming/test_aiortc_transport.py`.

Read first: ports.py (AudioTransportPort/FlushResult), types.py (AudioFrame/MediaFormat/StreamingCallContext + its required fields), fakes.py (FakeAudioTransport inbound/flush + the Fake* cognitive ports + their ctors), session.py (StreamingCallSession ctor — what ports/args it needs), clock.py (Clock/VirtualClock/MonotonicClock), cancellation.py.

- [ ] **Step 1: failing tests** (scenarios 1–6 + 7a + 7b):
  - Build a `MediaFormat(16000,1,20)`; a list-appending async `sink` (`async def sink(f): sent.append(f)`) optionally with a `drop` attribute/hook.
  1. `push_inbound(f)`→`inbound()` yields f; after `close()`, `inbound()` ends.
  2. `emit_outbound(f)` awaits sink(f) and records pending.
  3. after N emits, `flush_outbound("barge")` → pending cleared, sink.drop called (if present), returns `FlushResult(dropped_frames=N, dropped_ms=20*N, last_sent_mark=None)`.
  4. `StreamingPipeline` with a spy transport (or real transport + recording sink): `await pipe.process_pcm16(call_id="c", pcm16=<48k bytes>, sample_rate=48000)` → pushes one AudioFrame with `media.sample_rate==16000`, monotonic seq, timestamp from injected VirtualClock; returns promptly (ack truthy). Resample reduces byte count ~3x.
  5. full fake-port seam: build `StreamingPipeline` with Fake cognitive ports (FakeTurnDetection scripted to endpoint, FakeBrain text, FakeTTS on a VirtualClock) + recording sink; feed inbound frames via `process_pcm16`, advance the clock, `await pipe.aclose()`; assert outbound AudioFrames reached the sink and the session task completed.
  6. barge-in: drive the fake session so a turn event triggers `flush_outbound`; assert sink.drop fired and a FlushResult with dropped frames recorded (reuse the session's existing barge-in scenario shape from test_session_scenarios.py).
  7a. **aiortc-absent raise (always CI):** `monkeypatch` the engine's `_aiortc_available` → False; `build_native_pipeline(streaming-config, cognitive="fake", turn_based_factory=...)` raises a clear error (match a string like "aiortc"/"simplex-native-calls"). Do NOT import StreamingPipeline in a way that pulls aiortc (it doesn't, but assert via build_native_pipeline + the raise).
  7b. **aiortc-present:** `@pytest.mark.skipif(not _aiortc_available())` → returns a `StreamingPipeline` with `process_pcm16` + `is_streaming True`; turn_based default returns the factory product.
- [ ] **Step 2: red.**
- [ ] **Step 3: implement `aiortc_transport.py`** (NO top-level aiortc/av import):
  - `class AiortcStreamingTransport:` `__init__(self, media, *, clock, outbound_sink)`; `_inbound: asyncio.Queue`; `_pending: list[AudioFrame]`; `_closed=False`.
    - `@property media`; `async push_inbound(f): await self._inbound.put(f)`; `async def inbound(self): while True: f=await self._inbound.get(); if f is None: return; yield f`; `async close(self): if not self._closed: self._closed=True; await self._inbound.put(None)`.
    - `async emit_outbound(self,f): self._pending.append(f); await self._outbound_sink(f)`.
    - `async flush_outbound(self, reason): n=len(self._pending); ms=sum(f.duration_ms for f in self._pending); self._pending.clear(); drop=getattr(self._outbound_sink,"drop",None); if drop: await drop() (or drop());  return FlushResult(dropped_frames=n, dropped_ms=ms, last_sent_mark=None)`.
  - `class StreamingPipeline:` `is_streaming=True`. `__init__(self, *, media, session, transport, clock)` (or build session+transport internally from injected ports/clock — keep the ctor testable: accept the session + transport so tests can inject fakes; provide a `build_*` helper for the engine). `process_pcm16(self, *, call_id, pcm16, sample_rate)`: lazy-start `self._task = asyncio.create_task(self._session.run())`; `import audioop` lazily; `pcm16_16k = audioop.ratecv(pcm16, 2, 1, sample_rate, 16000, None)[0] if sample_rate!=16000 else pcm16`; `await self._transport.push_inbound(AudioFrame(pcm16_16k, media=self._media_16k, timestamp_ms=self._clock.now_ms(), seq=self._seq)); self._seq+=1`; return a simple ack (e.g. `SimpleNamespace(ok=True)` or a tiny dataclass). `async def aclose(self): await self._transport.close(); if self._task: await self._task`.
  - A factory the engine uses: `build_streaming_pipeline(config, *, cognitive="fake", clock=None) -> StreamingPipeline` — build a recording/buffer outbound sink (Slice 6 default), the transport, the Fake cognitive ports (cognitive=="fake"), the `StreamingCallSession`, wire them. (cognitive=="real" raises NotImplementedError for now — Slice 7.)
- [ ] **Step 4: implement engine change** in `engine.py`: add `def _aiortc_available() -> bool` (find_spec("aiortc")); add `cognitive: str = "fake"` param to `build_native_pipeline`; in the STREAMING branch: if not pipecat-needed... — actually per design D4: if `not _aiortc_available(): raise StreamingExtraNotInstalled("...needs aiortc; install hermes-agent[simplex-native-calls]")`; else `return build_streaming_pipeline(config, cognitive=cognitive)`. Remove the `build_pipeline`/`PipecatIntegrationDeferred` call from the streaming happy path (keep the import only if still needed). Update docstring.
- [ ] **Step 5: green.** Run `uv run --no-sync python -m pytest tests/gateway/streaming/test_aiortc_transport.py tests/gateway/streaming/test_engine_selection.py -q`.
- [ ] **Step 6: commit** — `feat(streaming): aiortc transport core + StreamingPipeline (process_pcm16); engine un-defers STREAMING`.

---

## Task 2: Gates
- [ ] ast-grep no-walltime over `aiortc_transport.py` → clean (injected Clock; no time/sleep).
- [ ] `ruff check` + `ty check` on aiortc_transport.py + engine.py → clean.
- [ ] Package imports without aiortc: `uv run --no-sync python -c "import gateway.calls.native.streaming; import gateway.calls.native.streaming.aiortc_transport; import sys; print('aiortc' not in sys.modules)"` → True (audioop/aiortc only touched inside process_pcm16; aiortc never imported by the core).
- [ ] Full streaming suite green: `uv run --no-sync python -m pytest tests/gateway/streaming/ -q`.

---

## Done criteria
Engine STREAMING constructs a `StreamingPipeline` (no deferral) with fake ports; transport + process_pcm16 media seam proven in CI (scenarios 1–6 + 7a); 7b skipif; package imports without aiortc; gates clean; turn-based + Fake* defaults unchanged. Then PR → CI → fix obvious → merge on green (modulo pre-existing `test_setup*` + `osv-scan`). Slice 6b (live track) follows adjacent to Slice 8.
