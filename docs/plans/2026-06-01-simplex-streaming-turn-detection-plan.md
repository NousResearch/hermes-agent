# Slice 3: Real Local Turn Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD throughout. Design: `docs/plans/2026-06-01-simplex-streaming-turn-detection-design.md`.

**Goal:** `LocalTurnDetector` (a real `TurnDetectionPort`) backed by Pipecat's bundled Silero VAD + Smart Turn v3, behind the port with contract tests. No engine/config wiring; `FakeTurnDetection` stays default.

**Validated API facts (rely on these — confirmed against installed pipecat 1.3.0):**
- `from pipecat.audio.vad.silero import SileroVADAnalyzer`; `from pipecat.audio.vad.vad_analyzer import VADState, VADParams`.
- `from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3`; `from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams`; `from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState`.
- Construct, then **`set_sample_rate(16000)` on BOTH** (VAD needs it to init internal byte counts).
- `await vad.analyze_audio(window_bytes) -> VADState` (**async**). `vad.voice_confidence(window_bytes) -> float` (sync). `vad.num_frames_required()` → 256 (so **512-byte** PCM16 windows).
- `smart_turn.append_audio(window_bytes, is_speech: bool) -> EndOfTurnState` (**sync**). `smart_turn.clear()` (sync). `EndOfTurnState.COMPLETE`/`INCOMPLETE`.
- `SmartTurnParams(stop_secs=0.5, pre_speech_ms=200, max_duration_secs=8)` makes endpoint fire promptly after trailing silence.
- Our `AudioFrame`: 20ms@16k = 320 samples = 640 bytes → rolling re-chunk to 512-byte windows.

---

## Task 1: Commit the speech fixture

**Files:** Create `tests/gateway/streaming/fixtures/turn_detection/speech_16k_mono.wav`

- [ ] **Step 1:** Generate on macOS (this host has `say`+`afconvert`):
  ```bash
  mkdir -p tests/gateway/streaming/fixtures/turn_detection
  say -o /tmp/_st.aiff "Hello Hermes, what is the weather forecast for today?"
  afconvert /tmp/_st.aiff -f WAVE -d LEI16@16000 -c 1 tests/gateway/streaming/fixtures/turn_detection/speech_16k_mono.wav
  ```
- [ ] **Step 2:** Verify it's 16k mono 16-bit and triggers the VAD cycle (sanity, not committed test):
  ```bash
  uv run --no-sync python -c "import wave; w=wave.open('tests/gateway/streaming/fixtures/turn_detection/speech_16k_mono.wav'); print(w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes())"
  ```
  Expected: `16000 1 2 <~46000>`.
- [ ] **Step 3:** Commit.
  ```bash
  git add tests/gateway/streaming/fixtures/turn_detection/speech_16k_mono.wav
  git commit -m "test(streaming): commit say-generated 16k speech fixture for turn detection"
  ```

---

## Task 2: `LocalTurnDetector` + factory (TDD, mocked analyzers)

**Files:** Create `gateway/calls/native/streaming/local_turn_detection.py`; Create `tests/gateway/streaming/test_local_turn_detection.py`.

Read first: `gateway/calls/native/streaming/ports.py` (TurnDetectionPort, TurnEvent, TurnEventKind, AudioFrame, MediaFormat), `gateway/calls/native/streaming/pipecat_runtime.py` (`pipecat_available`).

- [ ] **Step 1: Write failing mocked tests** (Scenarios 1–5, 7). Use `unittest.mock.AsyncMock` for the VAD (`analyze_audio` is async; `voice_confidence` sync via `Mock`) and plain `Mock` for SmartTurn (`append_audio` sync). The detector must accept **injected** analyzers for testing (factory wraps real ones). Sketch:
  ```python
  import pytest
  from unittest.mock import AsyncMock, Mock
  from gateway.calls.native.streaming.local_turn_detection import LocalTurnDetector, build_local_turn_detector
  from gateway.calls.native.streaming.types import AudioFrame, MediaFormat, TurnEventKind

  M16 = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)
  def frame(seq, ms, nbytes=640): return AudioFrame(pcm16=b"\x01\x02"*(nbytes//2), media=M16, timestamp_ms=ms, seq=seq)

  def make(vad_states, eot=None, conf=0.9):
      # vad_states: list[VADState-like] returned per 512-byte window
      from pipecat.audio.vad.vad_analyzer import VADState
      vad = Mock(); vad.num_frames_required = Mock(return_value=256)
      seq = iter(vad_states)
      vad.analyze_audio = AsyncMock(side_effect=lambda buf: next(seq))
      vad.voice_confidence = Mock(return_value=conf)
      st = Mock()
      from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
      st.append_audio = Mock(return_value=(eot or EndOfTurnState.INCOMPLETE))
      st.clear = Mock()
      return LocalTurnDetector(media=M16, vad=vad, smart_turn=st), vad, st
  ```
  Tests: (1) feeding two 640-byte frames passes only 512-byte windows to `vad.analyze_audio`/`st.append_audio` and rolls remainder (assert call arg lengths == 512; after 2 frames=1280 bytes → 2 windows, 256 bytes remain). (2) scripted states QUIET,STARTING,SPEAKING,SPEAKING,STOPPING,QUIET → exactly one STARTED (at the →SPEAKING window's frame ms) and one STOPPED; `vad_confidence` on STARTED == conf. (3) `st.append_audio` returns COMPLETE on a window → ENDPOINT_DETECTED emitted; `endpoint_confidence == 0.0`. (4) a frame with `sample_rate=8000` (or channels=2) → `observe()` raises `ValueError` mentioning 16k mono. (5) after `reset()`: buffer empty, `st.clear()` called, and a subsequent →SPEAKING edge re-emits STARTED. (7) `build_local_turn_detector` with `pipecat_available()` monkeypatched False → raises clear "install simplex-streaming" error.
- [ ] **Step 2: Run → red.** `uv run --no-sync python -m pytest tests/gateway/streaming/test_local_turn_detection.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** `local_turn_detection.py`:
  - **No top-level `import pipecat`** (keep the package importable without the extra). The factory does the lazy imports.
  - `class LocalTurnDetector:` with `__init__(self, *, media: MediaFormat, vad, smart_turn)` storing analyzers, a `bytearray` buffer, `_prev_state=None`, `_window_bytes = 256*2 = 512`, and the call_id/seq context (call_id from where? use `media`/context — TurnEvent needs call_id; accept `call_id: str = ""` param or pull from a StreamingCallContext; simplest: `call_id` constructor arg defaulting to "" and set by factory).
    - Wait: `TurnEvent` needs `call_id`. Add `call_id: str = ""` to `__init__`.
  - `async def observe(self, frame) -> tuple[TurnEvent, ...]`:
    - Guard: `if frame.media.sample_rate != 16000 or frame.media.channels != 1: raise ValueError("LocalTurnDetector requires 16kHz mono audio; got ...")`.
    - Append `frame.pcm16` to buffer. `events = []`. While `len(buffer) >= 512`: pop `window = bytes(buffer[:512]); del buffer[:512]`.
      - `state = await self.vad.analyze_audio(window)`.
      - `is_speech = state in (VADState.STARTING, VADState.SPEAKING)`.
      - STARTED edge: `if self._prev_state in (None, VADState.QUIET, VADState.STARTING) and state == VADState.SPEAKING:` append `TurnEvent(call_id, USER_SPEECH_STARTED, at_ms=frame.timestamp_ms, vad_confidence=float(self.vad.voice_confidence(window)), source="silero+smartturn-v3")`.
      - STOPPED edge: `elif self._prev_state == VADState.SPEAKING and state in (VADState.STOPPING, VADState.QUIET):` append `TurnEvent(..., USER_SPEECH_STOPPED, at_ms=frame.timestamp_ms, source=...)`.
      - `self._prev_state = state`.
      - `eot = self.smart_turn.append_audio(window, is_speech)` (sync). `if eot == EndOfTurnState.COMPLETE:` append `TurnEvent(..., ENDPOINT_DETECTED, at_ms=frame.timestamp_ms, source=...)` (endpoint_confidence left 0.0).
    - `return tuple(events)`.
  - `def reset(self) -> None:` clear buffer, `self._prev_state=None`, `self.smart_turn.clear()`.
  - `def build_local_turn_detector(media: MediaFormat, *, call_id: str = "", vad_params=None, turn_params=None) -> LocalTurnDetector:`
    - `from gateway.calls.native.streaming.pipecat_runtime import pipecat_available`
    - `if not pipecat_available(): raise RuntimeError("LocalTurnDetector requires the optional Pipecat dependency. Install: pip install 'hermes-agent[simplex-streaming]'")`
    - lazy import the pipecat classes; `vad = SileroVADAnalyzer(sample_rate=16000, params=vad_params); vad.set_sample_rate(16000)`; `st = LocalSmartTurnAnalyzerV3(params=turn_params or SmartTurnParams(stop_secs=0.5, pre_speech_ms=200, max_duration_secs=8)); st.set_sample_rate(16000)`; `return LocalTurnDetector(media=media, vad=vad, smart_turn=st, call_id=call_id)`.
  - Import `VADState`/`EndOfTurnState` **inside** functions or guard at module top? The detector body references `VADState`/`EndOfTurnState`. To keep the module importable without pipecat, import them lazily inside `observe` (cache on first use) OR accept that `LocalTurnDetector` is only ever constructed when pipecat is present and import them at the top of `observe`. Simplest & safe: a module-level `try: from ... import VADState, EndOfTurnState except Exception: VADState = EndOfTurnState = None`, and in tests the mocks return the real enums (tests import pipecat). Document this.
- [ ] **Step 4: Run → green.** Same pytest command. All mocked scenarios pass.
- [ ] **Step 5: Commit.**
  ```bash
  git add gateway/calls/native/streaming/local_turn_detection.py tests/gateway/streaming/test_local_turn_detection.py
  git commit -m "feat(streaming): LocalTurnDetector over Silero VAD + Smart Turn v3"
  ```

---

## Task 3: Real-onnx contract test (Scenario 6)

**Files:** extend `tests/gateway/streaming/test_local_turn_detection.py`.

- [ ] **Step 1:** Add, guarded by `@pytest.mark.skipif(not pipecat_available(), reason="simplex-streaming extra not installed")` (NOT `@pytest.mark.integration` — CI filters that out):
  ```python
  import wave
  from pathlib import Path
  FIX = Path(__file__).parent / "fixtures" / "turn_detection" / "speech_16k_mono.wav"

  @pytest.mark.skipif(not pipecat_available(), reason="simplex-streaming extra not installed")
  async def test_real_onnx_contract_emits_started_stopped_endpoint():
      w = wave.open(str(FIX), "rb"); pcm = w.readframes(w.getnframes()); w.close()
      pcm += b"\x00" * (16000 * 2)  # 1s trailing silence to trigger endpoint
      det = build_local_turn_detector(M16)
      kinds = []
      seq = 0
      for off in range(0, len(pcm) - 640, 640):  # 20ms frames
          evs = await det.observe(frame(seq, seq * 20, 640))  # use real bytes:
          ...
  ```
  Important: build frames from the REAL fixture bytes (slice `pcm[off:off+640]`), not synthetic. Assert the ordered kinds contain `USER_SPEECH_STARTED` before `USER_SPEECH_STOPPED` before at least one `ENDPOINT_DETECTED` (assert ordering/containment; do NOT assert counts or confidences — onnx output varies by CPU). Mark the test `async` (pytest-asyncio is configured).
- [ ] **Step 2: Run.** `uv run --no-sync python -m pytest tests/gateway/streaming/test_local_turn_detection.py -q` → passes locally (pipecat installed).
- [ ] **Step 3: Commit.**
  ```bash
  git add tests/gateway/streaming/test_local_turn_detection.py
  git commit -m "test(streaming): real-onnx turn-detection contract on speech fixture"
  ```

---

## Task 4: Gates

- [ ] **Step 1:** `ast-grep scan --config sgconfig.yml gateway/calls/native/streaming/local_turn_detection.py` → no no-walltime violations (uses only `frame.timestamp_ms`).
- [ ] **Step 2:** `uv run --no-sync ruff check gateway/calls/native/streaming/local_turn_detection.py tests/gateway/streaming/test_local_turn_detection.py` and `uv run --no-sync ty check gateway/calls/native/streaming/local_turn_detection.py` → clean.
- [ ] **Step 3:** Confirm package still imports without the extra would-be-absent: `uv run --no-sync python -c "import gateway.calls.native.streaming; print('ok')"` → ok (and the module body must not `import pipecat` at top).
- [ ] **Step 4:** Full streaming suite: `uv run --no-sync python -m pytest tests/gateway/streaming/ -q` → all pass.

---

## Done criteria
All 7 BDD scenarios pass; package imports without pipecat; gates clean; fakes remain default. Then finish-branch: PR → CI → fix obvious issues → merge on green (modulo documented pre-existing `test_setup*` + `osv-scan`).
