# SimpleX Streaming Voice — Reflex Foundation (Slice 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a streaming, full-duplex SimpleX voice reflex foundation where Hermes stays the reasoning brain and Pipecat supplies human reflexes (turn detection, barge-in, interruption), proven entirely in deterministic simulation (scenarios A–G) before any live device test.

**Architecture:** A new sibling slice `gateway/calls/native/streaming/` beside the untouched turn-based path. Hexagonal seams: frozen value types + `Protocol` ports (the contract), pure correctness components (`InterruptionPolicy`, `HeardSpanLedger`, `CancellationScope`), a transport-agnostic `StreamingCallSession` reflex core driven by fakes in tests and delegated-into by thin Pipecat adapters in production. A `calls.native.engine` flag selects `turn_based` (default) vs `streaming`.

**Tech Stack:** Python 3.13/3.14, `dataclass(frozen=True)` + `typing.Protocol`, `asyncio`, pytest (+pytest-asyncio), `ruff`, `ty`, `ast-grep`, `kit` (cased-kit). Pipecat for production frame plumbing (smoke-tested only, gated on wheel availability per Decision A). Brain = `AIAgent.run_conversation` via `asyncio.to_thread` + cooperative `AIAgent.interrupt()`.

**Spec:** `docs/plans/2026-05-31-simplex-streaming-voice-reflexes-design.md`

---

## File Structure

New slice `gateway/calls/native/streaming/`:

| File | Responsibility |
|---|---|
| `__init__.py` | Public exports for the slice |
| `clock.py` | `Clock` Protocol, `MonotonicClock` (prod), `VirtualClock` (tests). ONLY place `time.*` is allowed. |
| `types.py` | All frozen value types + enums (the data contract) |
| `ports.py` | All `Protocol` ports + `CallTurnCancelled` exception |
| `cancellation.py` | `CancellationScope` (cooperative) |
| `interruption.py` | `InterruptionPolicy` (pure `decide`) |
| `ledger.py` | `HeardSpanLedger` (playback-aware truncation) |
| `brain.py` | `HermesSyncBrain` (wraps `run_conversation`, deferred persist, cooperative interrupt) |
| `tracer.py` | `StreamingCallTracer` (extends `NativeCallTraceWriter`) |
| `session.py` | `StreamingCallSession` (reflex loop) |
| `fakes.py` | `FakeAudioTransport`, `FakeTurnDetection`, `FakeSTT`, `FakeTTS`, `FakeBrain` |
| `pipecat_transport.py` | Production Pipecat transport + `LLMService` adapter (WP10, conditional) |
| `engine.py` | `select_call_engine(config)` → `turn_based`/`streaming` |

Tests under `tests/gateway/streaming/`:

| File | Covers |
|---|---|
| `test_clock.py` | VirtualClock determinism |
| `test_types.py` | type invariants (frozen, derived props) |
| `test_cancellation.py` | CancellationScope semantics |
| `test_interruption_policy.py` | tier-1 decide() truth tables |
| `test_ledger.py` | heard/abandoned math |
| `test_brain.py` | HermesSyncBrain adapter (mocked AIAgent) |
| `test_tracer.py` | redaction + playhead events |
| `test_fakes.py` | fakes honor their contracts |
| `test_session_scenarios.py` | acceptance A–G |
| `test_engine_selection.py` | Scenario F |
| `test_pipecat_smoke.py` | WP10 (conditional skip if wheels absent) |

Tooling config (repo root): `sgconfig.yml` + `.ast-grep/rules/*.yml`.

---

## Conventions for every task

- **TDD:** write the failing test, run it red, implement minimal, run green, refactor, then run the gate, then commit.
- **Per-task gate (run before each commit):**
  ```bash
  uv run ruff check gateway/calls/native/streaming tests/gateway/streaming
  uv run ty check gateway/calls/native/streaming
  ast-grep scan gateway/calls/native/streaming
  uv run pytest tests/gateway/streaming -q
  ```
  (`kit` alignment check runs at WP12; `ast-grep` becomes meaningful once `sgconfig.yml` exists after WP0.)
- **Async tests:** module-level `pytestmark = pytest.mark.asyncio` where needed (repo pins `pytest-asyncio==1.3.0`).
- **Branch:** all work on `feat/simplex-streaming-voice-reflexes` (already created).

---

## Task 0 (WP0): Tooling + slice scaffold

**Files:**
- Create: `gateway/calls/native/streaming/__init__.py` (empty stub)
- Create: `tests/gateway/streaming/__init__.py` (empty)
- Create: `sgconfig.yml`, `.ast-grep/rules/streaming-purity.yml`, `.ast-grep/rules/no-walltime.yml`, `.ast-grep/rules/frozen-dataclass.yml`
- Modify: none of the existing slice yet

- [ ] **Step 1: Install ast-grep**

Run: `brew install ast-grep` then `ast-grep --version`
Expected: prints a version (e.g. `ast-grep 0.x`).

- [ ] **Step 2: Install & verify kit**

Run: `uv tool install cased-kit && kit --version`
Expected: a version prints.
**If it fails or the CLI is not `kit`:** record the actual package/command in this plan, and per Decision A/§8.1 fallback, mark the WP12 `kit` gate as degraded to `ty` + manual review. Do NOT block the DAG.

- [ ] **Step 3: Verify Pipecat wheel reality on this interpreter (informs WP10)**

Run: `uv pip install --dry-run pipecat-ai onnxruntime aiortc 2>&1 | tail -20`
Record the outcome in this plan. If `onnxruntime`/`pipecat-ai` have no wheel for the active Python, WP10's Pipecat smoke is **skipped** (Decision A) and `test_pipecat_smoke.py` uses `pytest.importorskip("pipecat")`.

- [ ] **Step 4: Create the slice + test packages**

Create `gateway/calls/native/streaming/__init__.py` with a docstring only:
```python
"""Streaming SimpleX voice reflex foundation (Slice 1).

Sibling to the turn-based native call path. Hermes stays the brain; Pipecat
supplies reflexes. See docs/plans/2026-05-31-simplex-streaming-voice-reflexes-design.md.
"""
```
Create empty `tests/gateway/streaming/__init__.py`.

- [ ] **Step 5: Write ast-grep architecture rules**

`sgconfig.yml`:
```yaml
ruleDirs:
  - .ast-grep/rules
```

`.ast-grep/rules/no-walltime.yml`:
```yaml
id: no-walltime-in-streaming
language: python
severity: error
message: Use the injected Clock, not wall-clock. Only clock.py may call time.*.
rule:
  any:
    - pattern: time.time()
    - pattern: time.sleep($$$)
    - pattern: asyncio.sleep($$$)
files:
  - "gateway/calls/native/streaming/**"
ignores:
  - "gateway/calls/native/streaming/clock.py"
```

`.ast-grep/rules/streaming-purity.yml`:
```yaml
id: ports-types-stay-pure
language: python
severity: error
message: types.py/ports.py must not import run_agent or heavy deps (keep the contract pure).
rule:
  any:
    - pattern: import run_agent
    - pattern: from run_agent import $$$
files:
  - "gateway/calls/native/streaming/types.py"
  - "gateway/calls/native/streaming/ports.py"
```

`.ast-grep/rules/frozen-dataclass.yml`:
```yaml
id: frozen-dataclass-in-contract
language: python
severity: error
message: dataclasses in types.py must be @dataclass(frozen=True).
rule:
  pattern: |
    @dataclass
    class $NAME:
      $$$
files:
  - "gateway/calls/native/streaming/types.py"
```

- [ ] **Step 6: Verify ast-grep runs clean on the empty slice**

Run: `ast-grep scan gateway/calls/native/streaming`
Expected: no errors (no violations yet).

- [ ] **Step 7: Commit**

```bash
git add gateway/calls/native/streaming tests/gateway/streaming sgconfig.yml .ast-grep
git commit -m "chore(streaming): scaffold slice + ast-grep architecture rules + tooling (WP0)"
```

---

## Task 1 (WP1): The contract — clock, types, ports (reviewed & locked)

This is the foundation everything imports. After it lands, request review before parallel WPs begin.

**Files:**
- Create: `gateway/calls/native/streaming/clock.py`
- Create: `gateway/calls/native/streaming/types.py`
- Create: `gateway/calls/native/streaming/ports.py`
- Test: `tests/gateway/streaming/test_clock.py`, `test_types.py`

- [ ] **Step 1: Write failing clock test**

`tests/gateway/streaming/test_clock.py`:
```python
import pytest
from gateway.calls.native.streaming.clock import VirtualClock

pytestmark = pytest.mark.asyncio


async def test_virtualclock_starts_at_zero_and_advances():
    clock = VirtualClock()
    assert clock.now_ms() == 0
    await clock.advance(500)
    assert clock.now_ms() == 500


async def test_virtualclock_sleep_resolves_only_after_advance():
    clock = VirtualClock()
    woke = []
    async def sleeper():
        await clock.sleep(300)
        woke.append(clock.now_ms())
    import asyncio
    task = asyncio.create_task(sleeper())
    await clock.advance(299)
    assert woke == []          # not yet
    await clock.advance(1)
    await asyncio.sleep(0)     # let the sleeper resume
    assert woke == [300]
```

- [ ] **Step 2: Run red** — `uv run pytest tests/gateway/streaming/test_clock.py -q` → FAIL (no module).

- [ ] **Step 3: Implement `clock.py`**

```python
from __future__ import annotations

import asyncio
import heapq
import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...
    def sleep(self, ms: int) -> "asyncio.Future[None] | asyncio.Task[None]": ...


class MonotonicClock:
    """Production clock backed by the event loop."""

    def now_ms(self) -> int:
        return int(time.monotonic() * 1000)

    async def sleep(self, ms: int) -> None:
        await asyncio.sleep(max(0, ms) / 1000)


class VirtualClock:
    """Deterministic test clock. Time only moves when advance() is called."""

    def __init__(self) -> None:
        self._now = 0
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._counter = 0

    def now_ms(self) -> int:
        return self._now

    def sleep(self, ms: int) -> "asyncio.Future[None]":
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[None] = loop.create_future()
        deadline = self._now + max(0, ms)
        self._counter += 1
        heapq.heappush(self._waiters, (deadline, self._counter, fut))
        return fut

    async def advance(self, ms: int) -> None:
        target = self._now + max(0, ms)
        while self._waiters and self._waiters[0][0] <= target:
            deadline, _, fut = heapq.heappop(self._waiters)
            self._now = deadline
            if not fut.done():
                fut.set_result(None)
            await asyncio.sleep(0)
        self._now = target
```

- [ ] **Step 4: Run green** — `uv run pytest tests/gateway/streaming/test_clock.py -q` → PASS.

- [ ] **Step 5: Write failing types test**

`tests/gateway/streaming/test_types.py`:
```python
import dataclasses
import pytest
from gateway.calls.native.streaming.types import (
    AudioFrame, MediaFormat, BrainEvent, BrainEventKind, TurnEndReason, CallTurnRecord,
)


def test_audioframe_duration_ms_derived():
    media = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)
    # 20ms of 16kHz mono s16 = 0.02 * 16000 * 2 bytes = 640 bytes
    frame = AudioFrame(pcm16=b"\x00" * 640, media=media, timestamp_ms=0, seq=0)
    assert frame.duration_ms == 20


def test_brainevent_is_final():
    e = BrainEvent(call_id="c", kind=BrainEventKind.FINAL_TEXT, text="hi")
    assert e.is_final is True
    assert BrainEvent(call_id="c", kind=BrainEventKind.PARTIAL_TEXT).is_final is False


def test_callturnrecord_is_frozen():
    r = CallTurnRecord(call_id="c", turn_index=0, user_transcript="u", assistant_heard_text="h")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.user_transcript = "x"  # type: ignore[misc]
    assert r.ended_reason is TurnEndReason.COMPLETED
```

- [ ] **Step 6: Run red** → FAIL (no module).

- [ ] **Step 7: Implement `types.py`** (exactly the §5.1/§5.2 contract)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from gateway.calls.native.voice_turn import VoiceDebugTracePolicy


class TranscriptKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


class TurnEventKind(str, Enum):
    USER_SPEECH_STARTED = "user_speech_started"
    USER_SPEECH_STOPPED = "user_speech_stopped"
    ENDPOINT_DETECTED = "endpoint_detected"
    POSSIBLE_BACKCHANNEL = "possible_backchannel"


class BrainEventKind(str, Enum):
    PARTIAL_TEXT = "partial_text"
    TOOL_STATUS = "tool_status"
    FINAL_TEXT = "final_text"
    ERROR = "error"


class TtsEventKind(str, Enum):
    AUDIO = "audio"
    MARK = "mark"
    DONE = "done"
    CANCELLED = "cancelled"


class InterruptionAction(str, Enum):
    INTERRUPT = "interrupt"
    IGNORE = "ignore"
    RESUME = "resume"
    WAIT = "wait"


class TurnEndReason(str, Enum):
    COMPLETED = "completed"
    BARGED_IN = "barged_in"
    FALSE_INTERRUPTION = "false_interruption"
    BRAIN_ERROR = "brain_error"


@dataclass(frozen=True)
class MediaFormat:
    sample_rate: int
    channels: int = 1
    frame_ms: int = 20


@dataclass(frozen=True)
class AudioFrame:
    pcm16: bytes
    media: MediaFormat
    timestamp_ms: int
    seq: int

    @property
    def duration_ms(self) -> int:
        samples = len(self.pcm16) / 2 / self.media.channels
        return int(samples / self.media.sample_rate * 1000)


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0


@dataclass(frozen=True)
class TranscriptEvent:
    call_id: str
    kind: TranscriptKind
    text: str
    start_ms: int = 0
    end_ms: int = 0
    stability: float = 1.0
    words: tuple[WordTiming, ...] = ()
    provider: str = ""


@dataclass(frozen=True)
class TurnEvent:
    call_id: str
    kind: TurnEventKind
    at_ms: int
    speech_duration_ms: int = 0
    vad_confidence: float = 0.0
    endpoint_confidence: float = 0.0
    source: str = ""


@dataclass(frozen=True)
class BrainEvent:
    call_id: str
    kind: BrainEventKind
    text: str = ""
    tool_name: str = ""
    error_code: str = ""

    @property
    def is_final(self) -> bool:
        return self.kind is BrainEventKind.FINAL_TEXT


@dataclass(frozen=True)
class PlaybackMark:
    call_id: str
    char_offset: int          # against the FULL response text
    text_so_far: str          # full_text[:char_offset]
    at_ms: int
    boundary: str = "word"


@dataclass(frozen=True)
class TtsAudioEvent:
    call_id: str
    kind: TtsEventKind
    frame: AudioFrame | None = None
    mark: PlaybackMark | None = None
    span_text: str = ""
    span_start_char: int = 0
    span_end_char: int = 0


@dataclass(frozen=True)
class InterruptionParams:
    min_speech_ms: int = 350
    min_words: int = 2
    backchannel_max_ms: int = 600
    false_interruption_timeout_ms: int = 2000


@dataclass(frozen=True)
class EndpointParams:
    vad_confidence: float = 0.7
    start_secs: float = 0.2
    stop_secs: float = 0.2
    endpoint_threshold: float = 0.5
    max_delay_ms: int = 3000


@dataclass(frozen=True)
class InterruptionSignal:
    call_id: str
    at_ms: int
    assistant_speaking: bool
    turn_event: "TurnEvent | None"
    latest_partial: "TranscriptEvent | None"
    playhead: "PlaybackMark | None"
    params: InterruptionParams
    ms_since_speech_start: int = 0
    ms_since_assistant_silent_partial: int = 0


@dataclass(frozen=True)
class InterruptionDecision:
    action: InterruptionAction
    reason: str
    at_ms: int


@dataclass(frozen=True)
class FlushResult:
    dropped_frames: int
    dropped_ms: int
    last_sent_mark: "PlaybackMark | None"


@dataclass(frozen=True)
class StreamingCallContext:
    call_id: str
    contact_id: str
    session_id: str
    media: MediaFormat
    interruption: InterruptionParams = field(default_factory=InterruptionParams)
    endpoint: EndpointParams = field(default_factory=EndpointParams)
    debug: VoiceDebugTracePolicy = field(default_factory=VoiceDebugTracePolicy)


@dataclass(frozen=True)
class CallTurnRecord:
    call_id: str
    turn_index: int
    user_transcript: str
    assistant_heard_text: str
    assistant_abandoned_text: str = ""
    interrupted: bool = False
    ended_reason: TurnEndReason = TurnEndReason.COMPLETED
```

- [ ] **Step 8: Run green** — `uv run pytest tests/gateway/streaming/test_types.py -q` → PASS.

- [ ] **Step 9: Implement `ports.py`** (Protocols only — no logic; §5.4)

```python
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .cancellation import CancellationScope
from .types import (
    AudioFrame, BrainEvent, CallTurnRecord, FlushResult, InterruptionDecision,
    InterruptionSignal, MediaFormat, PlaybackMark, StreamingCallContext,
    TranscriptEvent, TtsAudioEvent, TurnEvent,
)


class CallTurnCancelled(Exception):
    """Raised by CancellationScope.raise_if_cancelled()."""


@runtime_checkable
class AudioTransportPort(Protocol):
    @property
    def media(self) -> MediaFormat: ...
    async def push_inbound(self, frame: AudioFrame) -> None: ...
    def inbound(self) -> AsyncIterator[AudioFrame]: ...
    async def emit_outbound(self, frame: AudioFrame) -> None: ...
    async def flush_outbound(self, reason: str) -> FlushResult: ...
    async def close(self) -> None: ...


@runtime_checkable
class TurnDetectionPort(Protocol):
    async def observe(self, frame: AudioFrame) -> tuple[TurnEvent, ...]: ...
    def reset(self) -> None: ...


@runtime_checkable
class InterruptionPolicyPort(Protocol):
    def decide(self, signal: InterruptionSignal) -> InterruptionDecision: ...


@runtime_checkable
class SpeechToTextPort(Protocol):
    async def start(self, ctx: StreamingCallContext) -> None: ...
    async def push(self, frame: AudioFrame) -> None: ...
    def events(self) -> AsyncIterator[TranscriptEvent]: ...
    async def finalize(self) -> TranscriptEvent | None: ...
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class TextToSpeechPort(Protocol):
    def synthesize(
        self, text: str, ctx: StreamingCallContext, scope: CancellationScope
    ) -> AsyncIterator[TtsAudioEvent]: ...
    async def cancel(self) -> None: ...
    async def flush(self) -> None: ...


@runtime_checkable
class HermesBrainPort(Protocol):
    def respond(
        self, turn: TranscriptEvent, ctx: StreamingCallContext, scope: CancellationScope
    ) -> AsyncIterator[BrainEvent]: ...


@runtime_checkable
class StreamingCallTracerPort(Protocol):
    def transcript(self, event: TranscriptEvent) -> None: ...
    def turn(self, event: TurnEvent) -> None: ...
    def brain(self, event: BrainEvent) -> None: ...
    def playback(self, mark: PlaybackMark) -> None: ...
    def interruption(self, decision: InterruptionDecision, heard: str, abandoned: str) -> None: ...
    def turn_committed(self, record: CallTurnRecord) -> None: ...
```

(Note: `ports.py` imports `cancellation.CancellationScope`; build `cancellation.py` in this task's Step 11 before importing, or define the import after WP1.5. To keep WP1 self-contained, implement `cancellation.py` here.)

- [ ] **Step 10: Run ast-grep + ty on the contract**

Run: `ast-grep scan gateway/calls/native/streaming && uv run ty check gateway/calls/native/streaming`
Expected: clean (frozen-dataclass + purity rules satisfied; types align).

- [ ] **Step 11: Implement `cancellation.py`** (needed by ports import)

```python
from __future__ import annotations

from collections.abc import Callable


class CallTurnCancelled(Exception):
    pass


class CancellationScope:
    """Cooperative cancellation shared by the reflex loop and the brain worker."""

    def __init__(self) -> None:
        self._cancelled = False
        self._reason = ""
        self._listeners: list[Callable[[str], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self._reason = reason
        for cb in list(self._listeners):
            try:
                cb(reason)
            except Exception:
                pass

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CallTurnCancelled(self._reason)

    def add_listener(self, cb: Callable[[str], None]) -> None:
        self._listeners.append(cb)
        if self._cancelled:
            cb(self._reason)
```

Remove the duplicate `CallTurnCancelled` from `ports.py` and import it from `cancellation` instead (DRY).

- [ ] **Step 12: Full gate + commit**

```bash
uv run ruff check gateway/calls/native/streaming tests/gateway/streaming
uv run ty check gateway/calls/native/streaming
ast-grep scan gateway/calls/native/streaming
uv run pytest tests/gateway/streaming -q
git add gateway/calls/native/streaming tests/gateway/streaming
git commit -m "feat(streaming): contract — clock, types, ports, cancellation (WP1)"
```

- [ ] **Step 13: CONTRACT REVIEW GATE** — dispatch `superpowers:code-reviewer` against this task + the spec §5. Fix findings before any parallel WP starts. The contract is now locked.

---

## Task 2 (WP2): InterruptionPolicy (pure) — parallel after WP1

**Files:** Create `gateway/calls/native/streaming/interruption.py`; Test `tests/gateway/streaming/test_interruption_policy.py`.

The decision truth table (deterministic):
- assistant **not** speaking → `WAIT` (no interruption concept).
- assistant speaking + `turn_event.kind == POSSIBLE_BACKCHANNEL` → `IGNORE`.
- assistant speaking + speech ≥ `min_speech_ms` AND latest final/partial word-count ≥ `min_words` → `INTERRUPT`.
- assistant speaking + a flush already happened + `ms_since_speech_start ≥ false_interruption_timeout_ms` + still no qualifying transcript → `RESUME`.
- otherwise → `WAIT`.

- [ ] **Step 1: Write failing truth-table tests** (one assertion per row)

```python
from gateway.calls.native.streaming.interruption import InterruptionPolicy
from gateway.calls.native.streaming.types import (
    InterruptionSignal, InterruptionParams, InterruptionAction, TurnEvent,
    TurnEventKind, TranscriptEvent, TranscriptKind,
)

P = InterruptionParams()


def _sig(**kw):
    base = dict(call_id="c", at_ms=0, assistant_speaking=True, turn_event=None,
               latest_partial=None, playhead=None, params=P,
               ms_since_speech_start=0, ms_since_assistant_silent_partial=0)
    base.update(kw)
    return InterruptionSignal(**base)


def test_not_speaking_waits():
    d = InterruptionPolicy().decide(_sig(assistant_speaking=False))
    assert d.action is InterruptionAction.WAIT


def test_backchannel_ignored():
    te = TurnEvent(call_id="c", kind=TurnEventKind.POSSIBLE_BACKCHANNEL, at_ms=0)
    d = InterruptionPolicy().decide(_sig(turn_event=te))
    assert d.action is InterruptionAction.IGNORE


def test_real_interrupt():
    partial = TranscriptEvent(call_id="c", kind=TranscriptKind.PARTIAL, text="stop talking now")
    d = InterruptionPolicy().decide(_sig(latest_partial=partial, ms_since_speech_start=400))
    assert d.action is InterruptionAction.INTERRUPT


def test_one_word_below_min_words_waits():
    partial = TranscriptEvent(call_id="c", kind=TranscriptKind.PARTIAL, text="uh")
    d = InterruptionPolicy().decide(_sig(latest_partial=partial, ms_since_speech_start=400))
    assert d.action is InterruptionAction.WAIT


def test_false_positive_resume_after_timeout():
    d = InterruptionPolicy().decide(_sig(ms_since_speech_start=P.false_interruption_timeout_ms + 1))
    assert d.action is InterruptionAction.RESUME
```

- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement `interruption.py`:**

```python
from __future__ import annotations

from .types import InterruptionAction, InterruptionDecision, InterruptionSignal, TurnEventKind


def _word_count(signal: InterruptionSignal) -> int:
    p = signal.latest_partial
    return len((p.text or "").split()) if p else 0


class InterruptionPolicy:
    """Pure, deterministic barge-in policy. No I/O, no clock."""

    def decide(self, signal: InterruptionSignal) -> InterruptionDecision:
        at = signal.at_ms
        if not signal.assistant_speaking:
            return InterruptionDecision(InterruptionAction.WAIT, "assistant_idle", at)

        te = signal.turn_event
        if te is not None and te.kind is TurnEventKind.POSSIBLE_BACKCHANNEL:
            return InterruptionDecision(InterruptionAction.IGNORE, "backchannel", at)

        sustained = signal.ms_since_speech_start >= signal.params.min_speech_ms
        enough_words = _word_count(signal) >= signal.params.min_words
        if sustained and enough_words:
            return InterruptionDecision(InterruptionAction.INTERRUPT, "real_barge_in", at)

        if signal.ms_since_speech_start >= signal.params.false_interruption_timeout_ms:
            return InterruptionDecision(InterruptionAction.RESUME, "false_interruption_timeout", at)

        return InterruptionDecision(InterruptionAction.WAIT, "insufficient_evidence", at)
```

- [ ] **Step 4: Run green.**
- [ ] **Step 5: Gate + commit** `feat(streaming): pure interruption policy (WP2)`.

---

## Task 3 (WP3): HeardSpanLedger — parallel after WP1

**Files:** Create `gateway/calls/native/streaming/ledger.py`; Test `test_ledger.py`.

- [ ] **Step 1: Failing tests**

```python
from gateway.calls.native.streaming.ledger import HeardSpanLedger
from gateway.calls.native.streaming.types import PlaybackMark, FlushResult, TurnEndReason


def test_full_playback_heard_all():
    led = HeardSpanLedger("call")
    full = "hello there friend"
    led.note_mark(PlaybackMark("call", char_offset=len(full), text_so_far=full, at_ms=100))
    rec = led.record(user_transcript="hi", full_text=full, reason=TurnEndReason.COMPLETED)
    assert rec.assistant_heard_text == full
    assert rec.assistant_abandoned_text == ""
    assert rec.interrupted is False


def test_partial_heard_truncation():
    led = HeardSpanLedger("call")
    full = "the quick brown fox jumps"
    led.note_mark(PlaybackMark("call", char_offset=9, text_so_far="the quick", at_ms=80))
    led.note_flush(FlushResult(dropped_frames=4, dropped_ms=80, last_sent_mark=None), full)
    rec = led.record(user_transcript="u", full_text=full, reason=TurnEndReason.BARGED_IN)
    assert rec.assistant_heard_text == "the quick"
    assert rec.assistant_abandoned_text == full[9:]
    assert rec.interrupted is True
    assert rec.ended_reason is TurnEndReason.BARGED_IN


def test_barge_in_during_thinking_nothing_heard():
    led = HeardSpanLedger("call")
    full = "discarded answer"
    rec = led.record(user_transcript="u", full_text=full, reason=TurnEndReason.BARGED_IN)
    assert rec.assistant_heard_text == ""
    assert rec.assistant_abandoned_text == full
    assert rec.interrupted is True
```

- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement `ledger.py`:**

```python
from __future__ import annotations

from .types import CallTurnRecord, FlushResult, PlaybackMark, TurnEndReason


class HeardSpanLedger:
    """Single source of truth for playback-aware truncation (spec §6 D/E, constraint #6)."""

    def __init__(self, call_id: str, turn_index: int = 0) -> None:
        self.call_id = call_id
        self.turn_index = turn_index
        self._last_offset = 0
        self._last_text = ""

    def note_mark(self, mark: PlaybackMark) -> None:
        if mark.char_offset >= self._last_offset:
            self._last_offset = mark.char_offset
            self._last_text = mark.text_so_far

    def note_flush(self, flush: FlushResult, full_text: str) -> None:
        if flush.last_sent_mark is not None:
            self.note_mark(flush.last_sent_mark)
        # heard span is whatever marks already recorded; flush does not extend it.

    def record(self, *, user_transcript: str, full_text: str, reason: TurnEndReason) -> CallTurnRecord:
        heard = self._last_text if self._last_text else full_text[: self._last_offset]
        abandoned = full_text[len(heard):] if reason is not TurnEndReason.COMPLETED else ""
        if reason is TurnEndReason.COMPLETED and not heard:
            heard = full_text
        interrupted = reason in (TurnEndReason.BARGED_IN, TurnEndReason.FALSE_INTERRUPTION)
        return CallTurnRecord(
            call_id=self.call_id,
            turn_index=self.turn_index,
            user_transcript=user_transcript,
            assistant_heard_text=heard,
            assistant_abandoned_text=abandoned,
            interrupted=interrupted,
            ended_reason=reason,
        )
```

- [ ] **Step 4: Run green.** (Adjust `record` until all three tests pass — note the "thinking" case has `_last_offset==0` and `_last_text==""`, so heard="" and abandoned=full.)
- [ ] **Step 5: Gate + commit** `feat(streaming): heard-span ledger (WP3)`.

---

## Task 4 (WP4): Fakes + harness — parallel after WP1

**Files:** Create `gateway/calls/native/streaming/fakes.py`; Test `test_fakes.py`.

Implement (each honoring its Section-2 port):
- `FakeAudioTransport(media)` — `push_inbound`/`inbound()` via an `asyncio.Queue`; records `emit_outbound` frames in `self.sent`; `flush_outbound(reason)` clears a pending-outbound list and returns `FlushResult(dropped_frames=len(pending), dropped_ms=sum, last_sent_mark=self._last_mark)`.
- `FakeTurnDetection(script)` — `observe(frame)` returns the scripted `TurnEvent`s whose trigger seq matches `frame.seq`.
- `FakeSTT(script)` — `events()` yields scripted partials/finals; `finalize()` returns the final or `None` (the zero-final case for Scenario G).
- `FakeTTS(frames_per_word=N, clock)` — `synthesize(text, ctx, scope)` yields `TtsAudioEvent(AUDIO)` per chunk and a `MARK` (`PlaybackMark`) at each word boundary using cumulative `char_offset`; checks `scope.cancelled` between words and yields `CANCELLED` then stops; ends with `DONE`.
- `FakeBrain(text, delay_ms, clock)` — `respond()` awaits `clock.sleep(delay_ms)`, checks `scope.cancelled`/`AIAgent.interrupt`-equivalent flag, then yields one `BrainEvent(FINAL_TEXT)`; sets `self.abandoned=True` if cancelled before yielding.

- [ ] **Step 1: Failing tests** asserting: FakeTTS emits a MARK per word with monotonically increasing `char_offset`; FakeTTS stops + emits CANCELLED after `scope.cancel()`; FakeBrain marks itself abandoned when cancelled during its delay; FakeAudioTransport.flush_outbound returns the dropped count.
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement `fakes.py`.** (Full code; ~150 lines. Use `VirtualClock` injection — no real sleeps; `ast-grep` will fail otherwise.)
- [ ] **Step 4: Run green.**
- [ ] **Step 5: Gate + commit** `feat(streaming): deterministic fakes + harness (WP4)`.

---

## Task 5 (WP5): HermesSyncBrain adapter — parallel after WP1

**Files:** Create `gateway/calls/native/streaming/brain.py`; Test `test_brain.py` (mocks `AIAgent`, never calls a real model).

Behavior (spec §4.4, Decisions B & C):
- `respond(turn, ctx, scope)` runs `AIAgent.run_conversation(turn.text, persist_user_message=None)` inside `asyncio.to_thread`.
- Registers `scope.add_listener(lambda r: agent.interrupt(r))` so a barge-in fires the cooperative `AIAgent.interrupt()`.
- After the thread returns: if `scope.cancelled`, do NOT yield, do NOT persist; mark `self.abandoned=True`. Else yield one `BrainEvent(FINAL_TEXT, text=result["final_response"])`.
- **Deferred persistence (Decision C):** construct the call `AIAgent` with `skip_memory=True` (already used by the call agent path) so no `SessionDB` write occurs; durability is the tracer's `CallTurnRecord` only.

- [ ] **Step 1: Failing tests** with a `FakeAgent` exposing `run_conversation` + `interrupt`:
  - normal: yields one FINAL_TEXT equal to `final_response`.
  - cancelled-before-return: no event yielded, `abandoned is True`, `interrupt()` was called, `run_conversation` persistence not invoked (assert `FakeAgent.skip_memory is True`).
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement `brain.py`** (inject an `agent_factory: Callable[[StreamingCallContext], Any]` so tests pass a `FakeAgent`; production factory builds the real call `AIAgent` with `skip_memory=True`, low `max_iterations`/`max_tokens`, per the existing `_call_agent_kwargs` pattern in `voice_turn.py`).
- [ ] **Step 4: Run green.**
- [ ] **Step 5: Gate + commit** `feat(streaming): HermesSyncBrain with cooperative interrupt + deferred persist (WP5)`.

---

## Task 6 (WP6): StreamingCallTracer — parallel after WP1

**Files:** Create `gateway/calls/native/streaming/tracer.py`; Test `test_tracer.py`.

- Subclass/compose `NativeCallTraceWriter` (reuse its redaction). Each method calls `self.record(call_id, event, **fields)`.
- Sensitive previews (transcript/brain/playback text) are emitted ONLY when `debug.transcript_previews` is true, marked `sensitive=True`, truncated via the existing `_preview_text` from `voice_turn.py`. Reuse it (DRY) — import, don't reimplement.
- `interruption(...)` and `turn_committed(record)` always emit structural fields (counts, reasons, offsets) but gate the raw `heard`/`abandoned`/`user_transcript` text behind the debug policy.

- [ ] **Step 1: Failing tests:** with `debug.transcript_previews=False`, a `transcript()` call writes an event with NO `preview` key; with it True, writes `preview` + `sensitive=True`; sensitive keys (e.g. a value under `rawAudio`) are `[REDACTED]`.
- [ ] **Step 2–4:** red → implement `tracer.py` → green.
- [ ] **Step 5: Gate + commit** `feat(streaming): streaming call tracer with gated previews (WP6)`.

---

## Task 7 (WP7): StreamingCallSession reflex core — after WP2,3,4,6

**Files:** Create `gateway/calls/native/streaming/session.py`; Test `test_session_basic.py` (full A–G live in WP8).

Responsibilities (spec §5.5, §5.3 ordering):
- `run()` consumes `transport.inbound()`; pushes frames to `stt` + `turns`.
- On `ENDPOINT_DETECTED` with a FINAL transcript → start an assistant turn: open a `CancellationScope`, stream `brain.respond(...)`, then `_speak(final_text, scope)`.
- `_speak` consumes `tts.synthesize(...)`, calls `transport.emit_outbound` per AUDIO frame, feeds each `PlaybackMark` to `ledger.note_mark` + `tracer.playback`.
- While speaking, each inbound `TurnEvent` builds an `InterruptionSignal` (using the injected `Clock` for `at_ms`/`ms_since_speech_start`) and calls `policy.decide`:
  - `INTERRUPT` → `transport.flush_outbound("barge_in")` → `tts.cancel()` → `scope.cancel("barge_in")` (fires `AIAgent.interrupt()`) → `ledger.note_flush` → commit `CallTurnRecord(BARGED_IN)` via `tracer.turn_committed` → begin new turn.
  - `IGNORE` → keep speaking.
  - `RESUME` → resume `_speak` from `ledger` last offset (no flush, no new record).
  - On a transient VAD trigger (USER_SPEECH_STARTED while speaking) → `flush_outbound("vad_trigger")` immediately but DO NOT cancel the brain (deferred per §5.3).
- Brain latency must not block inbound: brain runs concurrently (its own task); the inbound loop keeps draining (Scenario E).

- [ ] **Step 1: Failing test** — a single normal turn (Scenario A subset) drives session with all fakes + `VirtualClock` and asserts a `CallTurnRecord(COMPLETED)` was committed and outbound frames were emitted.
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement `session.py`.** Keep it focused; push pure decisions to WP2/WP3.
- [ ] **Step 4: Run green.**
- [ ] **Step 5: Gate + commit** `feat(streaming): reflex session core (WP7)`.

---

## Task 8 (WP8): Acceptance suite A–G — after WP4,5,7

**Files:** Test `tests/gateway/streaming/test_session_scenarios.py`.

One test per scenario (spec §6). Each builds fakes + `VirtualClock`, scripts the inputs, runs the session, advances the clock, and asserts the exact `CallTurnRecord` + event ordering:
- `test_scenario_a_normal_turn` → COMPLETED, heard == full text.
- `test_scenario_b_barge_in_during_speech` → flush called, tts cancelled, brain abandoned, record BARGED_IN with heard+abandoned split.
- `test_scenario_c_backchannel_ignored` → no flush, speech finishes, no extra record.
- `test_scenario_d_partial_heard_truncation` → heard == prefix, abandoned == suffix, interrupted True.
- `test_scenario_e_brain_latency_nonblocking` → with `FakeBrain(delay_ms=4000)`, inbound frames keep being consumed during the delay; barge-in during thinking abandons the brain with heard="".
- `test_scenario_f_*` lives in WP11.
- `test_scenario_g_false_positive_resume` → flush happened, no qualifying transcript within timeout, RESUME, no duplicated prefix, no BARGED_IN record.

- [ ] **Step 1–2:** write all scenario tests; run red.
- [ ] **Step 3:** fix any session gaps surfaced (iterate WP7 ↔ WP8).
- [ ] **Step 4:** all green: `uv run pytest tests/gateway/streaming/test_session_scenarios.py -q`.
- [ ] **Step 5: Gate + commit** `test(streaming): acceptance scenarios A-E,G in simulation (WP8)`.

---

## Task 9 (WP9): CLI debug command `calls simplex-simulate-stream` — after WP7,8

**Files:** Modify `hermes_cli/calls.py` (add subcommand mirroring `simplex-simulate-voice-turn`); Modify `hermes_cli/main.py` if dispatch needs it; Test `tests/hermes_cli/test_calls_command.py` (extend).

- Command runs the session with fakes + a scripted caller utterance and prints a JSON summary (`connected`, `transcript_chars`, `heard_chars`, `abandoned_chars`, `ended_reason`, `interrupted`).
- [ ] TDD steps as above; assert `--json` output keys. Gate + commit `feat(cli): calls simplex-simulate-stream (WP9)`.

---

## Task 10 (WP10): Pipecat production transport + smoke — after WP1,7 (CONDITIONAL on WP0 Step 3)

**Files:** Create `gateway/calls/native/streaming/pipecat_transport.py`; Test `test_pipecat_smoke.py`.

- If WP0 Step 3 showed no wheels: `test_pipecat_smoke.py` begins with `pipecat = pytest.importorskip("pipecat")` and the adapter file contains only the interface + a `NotImplementedError`-guarded `build_pipeline()` documented as deferred (Decision A). Commit and move on.
- If wheels exist: implement the custom `BaseTransport` translating `AudioFrame`⇄`InputAudioRawFrame`/`OutputAudioRawFrame` over `aiortc_engine.py`, and `HermesLLMService(LLMService)` whose `run_inference` delegates to `HermesSyncBrain`. Smoke test: one canned turn reaches `DONE`.
- [ ] TDD/skip steps; gate + commit `feat(streaming): pipecat transport adapter + smoke (WP10)`.

---

## Task 11 (WP11): Engine selector + Scenario F — after WP7 (serial; touches shared files)

**Files:** Create `gateway/calls/native/streaming/engine.py`; Modify `hermes_cli/config.py` (default `calls.native.engine: turn_based`), `gateway/calls/native/application.py` (route via selector); Test `test_engine_selection.py`.

- `select_call_engine(config) -> str` returns `"streaming"` only when `calls.native.engine == "streaming"`, else `"turn_based"` (default; unknown values fall back to turn_based with a warning).
- `application.py` consults the selector at call setup; default path is byte-for-byte the existing behavior (assert existing native-call tests still pass).
- [ ] **Step 1: Failing tests** — Scenario F both directions; default/missing/garbage → turn_based.
- [ ] **Step 2–4:** red → implement → green.
- [ ] **Step 5: Regression** — `uv run pytest tests/gateway/test_native_call_application.py tests/gateway/test_simplex_plugin.py -q` → still pass (zero behavior change at default).
- [ ] **Step 6: Gate + commit** `feat(streaming): engine selector with turn_based default (WP11)`.

---

## Task 12 (WP12): Integration verification + full sweep — after all

**Files:** none (verification); update spec doc "Status" line.

- [ ] **Step 1: Update slice exports** in `gateway/calls/native/streaming/__init__.py` (public types/classes).
- [ ] **Step 2: `kit` alignment check** — verify every `Protocol` method in `ports.py` has a fake impl (`fakes.py`) AND, where applicable, a production impl; run the duplicate-function scan. If `kit` was degraded in WP0, do the manual review instead and note it.
  Run (verify exact subcommand in WP0): `kit symbols gateway/calls/native/streaming` / `kit review` per WP0 findings.
- [ ] **Step 3: Full lint + type + structural sweep**
  ```bash
  uv run ruff check gateway/calls/native/streaming hermes_cli tests/gateway/streaming tests/hermes_cli
  uv run ty check gateway/calls/native/streaming
  ast-grep scan gateway/calls/native/streaming
  ```
  Expected: all clean.
- [ ] **Step 4: Full focused test run (serial, per handoff caveat about xdist)**
  ```bash
  uv run pytest -n0 tests/gateway/streaming tests/hermes_cli/test_calls_command.py \
    tests/gateway/test_native_call_application.py tests/gateway/test_simplex_plugin.py -q
  ```
  Expected: all pass; A–G green; default engine regression intact.
- [ ] **Step 5: `superpowers:requesting-code-review`** for the whole slice against the spec; fix findings.
- [ ] **Step 6: `superpowers:verification-before-completion`** — paste the actual command output proving the Definition of Done (spec §8.4) before claiming completion.
- [ ] **Step 7: Commit** `chore(streaming): integration verification + DoD evidence (WP12)`.

---

## Definition of Done (spec §8.4)

- A–G + Scenario F green in simulation (paste output).
- Pipecat smoke green OR explicitly skipped with the WP0 wheel finding logged.
- `ruff` / `ty` / `ast-grep` clean; `kit` alignment clean (or degraded-with-manual-review noted).
- `calls.native.engine` defaults to `turn_based`; existing native-call tests unchanged.
- Streaming path reachable only via `engine=streaming`.
- No live iPhone test performed in this slice (gated to a follow-on).
