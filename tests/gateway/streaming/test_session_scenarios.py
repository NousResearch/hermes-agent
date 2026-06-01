"""Acceptance scenarios A-E,G for StreamingCallSession (WP8).

These tests drive the reflex core (WP7) in simulation with the deterministic
fakes + VirtualClock.  Each test asserts the committed CallTurnRecord(s) and the
key transport side effects (flushes / emitted audio / TTS cancel).

Scenario F (engine selector) lives in WP11 and is intentionally absent here.

Driver mechanics
----------------
The session runs as a background task.  The brain and TTS fakes sleep via the
injected VirtualClock, so the driver must *advance* the clock to let those
generators make progress.  We expose:

  * ``push(seq)``         — enqueue an inbound frame with the given seq (which
                            triggers any turn events scripted for that seq in
                            FakeTurnDetection), then yield control.
  * ``advance(ms)``       — move the virtual clock forward and yield control so
                            waiters resolve.
  * ``settle()``          — yield control repeatedly so chained tasks progress.
  * ``end()``             — close the inbound stream and await ``run()``.
"""
from __future__ import annotations

import asyncio

import pytest

from gateway.calls.native.streaming.fakes import (
    FakeAudioTransport,
    FakeBrain,
    FakeSTT,
    FakeTTS,
    FakeTurnDetection,
)
from gateway.calls.native.streaming.interruption import InterruptionPolicy
from gateway.calls.native.streaming.ledger import HeardSpanLedger
from gateway.calls.native.streaming.session import StreamingCallSession
from gateway.calls.native.streaming.tracer import StreamingCallTracer
from gateway.calls.native.streaming.types import (
    AudioFrame,
    InterruptionParams,
    MediaFormat,
    StreamingCallContext,
    TranscriptEvent,
    TranscriptKind,
    TurnEndReason,
    TurnEvent,
    TurnEventKind,
)
from gateway.calls.native.streaming.clock import VirtualClock

pytestmark = pytest.mark.asyncio

MEDIA = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)
CALL_ID = "call-scenario"


def make_ctx(interruption: InterruptionParams | None = None) -> StreamingCallContext:
    return StreamingCallContext(
        call_id=CALL_ID,
        contact_id="contact-1",
        session_id="session-1",
        media=MEDIA,
        interruption=interruption or InterruptionParams(),
    )


def make_frame(seq: int, duration_ms: int = 20) -> AudioFrame:
    samples = int(MEDIA.sample_rate * duration_ms / 1000)
    pcm = b"\x00" * samples * 2
    return AudioFrame(pcm16=pcm, media=MEDIA, timestamp_ms=seq * duration_ms, seq=seq)


def partial(text: str) -> TranscriptEvent:
    return TranscriptEvent(call_id=CALL_ID, kind=TranscriptKind.PARTIAL, text=text)


def final_transcript(text: str) -> TranscriptEvent:
    return TranscriptEvent(call_id=CALL_ID, kind=TranscriptKind.FINAL, text=text)


def turn_event(seq_kind: TurnEventKind, at_ms: int = 0) -> TurnEvent:
    return TurnEvent(call_id=CALL_ID, kind=seq_kind, at_ms=at_ms)


class Driver:
    """Drives a StreamingCallSession in simulation."""

    def __init__(
        self,
        *,
        ctx: StreamingCallContext,
        transport: FakeAudioTransport,
        stt: FakeSTT,
        turns: FakeTurnDetection,
        tts: FakeTTS,
        brains: list[FakeBrain],
        brain_factory,
        clock: VirtualClock,
    ) -> None:
        self.ctx = ctx
        self.transport = transport
        self.stt = stt
        self.turns = turns
        self.tts = tts
        self.brains = brains
        self.clock = clock
        self.session = StreamingCallSession(
            ctx,
            transport=transport,
            stt=stt,
            turns=turns,
            tts=tts,
            brain_factory=brain_factory,
            policy=InterruptionPolicy(),
            tracer=StreamingCallTracer(ctx.call_id),
            clock=clock,
        )
        self._run_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._run_task = asyncio.create_task(self.session.run())
        await self.settle()

    async def settle(self, ticks: int = 5) -> None:
        for _ in range(ticks):
            await asyncio.sleep(0)

    async def push(self, seq: int) -> None:
        await self.transport.push_inbound(make_frame(seq))
        await self.settle()

    async def advance(self, ms: int, *, settle: bool = True) -> None:
        await self.clock.advance(ms)
        if settle:
            await self.settle()

    async def drain_clock(self, total_ms: int, step_ms: int = 20) -> None:
        """Advance the clock in small steps so generator sleeps resolve in order."""
        steps = max(1, total_ms // step_ms)
        for _ in range(steps):
            await self.clock.advance(step_ms)
            await self.settle(ticks=3)

    async def end(self) -> None:
        await self.transport.end_inbound()
        assert self._run_task is not None
        # Drain the clock generously so any in-flight brain/TTS finishes.
        for _ in range(400):
            if self._run_task.done():
                break
            await self.clock.advance(20)
            await self.settle(ticks=3)
        await asyncio.wait_for(self._run_task, timeout=2.0)


def build_driver(
    *,
    script: list[tuple[int, TurnEvent]],
    stt: FakeSTT,
    brain_text: str,
    brain_delay_ms: int = 0,
    frames_per_word: int = 2,
    interruption: InterruptionParams | None = None,
) -> Driver:
    clock = VirtualClock()
    ctx = make_ctx(interruption)
    transport = FakeAudioTransport(MEDIA)
    turns = FakeTurnDetection(script)
    tts = FakeTTS(clock, frames_per_word=frames_per_word)
    brains: list[FakeBrain] = []

    def brain_factory() -> FakeBrain:
        b = FakeBrain(clock, text=brain_text, delay_ms=brain_delay_ms)
        brains.append(b)
        return b

    return Driver(
        ctx=ctx,
        transport=transport,
        stt=stt,
        turns=turns,
        tts=tts,
        brains=brains,
        brain_factory=brain_factory,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# Scenario A — normal turn
# ---------------------------------------------------------------------------


async def test_scenario_a_normal_turn():
    stt = FakeSTT(final=final_transcript("what's the weather"))
    driver = build_driver(
        script=[(0, turn_event(TurnEventKind.ENDPOINT_DETECTED))],
        stt=stt,
        brain_text="It's sunny.",
        brain_delay_ms=0,
    )
    await driver.start()
    await driver.push(0)  # ENDPOINT_DETECTED → launch assistant turn
    await driver.end()

    records = driver.session.records
    assert len(records) == 1
    rec = records[0]
    assert rec.ended_reason is TurnEndReason.COMPLETED
    assert rec.assistant_heard_text == "It's sunny."
    assert rec.interrupted is False
    assert len(driver.transport.sent) > 0  # audio frames emitted
    assert driver.transport.flushes == []  # no flush on a clean turn


# ---------------------------------------------------------------------------
# Scenario B — barge-in during speech
# ---------------------------------------------------------------------------


async def test_scenario_b_barge_in_during_speech():
    # Long response with many frames-per-word so TTS is still mid-stream when we
    # barge in.  Small min_speech_ms so the interrupt qualifies after a short
    # clock advance that does NOT drain the whole TTS stream.
    text = "one two three four five six seven eight nine ten"
    params = InterruptionParams(min_speech_ms=40, min_words=2)
    stt = FakeSTT(
        partials=[partial("hold on")],  # >=2 words → satisfies min_words
        final=final_transcript("question one"),
    )
    driver = build_driver(
        script=[
            (0, turn_event(TurnEventKind.ENDPOINT_DETECTED)),
            (1, turn_event(TurnEventKind.USER_SPEECH_STARTED)),
            # A non-speech-started event escalates without resetting the timer.
            (2, turn_event(TurnEventKind.USER_SPEECH_STOPPED)),  # triggers INTERRUPT
        ],
        stt=stt,
        brain_text=text,
        brain_delay_ms=0,
        frames_per_word=10,  # 200ms/word → plenty of runway to interrupt mid-stream
        interruption=params,
    )
    await driver.start()
    await driver.push(0)  # launch assistant turn (brain delay 0 → straight to TTS)

    # Let TTS emit the first word's worth of frames (1 MARK recorded).
    for _ in range(11):
        await driver.advance(20)

    # User starts speaking: stamp speech start + vad_trigger flush.
    await driver.push(1)
    # Sustain just past min_speech_ms (40) so policy escalates to INTERRUPT,
    # while the long TTS stream is still in progress.
    await driver.advance(params.min_speech_ms + 20)

    # Capture the exact heard prefix at the moment of interrupt (mirror D).
    last_mark_before = driver.session._last_mark
    assert last_mark_before is not None
    heard_at_interrupt = last_mark_before.text_so_far

    await driver.push(2)  # escalating event → INTERRUPT

    await driver.end()

    records = driver.session.records
    assert len(records) == 1
    rec = records[0]
    assert rec.ended_reason is TurnEndReason.BARGED_IN
    assert rec.interrupted is True
    # The transient USER_SPEECH_STARTED-while-speaking fires the fast-reflex
    # vad_trigger flush BEFORE the policy escalates to a barge_in interrupt
    # (spec §5.3 flush-vs-interrupt order). This regresses if
    # flush_outbound("vad_trigger") is removed from session._on_turn_event.
    assert "vad_trigger" in driver.transport.flushes
    assert "barge_in" in driver.transport.flushes
    assert driver.tts._cancelled is True
    # heard equals exactly the prefix confirmed at interrupt time; abandoned is
    # the remaining suffix.
    assert rec.assistant_heard_text == heard_at_interrupt
    assert rec.assistant_abandoned_text == text[len(heard_at_interrupt):]
    assert rec.assistant_heard_text + rec.assistant_abandoned_text == text


# ---------------------------------------------------------------------------
# Scenario C — backchannel ignored
# ---------------------------------------------------------------------------


async def test_scenario_c_backchannel_ignored():
    text = "one two three four"
    stt = FakeSTT(final=final_transcript("tell me a story"))
    driver = build_driver(
        script=[
            (0, turn_event(TurnEventKind.ENDPOINT_DETECTED)),
            (1, turn_event(TurnEventKind.POSSIBLE_BACKCHANNEL)),
        ],
        stt=stt,
        brain_text=text,
        brain_delay_ms=0,
    )
    await driver.start()
    await driver.push(0)
    await driver.advance(20)  # let TTS start speaking
    await driver.push(1)  # POSSIBLE_BACKCHANNEL while speaking → IGNORE
    await driver.end()

    records = driver.session.records
    assert len(records) == 1
    rec = records[0]
    assert rec.ended_reason is TurnEndReason.COMPLETED
    assert rec.assistant_heard_text == text
    assert rec.interrupted is False
    assert "barge_in" not in driver.transport.flushes
    assert driver.transport.flushes == []  # backchannel triggers no flush at all


# ---------------------------------------------------------------------------
# Scenario D — partial-heard truncation (exact prefix/suffix)
# ---------------------------------------------------------------------------


async def test_scenario_d_partial_heard_truncation():
    text = "alpha bravo charlie delta echo foxtrot golf hotel"
    params = InterruptionParams(min_speech_ms=40, min_words=2)
    stt = FakeSTT(
        partials=[partial("excuse me")],  # 2 words
        final=final_transcript("first question"),
    )
    driver = build_driver(
        script=[
            (0, turn_event(TurnEventKind.ENDPOINT_DETECTED)),
            (1, turn_event(TurnEventKind.USER_SPEECH_STARTED)),
            (2, turn_event(TurnEventKind.USER_SPEECH_STOPPED)),
        ],
        stt=stt,
        brain_text=text,
        brain_delay_ms=0,
        frames_per_word=10,  # 200ms/word → interrupt lands mid-stream
        interruption=params,
    )
    await driver.start()
    await driver.push(0)

    # Advance enough to complete the first two words (10 frames each = 20 frames),
    # which records two MARKs; stop before word three.
    for _ in range(21):
        await driver.advance(20)

    last_mark_before = driver.session._last_mark
    assert last_mark_before is not None
    heard_at_interrupt = last_mark_before.text_so_far

    await driver.push(1)  # speech start + vad_trigger flush
    await driver.advance(params.min_speech_ms + 20)
    await driver.push(2)  # INTERRUPT

    await driver.end()

    records = driver.session.records
    assert len(records) == 1
    rec = records[0]
    assert rec.ended_reason is TurnEndReason.BARGED_IN
    assert rec.interrupted is True
    assert rec.assistant_heard_text == heard_at_interrupt
    assert rec.assistant_abandoned_text == text[len(heard_at_interrupt):]
    assert rec.assistant_heard_text + rec.assistant_abandoned_text == text


# ---------------------------------------------------------------------------
# Scenario E — brain latency non-blocking + barge-in during thinking
# ---------------------------------------------------------------------------


async def test_scenario_e_brain_latency():
    stt = FakeSTT(
        partials=[partial("wait stop")],  # 2 words
        final=final_transcript("a long question"),
    )
    driver = build_driver(
        script=[
            (0, turn_event(TurnEventKind.ENDPOINT_DETECTED)),
            (1, turn_event(TurnEventKind.USER_SPEECH_STARTED)),
            (2, turn_event(TurnEventKind.USER_SPEECH_STOPPED)),
        ],
        stt=stt,
        brain_text="the eventual answer",
        brain_delay_ms=4000,  # brain "thinks" for 4s
        interruption=InterruptionParams(min_speech_ms=40, min_words=2),
    )
    await driver.start()
    await driver.push(0)  # launch assistant turn; brain begins thinking

    # Part 1: while the brain thinks (clock barely advanced), inbound keeps
    # draining.  Push extra frames and confirm they are consumed by STT.
    await driver.advance(40)
    assert driver.session._thinking is True
    assert driver.session._speaking is False
    await driver.push(5)  # no scripted turn event, but must be consumed
    await driver.push(6)
    # If inbound were blocked on the brain, these pushes would queue and the
    # session would never observe them.  Confirm the brain has not finished.
    assert driver.session._thinking is True
    assert driver.brains[0].started is True

    # Part 2: qualifying barge-in DURING thinking.
    await driver.push(1)  # USER_SPEECH_STARTED → speech start + vad_trigger flush
    await driver.advance(60)  # > min_speech_ms (40), still < brain delay (4000)
    await driver.push(2)  # escalating event → INTERRUPT (cancels brain)

    await driver.end()

    records = driver.session.records
    assert len(records) == 1
    rec = records[0]
    assert rec.ended_reason is TurnEndReason.BARGED_IN
    assert rec.interrupted is True
    assert driver.brains[0].abandoned is True
    # Nothing was spoken (brain cancelled before yielding FINAL_TEXT).
    assert rec.assistant_heard_text == ""


# ---------------------------------------------------------------------------
# Scenario G — false-positive resume
# ---------------------------------------------------------------------------


async def test_scenario_g_false_positive_resume():
    text = "one two three four"
    # No qualifying partial: empty STT partials → word_count 0 → never INTERRUPT.
    stt = FakeSTT(final=final_transcript("a question"))
    params = InterruptionParams(false_interruption_timeout_ms=100)
    driver = build_driver(
        script=[
            (0, turn_event(TurnEventKind.ENDPOINT_DETECTED)),
            (1, turn_event(TurnEventKind.USER_SPEECH_STARTED)),
            (2, turn_event(TurnEventKind.USER_SPEECH_STOPPED)),  # → RESUME
        ],
        stt=stt,
        brain_text=text,
        brain_delay_ms=0,
        frames_per_word=8,  # 160ms/word: still mid-stream past the 100ms timeout
        interruption=params,
    )
    await driver.start()
    await driver.push(0)
    await driver.advance(20)  # TTS starts speaking

    await driver.push(1)  # transient speech: vad_trigger flush, no qualifying transcript
    # Advance past the false-interruption timeout so the next event RESUMEs.
    await driver.advance(params.false_interruption_timeout_ms + 40)
    await driver.push(2)  # escalating event → policy RESUME (no qualifying words)

    await driver.end()

    # No barge-in: scope not cancelled, brain not abandoned, no BARGED_IN record.
    assert driver.brains[0].abandoned is False
    assert "barge_in" not in driver.transport.flushes
    records = driver.session.records
    assert len(records) == 1
    rec = records[0]
    assert rec.ended_reason is TurnEndReason.COMPLETED
    assert rec.interrupted is False
    assert rec.assistant_heard_text == text


# ---------------------------------------------------------------------------
# Sanity: custom ledger factory is honoured
# ---------------------------------------------------------------------------


async def test_ledger_factory_is_used():
    built: list[HeardSpanLedger] = []
    clock = VirtualClock()
    ctx = make_ctx()
    transport = FakeAudioTransport(MEDIA)
    turns = FakeTurnDetection([(0, turn_event(TurnEventKind.ENDPOINT_DETECTED))])
    tts = FakeTTS(clock)
    stt = FakeSTT(final=final_transcript("hi"))

    def brain_factory() -> FakeBrain:
        return FakeBrain(clock, text="hello there", delay_ms=0)

    def ledger_factory(call_id: str, turn_index: int) -> HeardSpanLedger:
        led = HeardSpanLedger(call_id, turn_index=turn_index)
        built.append(led)
        return led

    session = StreamingCallSession(
        ctx,
        transport=transport,
        stt=stt,
        turns=turns,
        tts=tts,
        brain_factory=brain_factory,
        policy=InterruptionPolicy(),
        tracer=StreamingCallTracer(ctx.call_id),
        clock=clock,
        ledger_factory=ledger_factory,
    )
    run_task = asyncio.create_task(session.run())
    await asyncio.sleep(0)
    await transport.push_inbound(make_frame(0))
    for _ in range(5):
        await asyncio.sleep(0)
    await transport.end_inbound()
    for _ in range(30):
        if run_task.done():
            break
        await clock.advance(20)
        await asyncio.sleep(0)
    await asyncio.wait_for(run_task, timeout=2.0)

    assert len(built) == 1
    assert built[0].turn_index == 1


# ---------------------------------------------------------------------------
# Fix B — unexpected brain exception leaves no dirty state / no leaked task
# ---------------------------------------------------------------------------


async def test_unexpected_brain_exception_cleans_up():
    # Behavior decision: an UNEXPECTED exception (not BrainEventKind.ERROR) is
    # PROPAGATED out of run() — it signals a real bug, not a recoverable brain
    # failure — but only AFTER the turn state is cleaned up and the STT task is
    # torn down, so nothing is left dirty or leaked.
    class _RaisingBrain:
        def respond(self, turn, ctx, scope):
            return self._gen()

        async def _gen(self):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

    clock = VirtualClock()
    ctx = make_ctx()
    transport = FakeAudioTransport(MEDIA)
    turns = FakeTurnDetection([(0, turn_event(TurnEventKind.ENDPOINT_DETECTED))])
    tts = FakeTTS(clock)
    stt = FakeSTT(final=final_transcript("hi"))

    session = StreamingCallSession(
        ctx,
        transport=transport,
        stt=stt,
        turns=turns,
        tts=tts,
        brain_factory=_RaisingBrain,
        policy=InterruptionPolicy(),
        tracer=StreamingCallTracer(ctx.call_id),
        clock=clock,
    )
    run_task = asyncio.create_task(session.run())
    await asyncio.sleep(0)
    await transport.push_inbound(make_frame(0))
    for _ in range(5):
        await asyncio.sleep(0)
    await transport.end_inbound()
    for _ in range(30):
        if run_task.done():
            break
        await clock.advance(20)
        await asyncio.sleep(0)

    # The unexpected exception propagates out of run().
    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(run_task, timeout=2.0)

    # No dirty state remains after the unexpected exception.
    assert session._thinking is False
    assert session._speaking is False
    assert session._active_scope is None
    assert session._active_ledger is None
    assert session._assistant_task is None
    # No record committed for the failed turn.
    assert session.records == []
    # The background STT task did not leak.
    assert session._stt_task is not None
    assert session._stt_task.done()
