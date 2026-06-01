"""Isolation tests for gateway/calls/native/streaming/fakes.py (WP4).

TDD: these are written before fakes.py exists.
"""
from __future__ import annotations

import asyncio

import pytest

from gateway.calls.native.streaming.cancellation import CancellationScope
from gateway.calls.native.streaming.clock import VirtualClock
from gateway.calls.native.streaming.fakes import (
    FakeAudioTransport,
    FakeBrain,
    FakeSTT,
    FakeTTS,
    FakeTurnDetection,
)
from gateway.calls.native.streaming.ports import (
    AudioTransportPort,
    HermesBrainPort,
    SpeechToTextPort,
    TextToSpeechPort,
    TurnDetectionPort,
)
from gateway.calls.native.streaming.types import (
    AudioFrame,
    BrainEventKind,
    MediaFormat,
    StreamingCallContext,
    TranscriptEvent,
    TranscriptKind,
    TtsEventKind,
    TurnEvent,
    TurnEventKind,
)

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MEDIA = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)
CALL_ID = "call-test"


def make_ctx() -> StreamingCallContext:
    return StreamingCallContext(
        call_id=CALL_ID,
        contact_id="contact-1",
        session_id="session-1",
        media=MEDIA,
    )


def make_frame(seq: int = 0, duration_ms: int = 20) -> AudioFrame:
    """Build a synthetic AudioFrame with the given seq."""
    samples = int(MEDIA.sample_rate * duration_ms / 1000)
    pcm = b"\x00" * samples * 2
    return AudioFrame(pcm16=pcm, media=MEDIA, timestamp_ms=seq * duration_ms, seq=seq)


def make_transcript(text: str = "hello world", kind: TranscriptKind = TranscriptKind.FINAL) -> TranscriptEvent:
    return TranscriptEvent(call_id=CALL_ID, kind=kind, text=text)


# ---------------------------------------------------------------------------
# 1. Protocol compliance (runtime_checkable — name-only check)
# ---------------------------------------------------------------------------


async def test_protocol_compliance():
    clock = VirtualClock()
    transport = FakeAudioTransport(MEDIA)
    turn_det = FakeTurnDetection([])
    stt = FakeSTT()
    tts = FakeTTS(clock)
    brain = FakeBrain(clock, text="hi")

    assert isinstance(transport, AudioTransportPort)
    assert isinstance(turn_det, TurnDetectionPort)
    assert isinstance(stt, SpeechToTextPort)
    assert isinstance(tts, TextToSpeechPort)
    assert isinstance(brain, HermesBrainPort)


# ---------------------------------------------------------------------------
# 2. FakeTTS: marks + DONE
# ---------------------------------------------------------------------------


async def _collect_tts(clock: VirtualClock, tts: FakeTTS, text: str, ctx: StreamingCallContext, scope: CancellationScope) -> list:
    """Drive a FakeTTS.synthesize generator, advancing the clock step-by-step."""
    events: list = []

    async def consumer():
        async for event in tts.synthesize(text, ctx, scope):
            events.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let the task start and register its first clock.sleep waiter

    # Advance one frame_ms at a time so each sleep() resolves in order.
    words = len(text.split())
    total_steps = words * tts._frames_per_word + 2  # +2 for headroom
    for _ in range(total_steps):
        if task.done():
            break
        await clock.advance(tts._frame_ms)
        await asyncio.sleep(0)

    if not task.done():
        # Extra headroom just in case
        await clock.advance(100)
        await asyncio.sleep(0)

    await task
    return events


async def test_tts_marks_offsets_and_done():
    """MARK.char_offset is strictly increasing; text_so_far == full_text[:offset]; ends with DONE."""
    clock = VirtualClock()
    tts = FakeTTS(clock, frames_per_word=2, frame_ms=20)
    ctx = make_ctx()
    scope = CancellationScope()

    text = "hello world foo"
    events = await _collect_tts(clock, tts, text, ctx, scope)

    mark_events = [e for e in events if e.kind == TtsEventKind.MARK]
    done_events = [e for e in events if e.kind == TtsEventKind.DONE]

    words = text.split()
    assert len(mark_events) == len(words), "One MARK per word"
    assert len(done_events) == 1, "Exactly one DONE at end"

    # DONE must be last
    assert events[-1].kind == TtsEventKind.DONE

    # Offsets strictly increasing; text_so_far == text[:offset]
    prev_offset = -1
    for mark_ev in mark_events:
        mark = mark_ev.mark
        assert mark is not None
        assert mark.char_offset > prev_offset, "char_offset must be strictly increasing"
        assert mark.text_so_far == text[: mark.char_offset], (
            f"text_so_far mismatch: {mark.text_so_far!r} != {text[:mark.char_offset]!r}"
        )
        prev_offset = mark.char_offset

    # Final offset equals len(text)
    assert mark_events[-1].mark.char_offset == len(text), "Last MARK should cover the full text"


async def test_tts_cancelled_before_start():
    """Cancelling scope before synthesize starts → first event is CANCELLED, no DONE."""
    clock = VirtualClock()
    tts = FakeTTS(clock)
    ctx = make_ctx()
    scope = CancellationScope()
    scope.cancel("test")

    events: list = []
    async for event in tts.synthesize("hello world", ctx, scope):
        events.append(event)

    assert len(events) == 1, f"Expected exactly 1 event, got {len(events)}: {events}"
    assert events[0].kind == TtsEventKind.CANCELLED
    assert not any(e.kind == TtsEventKind.DONE for e in events)


async def test_tts_cancelled_mid_synthesis():
    """Cancelling scope mid-synthesis stops early and emits CANCELLED."""
    clock = VirtualClock()
    tts = FakeTTS(clock, frames_per_word=3, frame_ms=20)
    ctx = make_ctx()
    scope = CancellationScope()

    text = "one two three four five"
    events: list = []

    async def consumer():
        async for event in tts.synthesize(text, ctx, scope):
            events.append(event)
            # Cancel after the first MARK (first word done)
            if event.kind == TtsEventKind.MARK:
                scope.cancel("barge-in")

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let task register its first waiter
    # Advance step-by-step
    for _ in range(30):
        if task.done():
            break
        await clock.advance(20)
        await asyncio.sleep(0)
    await task

    assert not any(e.kind == TtsEventKind.DONE for e in events), "Should not reach DONE after cancel"
    assert any(e.kind == TtsEventKind.CANCELLED for e in events), "Should emit CANCELLED"


# ---------------------------------------------------------------------------
# 3. FakeBrain: delay + cancellation
# ---------------------------------------------------------------------------


async def test_brain_cancelled_before_advance():
    """Brain with delay_ms>0: cancel before advancing clock → abandoned=True, no events."""
    clock = VirtualClock()
    brain = FakeBrain(clock, text="the answer", delay_ms=200)
    ctx = make_ctx()
    scope = CancellationScope()
    turn = make_transcript()

    events: list = []

    async def consumer():
        async for event in brain.respond(turn, ctx, scope):
            events.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let it register the first sleep
    scope.cancel("interrupted")
    await clock.advance(200)
    await asyncio.sleep(0)
    await task

    assert brain.started is True
    assert brain.abandoned is True
    assert events == [], f"Should yield nothing when cancelled: {events}"


async def test_brain_no_cancel_yields_final_text():
    """Brain without cancel yields exactly one FINAL_TEXT event with configured text."""
    clock = VirtualClock()
    brain = FakeBrain(clock, text="the answer", delay_ms=100)
    ctx = make_ctx()
    scope = CancellationScope()
    turn = make_transcript()

    events: list = []

    async def consumer():
        async for event in brain.respond(turn, ctx, scope):
            events.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let task register its first sleep waiter
    await clock.advance(200)
    await asyncio.sleep(0)
    await task

    assert brain.started is True
    assert brain.abandoned is False
    assert len(events) == 1
    assert events[0].kind == BrainEventKind.FINAL_TEXT
    assert events[0].text == "the answer"


async def test_brain_zero_delay():
    """Brain with delay_ms=0 immediately yields FINAL_TEXT."""
    clock = VirtualClock()
    brain = FakeBrain(clock, text="instant", delay_ms=0)
    ctx = make_ctx()
    scope = CancellationScope()
    turn = make_transcript()

    events: list = []
    async for event in brain.respond(turn, ctx, scope):
        events.append(event)

    assert len(events) == 1
    assert events[0].kind == BrainEventKind.FINAL_TEXT
    assert events[0].text == "instant"


# ---------------------------------------------------------------------------
# 4. FakeAudioTransport: flush accounting
# ---------------------------------------------------------------------------


async def test_transport_flush_counts_dropped():
    """flush_outbound returns dropped count and clears pending; second flush returns 0."""
    transport = FakeAudioTransport(MEDIA)
    assert transport.media is MEDIA

    frame1 = make_frame(seq=0)
    frame2 = make_frame(seq=1)
    frame3 = make_frame(seq=2)

    await transport.emit_outbound(frame1)
    await transport.emit_outbound(frame2)
    await transport.emit_outbound(frame3)

    result1 = await transport.flush_outbound("barge-in")
    assert result1.dropped_frames == 3
    assert result1.dropped_ms == frame1.duration_ms + frame2.duration_ms + frame3.duration_ms
    assert result1.last_sent_mark is None
    assert transport.flushes == ["barge-in"]

    # Pending should be cleared
    result2 = await transport.flush_outbound("second")
    assert result2.dropped_frames == 0
    assert result2.dropped_ms == 0
    assert transport.flushes == ["barge-in", "second"]


async def test_transport_emit_records_sent():
    """emit_outbound appends to both sent and _pending."""
    transport = FakeAudioTransport(MEDIA)
    frame = make_frame(seq=0)
    await transport.emit_outbound(frame)
    assert transport.sent == [frame]
    assert transport._pending == [frame]


async def test_transport_inbound_queue():
    """push_inbound / inbound() / end_inbound() roundtrip."""
    transport = FakeAudioTransport(MEDIA)
    frame = make_frame(seq=0)

    received: list[AudioFrame] = []

    async def reader():
        async for f in transport.inbound():
            received.append(f)

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    await transport.push_inbound(frame)
    await asyncio.sleep(0)
    await transport.end_inbound()
    await asyncio.sleep(0)
    await task

    assert received == [frame]


async def test_transport_close():
    """close() marks the transport as closed."""
    transport = FakeAudioTransport(MEDIA)
    assert not transport._closed
    await transport.close()
    assert transport._closed


# ---------------------------------------------------------------------------
# 5. FakeSTT: finalize + events
# ---------------------------------------------------------------------------


async def test_stt_finalize_returns_configured():
    """finalize() returns the configured final transcript or None."""
    final = make_transcript("yes we can")
    stt = FakeSTT(final=final)

    result = await stt.finalize()
    assert result is final


async def test_stt_finalize_none():
    """finalize() returns None when no final configured (Scenario G)."""
    stt = FakeSTT()
    result = await stt.finalize()
    assert result is None


async def test_stt_events_yields_partials_and_stops():
    """events() yields configured partials in order then stops."""
    p1 = make_transcript("he", TranscriptKind.PARTIAL)
    p2 = make_transcript("hello", TranscriptKind.PARTIAL)
    stt = FakeSTT(partials=[p1, p2])

    collected: list[TranscriptEvent] = []
    async for event in stt.events():
        collected.append(event)

    assert collected == [p1, p2]


async def test_stt_lifecycle_methods():
    """start/push/cancel/close are callable without error."""
    stt = FakeSTT()
    ctx = make_ctx()
    frame = make_frame()

    await stt.start(ctx)
    await stt.push(frame)
    await stt.cancel()
    await stt.close()

    assert stt._started
    assert stt._cancelled
    assert stt._closed


# ---------------------------------------------------------------------------
# 6. FakeTurnDetection: observe scripted by frame.seq
# ---------------------------------------------------------------------------


async def test_turn_detection_scripted_events():
    """observe() returns scripted TurnEvents keyed by frame.seq; empty tuple otherwise."""
    event_at_3 = TurnEvent(call_id=CALL_ID, kind=TurnEventKind.ENDPOINT_DETECTED, at_ms=60)
    event_at_5 = TurnEvent(call_id=CALL_ID, kind=TurnEventKind.USER_SPEECH_STOPPED, at_ms=100)

    td = FakeTurnDetection([(3, event_at_3), (5, event_at_5)])

    frame0 = make_frame(seq=0)
    frame3 = make_frame(seq=3)
    frame5 = make_frame(seq=5)
    frame9 = make_frame(seq=9)

    assert await td.observe(frame0) == ()
    assert await td.observe(frame3) == (event_at_3,)
    assert await td.observe(frame5) == (event_at_5,)
    assert await td.observe(frame9) == ()


async def test_turn_detection_multiple_events_same_seq():
    """Multiple events scripted for the same seq all returned together."""
    ev_a = TurnEvent(call_id=CALL_ID, kind=TurnEventKind.USER_SPEECH_STARTED, at_ms=10)
    ev_b = TurnEvent(call_id=CALL_ID, kind=TurnEventKind.POSSIBLE_BACKCHANNEL, at_ms=10)

    td = FakeTurnDetection([(2, ev_a), (2, ev_b)])

    frame2 = make_frame(seq=2)
    result = await td.observe(frame2)
    assert result == (ev_a, ev_b)


async def test_turn_detection_reset_is_noop():
    """reset() is callable without error."""
    td = FakeTurnDetection([])
    td.reset()  # just must not raise
