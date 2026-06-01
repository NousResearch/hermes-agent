"""Slice 6 core: AiortcStreamingTransport + StreamingPipeline (pure asyncio).

These tests exercise the transport contract and the process_pcm16 media seam
entirely without aiortc/av installed. The transport uses an injected outbound
sink (async callable, optional .drop hook) and an injected Clock — no walltime.
"""
from __future__ import annotations

import asyncio

import pytest

from gateway.calls.native.streaming.aiortc_transport import (
    AiortcStreamingTransport,
    StreamingPipeline,
    build_streaming_pipeline,
)
from gateway.calls.native.streaming.clock import VirtualClock
from gateway.calls.native.streaming.types import (
    AudioFrame,
    FlushResult,
    MediaFormat,
)

pytestmark = pytest.mark.asyncio

MEDIA_16K = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)


def make_frame(seq: int, duration_ms: int = 20, media: MediaFormat = MEDIA_16K) -> AudioFrame:
    samples = int(media.sample_rate * duration_ms / 1000)
    pcm = b"\x00" * samples * 2
    return AudioFrame(pcm16=pcm, media=media, timestamp_ms=seq * duration_ms, seq=seq)


class RecordingSink:
    """Async callable outbound sink with an optional drop() hook."""

    def __init__(self, *, with_drop: bool = True) -> None:
        self.sent: list[AudioFrame] = []
        self.drops = 0
        if with_drop:
            async def _drop() -> None:
                self.drops += 1

            self.drop = _drop  # type: ignore[attr-defined]

    async def __call__(self, frame: AudioFrame) -> None:
        self.sent.append(frame)


# ---------------------------------------------------------------------------
# Scenario 1 — push_inbound / inbound / close
# ---------------------------------------------------------------------------


async def test_inbound_yields_then_ends_on_close():
    clock = VirtualClock()
    sink = RecordingSink()
    transport = AiortcStreamingTransport(MEDIA_16K, clock=clock, outbound_sink=sink)

    received: list[AudioFrame] = []

    async def consume() -> None:
        async for f in transport.inbound():
            received.append(f)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    f = make_frame(0)
    await transport.push_inbound(f)
    await asyncio.sleep(0)
    assert received == [f]

    await transport.close()
    await asyncio.wait_for(task, timeout=1.0)
    assert task.done()


async def test_transport_media_property():
    transport = AiortcStreamingTransport(MEDIA_16K, clock=VirtualClock(), outbound_sink=RecordingSink())
    assert transport.media is MEDIA_16K


# ---------------------------------------------------------------------------
# Scenario 2 — emit_outbound awaits sink + records pending
# ---------------------------------------------------------------------------


async def test_emit_outbound_calls_sink_and_records_pending():
    sink = RecordingSink()
    transport = AiortcStreamingTransport(MEDIA_16K, clock=VirtualClock(), outbound_sink=sink)

    f0 = make_frame(0)
    f1 = make_frame(1)
    await transport.emit_outbound(f0)
    await transport.emit_outbound(f1)

    assert sink.sent == [f0, f1]


# ---------------------------------------------------------------------------
# Scenario 3 — flush_outbound clears pending, fires drop, returns FlushResult
# ---------------------------------------------------------------------------


async def test_flush_outbound_returns_flushresult_and_fires_drop():
    sink = RecordingSink()
    transport = AiortcStreamingTransport(MEDIA_16K, clock=VirtualClock(), outbound_sink=sink)

    n = 3
    for i in range(n):
        await transport.emit_outbound(make_frame(i))

    result = await transport.flush_outbound("barge")
    assert isinstance(result, FlushResult)
    assert result.dropped_frames == n
    assert result.dropped_ms == 20 * n
    assert result.last_sent_mark is None
    assert sink.drops == 1

    # Pending cleared: a second flush drops nothing.
    result2 = await transport.flush_outbound("barge")
    assert result2.dropped_frames == 0
    assert result2.dropped_ms == 0


async def test_flush_outbound_without_drop_hook():
    sink = RecordingSink(with_drop=False)
    assert not hasattr(sink, "drop")
    transport = AiortcStreamingTransport(MEDIA_16K, clock=VirtualClock(), outbound_sink=sink)
    await transport.emit_outbound(make_frame(0))
    result = await transport.flush_outbound("barge")
    assert result.dropped_frames == 1


# ---------------------------------------------------------------------------
# Scenario 4 — process_pcm16 resamples + pushes one AudioFrame
# ---------------------------------------------------------------------------


class SpyTransport:
    """Records push_inbound frames; satisfies the subset the pipeline needs."""

    def __init__(self, media: MediaFormat) -> None:
        self._media = media
        self.pushed: list[AudioFrame] = []
        self.closed = False

    @property
    def media(self) -> MediaFormat:
        return self._media

    async def push_inbound(self, frame: AudioFrame) -> None:
        self.pushed.append(frame)

    def inbound(self):
        async def _gen():
            if False:
                yield  # pragma: no cover
            return
        return _gen()

    async def emit_outbound(self, frame: AudioFrame) -> None:  # pragma: no cover
        pass

    async def flush_outbound(self, reason: str):  # pragma: no cover
        return FlushResult(0, 0, None)

    async def close(self) -> None:
        self.closed = True


class _NullSession:
    async def run(self) -> None:
        # Idle until cancelled / completed; the pipeline awaits it on aclose.
        return None


async def test_process_pcm16_resamples_to_16k_and_pushes_frame():
    clock = VirtualClock()
    transport = SpyTransport(MEDIA_16K)
    pipe = StreamingPipeline(
        media=MEDIA_16K,
        session=_NullSession(),
        transport=transport,
        clock=clock,
    )

    # 48kHz, 20ms mono = 960 samples = 1920 bytes.
    samples_48k = int(48000 * 20 / 1000)
    pcm_48k = b"\x00" * samples_48k * 2

    ack = await pipe.process_pcm16(call_id="c", pcm16=pcm_48k, sample_rate=48000)
    assert ack  # truthy ack returned promptly

    assert len(transport.pushed) == 1
    frame = transport.pushed[0]
    assert frame.media.sample_rate == 16000
    assert frame.seq == 0
    assert frame.timestamp_ms == clock.now_ms()
    # Resample 48k->16k reduces byte count ~3x.
    assert len(frame.pcm16) < len(pcm_48k)
    assert abs(len(frame.pcm16) - len(pcm_48k) // 3) <= 8

    await pipe.aclose()
    assert transport.closed is True


async def test_process_pcm16_monotonic_seq_and_passthrough_16k():
    clock = VirtualClock()
    transport = SpyTransport(MEDIA_16K)
    pipe = StreamingPipeline(
        media=MEDIA_16K,
        session=_NullSession(),
        transport=transport,
        clock=clock,
    )

    samples_16k = int(16000 * 20 / 1000)
    pcm_16k = b"\x01\x00" * samples_16k

    await pipe.process_pcm16(call_id="c", pcm16=pcm_16k, sample_rate=16000)
    await pipe.process_pcm16(call_id="c", pcm16=pcm_16k, sample_rate=16000)

    assert [f.seq for f in transport.pushed] == [0, 1]
    # 16k passthrough: no resample, identical byte count.
    assert len(transport.pushed[0].pcm16) == len(pcm_16k)

    await pipe.aclose()


async def test_pipeline_is_streaming_flag():
    pipe = StreamingPipeline(
        media=MEDIA_16K,
        session=_NullSession(),
        transport=SpyTransport(MEDIA_16K),
        clock=VirtualClock(),
    )
    assert pipe.is_streaming is True


# ---------------------------------------------------------------------------
# Scenario 5 — full fake-port seam end to end
# ---------------------------------------------------------------------------


async def test_full_fake_port_seam_emits_outbound():
    from gateway.calls.native.streaming.fakes import (
        FakeBrain,
        FakeSTT,
        FakeTTS,
        FakeTurnDetection,
    )
    from gateway.calls.native.streaming.interruption import InterruptionPolicy
    from gateway.calls.native.streaming.session import StreamingCallSession
    from gateway.calls.native.streaming.tracer import StreamingCallTracer
    from gateway.calls.native.streaming.types import (
        StreamingCallContext,
        TranscriptEvent,
        TranscriptKind,
        TurnEvent,
        TurnEventKind,
    )

    clock = VirtualClock()
    sink = RecordingSink()
    transport = AiortcStreamingTransport(MEDIA_16K, clock=clock, outbound_sink=sink)
    ctx = StreamingCallContext(
        call_id="c", contact_id="ct", session_id="s", media=MEDIA_16K
    )
    stt = FakeSTT(final=TranscriptEvent(call_id="c", kind=TranscriptKind.FINAL, text="hello"))
    turns = FakeTurnDetection([(0, TurnEvent(call_id="c", kind=TurnEventKind.ENDPOINT_DETECTED, at_ms=0))])
    tts = FakeTTS(clock)

    def brain_factory() -> FakeBrain:
        return FakeBrain(clock, text="hi there", delay_ms=0)

    session = StreamingCallSession(
        ctx,
        transport=transport,
        stt=stt,
        turns=turns,
        tts=tts,
        brain_factory=brain_factory,
        policy=InterruptionPolicy(),
        tracer=StreamingCallTracer("c"),
        clock=clock,
    )
    pipe = StreamingPipeline(
        media=MEDIA_16K, session=session, transport=transport, clock=clock
    )

    # Feed one inbound frame (seq 0 → ENDPOINT_DETECTED → assistant turn).
    samples = int(16000 * 20 / 1000)
    await pipe.process_pcm16(call_id="c", pcm16=b"\x00" * samples * 2, sample_rate=16000)

    # Advance the clock so brain/TTS generators make progress.
    for _ in range(20):
        await clock.advance(20)
        await asyncio.sleep(0)

    # End inbound, then drain the virtual clock until the session task settles
    # (the in-flight assistant turn sleeps on the VirtualClock).
    await transport.close()
    for _ in range(200):
        if pipe._task is not None and pipe._task.done():
            break
        await clock.advance(20)
        await asyncio.sleep(0)
    await pipe.aclose()

    # The session task completed and outbound audio reached the sink.
    assert pipe._task is None  # aclose() awaited + cleared the task
    assert len(sink.sent) > 0
    assert len(session.records) == 1


# ---------------------------------------------------------------------------
# Scenario 6 — barge-in drives flush + drop
# ---------------------------------------------------------------------------


async def test_barge_in_fires_flush_and_drop():
    from gateway.calls.native.streaming.fakes import (
        FakeBrain,
        FakeSTT,
        FakeTTS,
        FakeTurnDetection,
    )
    from gateway.calls.native.streaming.interruption import InterruptionPolicy
    from gateway.calls.native.streaming.session import StreamingCallSession
    from gateway.calls.native.streaming.tracer import StreamingCallTracer
    from gateway.calls.native.streaming.types import (
        InterruptionParams,
        StreamingCallContext,
        TranscriptEvent,
        TranscriptKind,
        TurnEndReason,
        TurnEvent,
        TurnEventKind,
    )

    clock = VirtualClock()
    sink = RecordingSink()
    transport = AiortcStreamingTransport(MEDIA_16K, clock=clock, outbound_sink=sink)
    params = InterruptionParams(min_speech_ms=40, min_words=2)
    ctx = StreamingCallContext(
        call_id="c", contact_id="ct", session_id="s", media=MEDIA_16K, interruption=params
    )
    stt = FakeSTT(
        partials=[TranscriptEvent(call_id="c", kind=TranscriptKind.PARTIAL, text="hold on")],
        final=TranscriptEvent(call_id="c", kind=TranscriptKind.FINAL, text="q one"),
    )
    turns = FakeTurnDetection([
        (0, TurnEvent(call_id="c", kind=TurnEventKind.ENDPOINT_DETECTED, at_ms=0)),
        (1, TurnEvent(call_id="c", kind=TurnEventKind.USER_SPEECH_STARTED, at_ms=0)),
        (2, TurnEvent(call_id="c", kind=TurnEventKind.USER_SPEECH_STOPPED, at_ms=0)),
    ])
    tts = FakeTTS(clock, frames_per_word=10)

    def brain_factory() -> FakeBrain:
        return FakeBrain(clock, text="one two three four five six", delay_ms=0)

    session = StreamingCallSession(
        ctx,
        transport=transport,
        stt=stt,
        turns=turns,
        tts=tts,
        brain_factory=brain_factory,
        policy=InterruptionPolicy(),
        tracer=StreamingCallTracer("c"),
        clock=clock,
    )
    pipe = StreamingPipeline(
        media=MEDIA_16K, session=session, transport=transport, clock=clock
    )
    samples = int(16000 * 20 / 1000)
    pcm = b"\x00" * samples * 2

    await pipe.process_pcm16(call_id="c", pcm16=pcm, sample_rate=16000)  # seq 0 → turn
    for _ in range(11):
        await clock.advance(20)
        await asyncio.sleep(0)
    await pipe.process_pcm16(call_id="c", pcm16=pcm, sample_rate=16000)  # seq 1 → speech started
    await clock.advance(params.min_speech_ms + 20)
    await asyncio.sleep(0)
    await pipe.process_pcm16(call_id="c", pcm16=pcm, sample_rate=16000)  # seq 2 → INTERRUPT

    await transport.close()
    for _ in range(200):
        if pipe._task is not None and pipe._task.done():
            break
        await clock.advance(20)
        await asyncio.sleep(0)
    await pipe.aclose()

    assert sink.drops >= 1
    assert len(session.records) == 1
    assert session.records[0].ended_reason is TurnEndReason.BARGED_IN


# ---------------------------------------------------------------------------
# build_streaming_pipeline factory
# ---------------------------------------------------------------------------


async def test_build_streaming_pipeline_fake():
    pipe = build_streaming_pipeline({}, cognitive="fake")
    assert isinstance(pipe, StreamingPipeline)
    assert pipe.is_streaming is True
    samples = int(16000 * 20 / 1000)
    ack = await pipe.process_pcm16(call_id="c", pcm16=b"\x00" * samples * 2, sample_rate=16000)
    assert ack
    await pipe.aclose()


async def test_build_streaming_pipeline_real_not_implemented():
    with pytest.raises(NotImplementedError):
        build_streaming_pipeline({}, cognitive="real")


# ---------------------------------------------------------------------------
# Slice 6b carry-in fixes (B1/B2/B3/I3/M4)
# ---------------------------------------------------------------------------


async def test_pipeline_exposes_transport():
    transport = SpyTransport(MEDIA_16K)
    pipe = StreamingPipeline(
        media=MEDIA_16K,
        session=_NullSession(),
        transport=transport,
        clock=VirtualClock(),
    )
    assert pipe.transport is pipe._transport
    assert pipe.transport is transport


async def test_set_outbound_sink_replaces_and_drop_hook_fires():
    initial = RecordingSink(with_drop=False)
    transport = AiortcStreamingTransport(
        MEDIA_16K, clock=VirtualClock(), outbound_sink=initial
    )

    new_sink = RecordingSink(with_drop=False)
    drop_calls = {"count": 0}

    async def drop_spy() -> None:
        drop_calls["count"] += 1

    transport.set_outbound_sink(new_sink, drop=drop_spy)

    f0 = make_frame(0)
    await transport.emit_outbound(f0)
    # Routed to the new sink, not the original.
    assert new_sink.sent == [f0]
    assert initial.sent == []

    result = await transport.flush_outbound("x")
    assert isinstance(result, FlushResult)
    assert result.dropped_frames == 1
    assert result.dropped_ms == 20
    assert drop_calls["count"] == 1


async def test_set_outbound_sink_sync_drop_hook_fires():
    transport = AiortcStreamingTransport(
        MEDIA_16K, clock=VirtualClock(), outbound_sink=RecordingSink(with_drop=False)
    )
    drop_calls = {"count": 0}

    def drop_spy() -> int:
        drop_calls["count"] += 1
        return 0

    transport.set_outbound_sink(RecordingSink(with_drop=False), drop=drop_spy)
    await transport.emit_outbound(make_frame(0))
    await transport.flush_outbound("x")
    assert drop_calls["count"] == 1


async def test_flush_drop_fallback_to_sink_attribute():
    # No set_outbound_sink: ctor sink with a .drop attr still fires (back-compat).
    sink = RecordingSink(with_drop=True)
    transport = AiortcStreamingTransport(
        MEDIA_16K, clock=VirtualClock(), outbound_sink=sink
    )
    await transport.emit_outbound(make_frame(0))
    await transport.flush_outbound("x")
    assert sink.drops == 1


async def test_build_streaming_pipeline_sink_none_uses_noop():
    pipe = build_streaming_pipeline({}, cognitive="fake", sink=None)
    assert isinstance(pipe, StreamingPipeline)
    # Emitting before set_outbound_sink does not crash (no-op sink buffers).
    await pipe.transport.emit_outbound(make_frame(0))
    result = await pipe.transport.flush_outbound("x")
    assert isinstance(result, FlushResult)
    assert result.dropped_frames == 1
    await pipe.aclose()


class _RunningSession:
    """A session whose run() blocks until cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self) -> None:
        self.started.set()
        await asyncio.Event().wait()


async def test_aclose_idempotent_and_abort():
    # Idempotent: second aclose() is a no-op (no re-raise).
    transport = SpyTransport(MEDIA_16K)
    pipe = StreamingPipeline(
        media=MEDIA_16K,
        session=_NullSession(),
        transport=transport,
        clock=VirtualClock(),
    )
    samples = int(16000 * 20 / 1000)
    await pipe.process_pcm16(call_id="c", pcm16=b"\x00" * samples * 2, sample_rate=16000)
    await pipe.aclose()
    assert pipe._task is None
    await pipe.aclose()  # second call: no raise
    assert pipe._task is None

    # abort=True cancels a still-running session task and returns without raising.
    transport2 = SpyTransport(MEDIA_16K)
    session = _RunningSession()
    pipe2 = StreamingPipeline(
        media=MEDIA_16K,
        session=session,
        transport=transport2,
        clock=VirtualClock(),
    )
    await pipe2.process_pcm16(call_id="c", pcm16=b"\x00" * samples * 2, sample_rate=16000)
    await asyncio.wait_for(session.started.wait(), timeout=1.0)
    await pipe2.aclose(abort=True)
    assert pipe2._task is None
