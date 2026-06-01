"""Slice 6 core: aiortc AudioTransportPort bridge + StreamingPipeline.

Pure-asyncio core for the SimpleX streaming voice path. There is intentionally
NO top-level ``import aiortc`` / ``import av`` here — this module (and the whole
``streaming`` package) imports cleanly on CI where the optional
``simplex-native-calls`` extra is absent. The live aiortc PCM track is built in
Slice 6b.

``AiortcStreamingTransport`` satisfies ``AudioTransportPort`` (ports.py) using an
injected ``outbound_sink`` (an async callable with an optional ``.drop`` hook) so
the same transport can be unit-tested with a recording sink and later wired to a
real aiortc track. Inbound audio flows through an ``asyncio.Queue`` drained to a
``None`` sentinel — identical to ``FakeAudioTransport``.

``StreamingPipeline`` exposes ``process_pcm16`` — the entry point the gateway
calls with raw SimpleX PCM16 (48kHz). It lazily starts the session task on the
first frame, resamples to the ports' 16kHz mono format via stdlib
``audioop.ratecv`` (imported lazily inside ``process_pcm16``; py3.11 stdlib),
pushes one ``AudioFrame`` (timestamped from the injected ``Clock``), and returns
a small non-blocking ack.

No ``time.*`` / ``asyncio.sleep`` appears here: all timing is the injected
``Clock.now_ms()`` (no-walltime gate scopes ``streaming/**``).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from typing import Any, Protocol

from .clock import Clock
from .types import AudioFrame, FlushResult, MediaFormat

__all__ = [
    "AiortcStreamingTransport",
    "StreamingPipeline",
    "ProcessAck",
    "build_streaming_pipeline",
]


class OutboundSink(Protocol):
    """Typed outbound sink contract (M4).

    An async callable consuming one ``AudioFrame``. The barge-in ``drop`` hook is
    a SEPARATE optional callable passed to ``set_outbound_sink`` — NOT a required
    attribute on the sink itself (see B2).
    """

    def __call__(self, frame: AudioFrame) -> Awaitable[None]: ...


# Back-compat alias for the pre-M4 private name.
_OutboundSink = OutboundSink


async def _noop_sink(frame: AudioFrame) -> None:
    """Module-level no-op outbound sink (B3).

    Used on the live engine path before ``start()`` swaps in the real aiortc
    track via ``set_outbound_sink``: frames buffer harmlessly (are discarded)
    rather than landing in a throwaway recording sink.
    """
    return None


# ---------------------------------------------------------------------------
# AiortcStreamingTransport
# ---------------------------------------------------------------------------


class AiortcStreamingTransport:
    """AudioTransportPort backed by an injected async outbound sink.

    No aiortc import: the sink is the seam. In CI/tests the sink is a recording
    buffer; in production (Slice 6b) it enqueues PCM onto the live aiortc track.
    """

    def __init__(
        self,
        media: MediaFormat,
        *,
        clock: Clock,
        outbound_sink: OutboundSink,
    ) -> None:
        self._media = media
        self._clock = clock
        self._outbound_sink: OutboundSink = outbound_sink
        # Barge-in drop hook is a SEPARATE field (B2), not a sink attribute.
        self._outbound_drop: Any | None = None
        self._inbound: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._pending: list[AudioFrame] = []
        self._closed = False

    def set_outbound_sink(
        self, sink: OutboundSink, *, drop: Any | None = None
    ) -> None:
        """Replace the outbound sink + its barge-in drop hook (B2/M4).

        ``drop`` is stored as a SEPARATE field so a bound method (``track.enqueue``)
        — which has no ``.drop`` attribute — still gets a working flush hook.
        """
        self._outbound_sink = sink
        self._outbound_drop = drop

    @property
    def media(self) -> MediaFormat:
        return self._media

    async def push_inbound(self, frame: AudioFrame) -> None:
        await self._inbound.put(frame)

    def inbound(self) -> AsyncIterator[AudioFrame]:
        return self._inbound_gen()

    async def _inbound_gen(self) -> AsyncIterator[AudioFrame]:
        while True:
            item = await self._inbound.get()
            if item is None:
                return
            yield item

    async def emit_outbound(self, frame: AudioFrame) -> None:
        self._pending.append(frame)
        await self._outbound_sink(frame)

    async def flush_outbound(self, reason: str) -> FlushResult:
        dropped_frames = len(self._pending)
        dropped_ms = sum(f.duration_ms for f in self._pending)
        self._pending.clear()
        # Prefer the explicit drop hook (B2); fall back to a ctor-sink .drop attr.
        drop = self._outbound_drop or getattr(self._outbound_sink, "drop", None)
        if drop is not None:
            result = drop()
            if asyncio.iscoroutine(result):
                await result
        return FlushResult(
            dropped_frames=dropped_frames,
            dropped_ms=dropped_ms,
            last_sent_mark=None,
        )

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._inbound.put(None)


# ---------------------------------------------------------------------------
# StreamingPipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessAck:
    """Truthy ack returned by process_pcm16 (non-blocking)."""

    ok: bool = True
    seq: int = 0

    def __bool__(self) -> bool:
        return self.ok


class _SessionLike(Protocol):
    async def run(self) -> None: ...


class _TransportLike(Protocol):
    @property
    def media(self) -> MediaFormat: ...
    async def push_inbound(self, frame: AudioFrame) -> None: ...
    async def close(self) -> None: ...


class StreamingPipeline:
    """Streaming call pipeline: process_pcm16 → resample → push_inbound.

    The session reflex loop runs as a lazily-started background task so the very
    first ``process_pcm16`` does not deadlock waiting for inbound (the session
    starts ``run()`` and immediately drains ``inbound()``).
    """

    is_streaming = True

    def __init__(
        self,
        *,
        media: MediaFormat,
        session: _SessionLike,
        transport: _TransportLike,
        clock: Clock,
    ) -> None:
        self._media_16k = media
        self._session = session
        self._transport = transport
        self._clock = clock
        self._seq = 0
        self._task: asyncio.Task[None] | None = None

    @property
    def transport(self) -> _TransportLike:
        """Expose the underlying transport (B1) so the engine can swap its sink."""
        return self._transport

    def _ensure_started(self) -> None:
        # Invariant: NO ``await`` here — process_pcm16's first call must start the
        # session task and return promptly without yielding to the loop, so the
        # caller never blocks waiting for the session to drain inbound.
        if self._task is None:
            self._task = asyncio.create_task(self._session.run())

    async def process_pcm16(
        self, *, call_id: str, pcm16: bytes, sample_rate: int
    ) -> ProcessAck:
        self._ensure_started()
        if sample_rate != self._media_16k.sample_rate:
            import audioop  # ty: ignore[unresolved-import]  # stdlib on py3.11; audioop-lts on 3.13+

            converted, _state = audioop.ratecv(
                pcm16, 2, self._media_16k.channels, sample_rate,
                self._media_16k.sample_rate, None,
            )
        else:
            converted = pcm16
        frame = AudioFrame(
            pcm16=converted,
            media=self._media_16k,
            timestamp_ms=self._clock.now_ms(),
            seq=self._seq,
        )
        seq = self._seq
        self._seq += 1
        await self._transport.push_inbound(frame)
        return ProcessAck(ok=True, seq=seq)

    async def aclose(self, *, abort: bool = False) -> None:
        """Drain (or abort) the session task and close the transport (I3).

        Idempotent: a second call no-ops (``_task`` is cleared in ``finally``,
        no re-raise). ``abort=True`` cancels a still-running task then awaits with
        ``CancelledError`` suppressed (forced teardown); the default drains.
        """
        await self._transport.close()
        task = self._task
        if task is None:
            return
        try:
            if abort and not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                if not abort:
                    raise
        finally:
            self._task = None


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def _recording_outbound_sink() -> Any:
    """Default Slice 6 outbound sink: buffers frames + records drops.

    Replaced by the live aiortc track sink in Slice 6b. Exposed as ``.sent`` /
    ``.drops`` on the returned callable for inspection.
    """

    sent: list[AudioFrame] = []
    drops = {"count": 0}

    async def sink(frame: AudioFrame) -> None:
        sent.append(frame)

    async def drop() -> None:
        drops["count"] += 1

    sink.sent = sent  # type: ignore[attr-defined]
    sink.drops = drops  # type: ignore[attr-defined]
    sink.drop = drop  # type: ignore[attr-defined]
    return sink


def build_streaming_pipeline(
    config: Any,
    *,
    cognitive: str = "fake",
    clock: Clock | None = None,
    sink: OutboundSink | None = None,
) -> StreamingPipeline:
    """Wire a StreamingPipeline for the engine.

    ``cognitive="fake"`` (Slice 6 default) builds deterministic Fake cognitive
    ports on a VirtualClock. ``cognitive="real"`` raises NotImplementedError
    (lands in Slice 7).

    ``sink`` (B3): when ``None`` (the live engine path) the transport is built
    with a module-level no-op sink — frames buffer harmlessly until ``start()``
    calls ``set_outbound_sink`` with the real aiortc track. Tests pass an explicit
    recording sink to inspect emitted frames.
    """
    if cognitive == "real":
        raise NotImplementedError(
            "cognitive='real' streaming ports land in Slice 7; "
            "Slice 6 ships the pure-asyncio core with cognitive='fake'."
        )
    if cognitive != "fake":
        raise ValueError(f"unknown cognitive mode: {cognitive!r}")

    from .clock import VirtualClock
    from .fakes import FakeBrain, FakeSTT, FakeTTS, FakeTurnDetection
    from .interruption import InterruptionPolicy
    from .session import StreamingCallSession
    from .tracer import StreamingCallTracer
    from .types import StreamingCallContext

    # The fake cognitive ports are driven by a deterministic VirtualClock.
    vclock: VirtualClock = clock if isinstance(clock, VirtualClock) else VirtualClock()
    media = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)
    call_id = "streaming-fake"
    ctx = StreamingCallContext(
        call_id=call_id,
        contact_id="fake-contact",
        session_id="fake-session",
        media=media,
    )

    outbound_sink: OutboundSink = _noop_sink if sink is None else sink
    transport = AiortcStreamingTransport(media, clock=vclock, outbound_sink=outbound_sink)
    stt = FakeSTT()
    turns = FakeTurnDetection([])
    tts = FakeTTS(vclock)

    def brain_factory() -> FakeBrain:
        return FakeBrain(vclock, text="", delay_ms=0)

    session = StreamingCallSession(
        ctx,
        transport=transport,
        stt=stt,
        turns=turns,
        tts=tts,
        brain_factory=brain_factory,
        policy=InterruptionPolicy(),
        tracer=StreamingCallTracer(call_id),
        clock=vclock,
    )
    return StreamingPipeline(
        media=media, session=session, transport=transport, clock=vclock
    )
