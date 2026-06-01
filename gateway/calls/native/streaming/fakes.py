"""Deterministic fakes for the streaming voice slice (WP4).

Each fake satisfies its corresponding Port Protocol from `ports.py` and is
driven entirely by the injected VirtualClock — no ``asyncio.sleep`` / ``time``
calls appear here; all timing goes through ``clock.sleep()``.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .cancellation import CancellationScope
from .clock import VirtualClock
from .types import (
    AudioFrame,
    BrainEvent,
    BrainEventKind,
    FlushResult,
    MediaFormat,
    PlaybackMark,
    StreamingCallContext,
    TranscriptEvent,
    TtsAudioEvent,
    TtsEventKind,
    TurnEvent,
)

__all__ = [
    "FakeAudioTransport",
    "FakeTurnDetection",
    "FakeSTT",
    "FakeTTS",
    "FakeBrain",
]


# ---------------------------------------------------------------------------
# FakeAudioTransport
# ---------------------------------------------------------------------------


class FakeAudioTransport:
    """Satisfies AudioTransportPort.  All data is in-memory; clock-free."""

    def __init__(self, media: MediaFormat) -> None:
        self._media = media
        self.sent: list[AudioFrame] = []
        self._pending: list[AudioFrame] = []
        self.flushes: list[str] = []
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._closed = False

    @property
    def media(self) -> MediaFormat:
        return self._media

    async def push_inbound(self, frame: AudioFrame) -> None:
        await self._queue.put(frame)

    async def end_inbound(self) -> None:
        """Put a None sentinel to signal end-of-stream to inbound()."""
        await self._queue.put(None)

    def inbound(self) -> AsyncIterator[AudioFrame]:
        return self._inbound_gen()

    async def _inbound_gen(self) -> AsyncIterator[AudioFrame]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def emit_outbound(self, frame: AudioFrame) -> None:
        self.sent.append(frame)
        self._pending.append(frame)

    async def flush_outbound(self, reason: str) -> FlushResult:
        self.flushes.append(reason)
        dropped_frames = len(self._pending)
        dropped_ms = sum(f.duration_ms for f in self._pending)
        self._pending.clear()
        return FlushResult(
            dropped_frames=dropped_frames,
            dropped_ms=dropped_ms,
            last_sent_mark=None,
        )

    async def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# FakeTurnDetection
# ---------------------------------------------------------------------------


class FakeTurnDetection:
    """Satisfies TurnDetectionPort.  Events are triggered by frame.seq."""

    def __init__(self, script: list[tuple[int, TurnEvent]]) -> None:
        # Build a mapping from seq → ordered list of TurnEvents
        self._script: dict[int, list[TurnEvent]] = {}
        for seq, event in script:
            self._script.setdefault(seq, []).append(event)

    async def observe(self, frame: AudioFrame) -> tuple[TurnEvent, ...]:
        return tuple(self._script.get(frame.seq, []))

    def reset(self) -> None:
        pass  # stateless reset


# ---------------------------------------------------------------------------
# FakeSTT
# ---------------------------------------------------------------------------


class FakeSTT:
    """Satisfies SpeechToTextPort.  Partials and final are configured at construction."""

    def __init__(
        self,
        partials: list[TranscriptEvent] | tuple[TranscriptEvent, ...] = (),
        final: TranscriptEvent | None = None,
    ) -> None:
        self._partials = list(partials)
        self._final = final
        self._started = False
        self._cancelled = False
        self._closed = False

    async def start(self, ctx: StreamingCallContext) -> None:
        self._started = True

    async def push(self, frame: AudioFrame) -> None:
        pass

    async def cancel(self) -> None:
        self._cancelled = True

    async def close(self) -> None:
        self._closed = True

    def events(self) -> AsyncIterator[TranscriptEvent]:
        return self._events_gen()

    async def _events_gen(self) -> AsyncIterator[TranscriptEvent]:
        for event in self._partials:
            yield event

    async def finalize(self) -> TranscriptEvent | None:
        return self._final


# ---------------------------------------------------------------------------
# FakeTTS
# ---------------------------------------------------------------------------


class FakeTTS:
    """Satisfies TextToSpeechPort.  Driven by an injected VirtualClock."""

    def __init__(
        self,
        clock: VirtualClock,
        frames_per_word: int = 2,
        frame_ms: int = 20,
        sample_rate: int = 16000,
    ) -> None:
        self._clock = clock
        self._frames_per_word = frames_per_word
        self._frame_ms = frame_ms
        self._sample_rate = sample_rate
        self._cancelled = False
        self._flushed = False
        self._seq = 0

    def synthesize(
        self,
        text: str,
        ctx: StreamingCallContext,
        scope: CancellationScope,
    ) -> AsyncIterator[TtsAudioEvent]:
        return self._synthesize_gen(text, ctx, scope)

    async def _synthesize_gen(
        self,
        text: str,
        ctx: StreamingCallContext,
        scope: CancellationScope,
    ) -> AsyncIterator[TtsAudioEvent]:
        words = text.split()
        # We track the char offset as a running cursor into `text`.
        # After each word we advance the cursor past the word AND past any
        # following space so that text[:char_offset] exactly equals the
        # "heard prefix" including trailing whitespace.
        search_start = 0

        for word in words:
            # Check cancellation before emitting anything for this word.
            if scope.cancelled:
                yield TtsAudioEvent(call_id=ctx.call_id, kind=TtsEventKind.CANCELLED)
                return

            # Emit `frames_per_word` AUDIO frames, sleeping frame_ms between them.
            samples_per_frame = int(self._sample_rate * self._frame_ms / 1000)
            pcm = b"\x00" * samples_per_frame * 2  # 16-bit mono
            for _ in range(self._frames_per_word):
                frame = AudioFrame(
                    pcm16=pcm,
                    media=ctx.media,
                    timestamp_ms=self._clock.now_ms(),
                    seq=self._seq,
                )
                self._seq += 1
                yield TtsAudioEvent(
                    call_id=ctx.call_id,
                    kind=TtsEventKind.AUDIO,
                    frame=frame,
                )
                await self._clock.sleep(self._frame_ms)

            # Compute char_offset: advance search_start to just past this word's
            # occurrence in `text`, then include any trailing space.
            word_pos = text.index(word, search_start)
            char_offset = word_pos + len(word)
            # If this word is followed by a space in the original text, include it.
            if char_offset < len(text) and text[char_offset] == " ":
                char_offset += 1
            search_start = char_offset

            mark = PlaybackMark(
                call_id=ctx.call_id,
                char_offset=char_offset,
                text_so_far=text[:char_offset],
                at_ms=self._clock.now_ms(),
            )
            yield TtsAudioEvent(
                call_id=ctx.call_id,
                kind=TtsEventKind.MARK,
                mark=mark,
                span_text=word,
            )

        # All words emitted (and scope not cancelled during the loop).
        if not scope.cancelled:
            yield TtsAudioEvent(call_id=ctx.call_id, kind=TtsEventKind.DONE)

    async def cancel(self) -> None:
        self._cancelled = True

    async def flush(self) -> None:
        self._flushed = True


# ---------------------------------------------------------------------------
# FakeBrain
# ---------------------------------------------------------------------------


class FakeBrain:
    """Satisfies HermesBrainPort.  Delays via clock.sleep(); respects cancellation."""

    def __init__(
        self,
        clock: VirtualClock,
        text: str,
        delay_ms: int = 0,
    ) -> None:
        self._clock = clock
        self.text = text
        self.delay_ms = delay_ms
        self.abandoned = False
        self.started = False

    def respond(
        self,
        turn: TranscriptEvent,
        ctx: StreamingCallContext,
        scope: CancellationScope,
    ) -> AsyncIterator[BrainEvent]:
        return self._respond_gen(ctx, scope)

    async def _respond_gen(
        self,
        ctx: StreamingCallContext,
        scope: CancellationScope,
    ) -> AsyncIterator[BrainEvent]:
        self.started = True
        remaining = self.delay_ms
        step = 20  # ms per sleep increment

        while remaining > 0:
            if scope.cancelled:
                self.abandoned = True
                return
            await self._clock.sleep(min(step, remaining))
            remaining -= min(step, remaining)

        # Final cancellation check after all sleep steps.
        if scope.cancelled:
            self.abandoned = True
            return

        yield BrainEvent(
            call_id=ctx.call_id,
            kind=BrainEventKind.FINAL_TEXT,
            text=self.text,
        )
