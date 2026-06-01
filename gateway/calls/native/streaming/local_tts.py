"""Real ``TextToSpeechPort`` over an injected PCM synthesis backend.

``StreamingTTS`` turns an injected ``synthesize_pcm(text) -> bytes`` backend
(contracted to return 16 kHz mono int16 PCM) into the port event stream:
frame the PCM into 20 ms ``AUDIO`` frames, interleave word-boundary ``MARK``
events so the heard-span ledger stays meaningful, honour mid-stream barge-in
(``scope.cancelled`` / ``cancel()`` -> ``CANCELLED``), and finish with ``DONE``.

No wall-clock: timestamps are derived from an injected ``Clock`` captured once
at synthesis start plus a per-frame increment. The CPU-bound backend is
offloaded with ``asyncio.to_thread``. The generic seam does **no** resampling
and asserts the call target rate is 16 kHz mono; the ``build_piper_tts``
factory owns the 22050 -> 16000 resampling (Piper's default voice rate).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from .cancellation import CancellationScope
from .clock import Clock, MonotonicClock
from .types import (
    AudioFrame,
    MediaFormat,
    PlaybackMark,
    StreamingCallContext,
    TtsAudioEvent,
    TtsEventKind,
)

__all__ = ["StreamingTTS", "build_piper_tts"]

_TARGET_RATE = 16000


def _word_boundaries(text: str) -> list[int]:
    """Return sorted char offsets that end a word (just past a word + space).

    Each boundary is a valid ``char_offset`` whose ``text[:offset]`` is a real
    word-prefix: the index just past a run of non-space characters, plus the
    immediately following space when present, and finally ``len(text)``.
    """
    boundaries: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] == " ":
            i += 1
            continue
        # advance to end of word
        j = i
        while j < n and text[j] != " ":
            j += 1
        # include a single trailing space if present
        if j < n and text[j] == " ":
            j += 1
        boundaries.append(j)
        i = j
    if not boundaries or boundaries[-1] != n:
        boundaries.append(n)
    return boundaries


class StreamingTTS:
    """A real ``TextToSpeechPort`` over an injected 16 kHz mono PCM backend."""

    def __init__(
        self,
        *,
        media: MediaFormat,
        synthesize_pcm: Callable[[str], bytes],
        clock: Clock,
        call_id: str = "",
        frame_ms: int = 20,
    ) -> None:
        self._media = media
        self._synthesize_pcm = synthesize_pcm
        self._clock = clock
        self._call_id = call_id
        self._frame_ms = frame_ms
        self._cancelled = False

    def synthesize(
        self,
        text: str,
        ctx: StreamingCallContext,
        scope: CancellationScope,
    ) -> AsyncIterator[TtsAudioEvent]:
        return self._gen(text, ctx, scope)

    async def _gen(
        self,
        text: str,
        ctx: StreamingCallContext,
        scope: CancellationScope,
    ) -> AsyncIterator[TtsAudioEvent]:
        if ctx.media.sample_rate != _TARGET_RATE or ctx.media.channels != 1:
            raise ValueError(
                "StreamingTTS requires 16kHz mono; got "
                f"sample_rate={ctx.media.sample_rate}, channels={ctx.media.channels}"
            )

        if scope.cancelled or self._cancelled:
            yield TtsAudioEvent(call_id=self._call_id, kind=TtsEventKind.CANCELLED)
            return

        pcm = await asyncio.to_thread(self._synthesize_pcm, text)
        if not pcm:
            yield TtsAudioEvent(call_id=self._call_id, kind=TtsEventKind.DONE)
            return

        frame_bytes = int(_TARGET_RATE * self._frame_ms / 1000) * 2  # 640
        total = len(pcm)
        boundaries = _word_boundaries(text)
        t0 = self._clock.now_ms()
        seq = 0
        emitted = 0
        prev_off = 0

        for i in range(0, total, frame_bytes):
            if scope.cancelled or self._cancelled:
                yield TtsAudioEvent(
                    call_id=self._call_id, kind=TtsEventKind.CANCELLED
                )
                return

            chunk = pcm[i : i + frame_bytes]
            real_len = len(chunk)
            if real_len < frame_bytes:
                chunk = chunk + b"\x00" * (frame_bytes - real_len)

            frame = AudioFrame(
                pcm16=chunk,
                media=ctx.media,
                timestamp_ms=t0 + seq * self._frame_ms,
                seq=seq,
            )
            yield TtsAudioEvent(
                call_id=self._call_id, kind=TtsEventKind.AUDIO, frame=frame
            )
            at_ms = t0 + seq * self._frame_ms
            seq += 1
            emitted += real_len

            # Proportional char offset, snapped BACKWARD to the last word
            # boundary at or before it, then made monotonic.
            prop = len(text) * emitted // total
            snapped = 0
            for b in boundaries:
                if b <= prop:
                    snapped = b
                else:
                    break
            off = max(prev_off, snapped)
            if off > prev_off:
                yield TtsAudioEvent(
                    call_id=self._call_id,
                    kind=TtsEventKind.MARK,
                    mark=PlaybackMark(
                        call_id=self._call_id,
                        char_offset=off,
                        text_so_far=text[:off],
                        at_ms=at_ms,
                        boundary="word",
                    ),
                )
                prev_off = off

        if prev_off < len(text):
            yield TtsAudioEvent(
                call_id=self._call_id,
                kind=TtsEventKind.MARK,
                mark=PlaybackMark(
                    call_id=self._call_id,
                    char_offset=len(text),
                    text_so_far=text,
                    at_ms=t0 + max(0, seq - 1) * self._frame_ms,
                    boundary="word",
                ),
            )

        yield TtsAudioEvent(call_id=self._call_id, kind=TtsEventKind.DONE)

    async def cancel(self) -> None:
        self._cancelled = True

    async def flush(self) -> None:
        pass


def build_piper_tts(
    media: MediaFormat,
    *,
    clock: Clock | None = None,
    call_id: str = "",
    voice: str = "en_US-lessac-medium",
) -> StreamingTTS:
    """Construct a ``StreamingTTS`` backed by a real local Piper voice.

    Piper's default voices synthesize at 22050 Hz; this factory's backend
    resamples to 16 kHz mono with ``audioop.ratecv`` (stdlib) before returning
    bytes, so the generic ``StreamingTTS`` seam stays 16 kHz-only. Raises a
    clear ``RuntimeError`` naming the extra when piper is absent; the piper
    import is lazy so the package imports without the extra installed.
    """
    from importlib.util import find_spec

    if find_spec("piper") is None:
        raise RuntimeError(
            "StreamingTTS Piper backend requires the optional piper-tts "
            "dependency. Install: pip install "
            "'hermes-agent[simplex-streaming-local-tts]'"
        )

    import audioop  # ty: ignore[unresolved-import]  # audioop-lts backports this on 3.13+
    from pathlib import Path

    from piper import PiperVoice

    from tools.tts_tool import _get_piper_voices_dir, _resolve_piper_voice_path

    download_dir = Path(_get_piper_voices_dir())
    model_path = _resolve_piper_voice_path(voice, download_dir)
    loaded_voice = PiperVoice.load(model_path)

    def _synthesize_pcm(text: str) -> bytes:
        src_rate = 0
        raw = bytearray()
        for chunk in loaded_voice.synthesize(text):
            rate = int(chunk.sample_rate)
            if src_rate == 0:
                src_rate = rate
            elif rate != src_rate:
                # The single-ratecv resample below assumes one uniform rate;
                # fail loudly rather than emit garbled audio on a multi-rate voice.
                raise RuntimeError(
                    f"Piper emitted mixed sample rates ({src_rate} then {rate}); "
                    "the resampler assumes a single rate per utterance"
                )
            raw.extend(chunk.audio_int16_bytes)
        data = bytes(raw)
        if not data or src_rate == _TARGET_RATE:
            return data
        # int16 mono resample src_rate -> 16000 (stdlib, no numpy).
        converted, _state = audioop.ratecv(data, 2, 1, src_rate, _TARGET_RATE, None)
        return converted

    return StreamingTTS(
        media=media,
        synthesize_pcm=_synthesize_pcm,
        clock=clock or MonotonicClock(),
        call_id=call_id,
    )
