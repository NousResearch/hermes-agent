from __future__ import annotations

import asyncio
import threading
from importlib.util import find_spec

import pytest

from gateway.calls.native.streaming.cancellation import CancellationScope
from gateway.calls.native.streaming.clock import VirtualClock
from gateway.calls.native.streaming.local_tts import (
    StreamingTTS,
    build_piper_tts,
)
from gateway.calls.native.streaming.types import (
    MediaFormat,
    StreamingCallContext,
    TtsEventKind,
)

pytestmark = pytest.mark.asyncio

M16 = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)
M8 = MediaFormat(sample_rate=8000, channels=1, frame_ms=20)
M16_STEREO = MediaFormat(sample_rate=16000, channels=2, frame_ms=20)

FRAME_BYTES = 640  # 16000 * 0.02 * 2


def ctx(call_id: str = "c1", media: MediaFormat = M16) -> StreamingCallContext:
    return StreamingCallContext(
        call_id=call_id, contact_id="x", session_id="s", media=media
    )


def fixed_pcm(nbytes: int) -> bytes:
    return (b"\x01\x02" * ((nbytes // 2) + 1))[:nbytes]


def make_tts(
    pcm: bytes,
    *,
    media: MediaFormat = M16,
    clock: VirtualClock | None = None,
    call_id: str = "c1",
) -> StreamingTTS:
    def _synth(text: str) -> bytes:
        return pcm

    return StreamingTTS(
        media=media,
        synthesize_pcm=_synth,
        clock=clock or VirtualClock(),
        call_id=call_id,
    )


# Scenario 1: framing -> AUDIO frames + DONE, monotonic timestamps, payload match
async def test_framing_produces_audio_frames_then_done() -> None:
    pcm = fixed_pcm(640 * 2 + 100)  # 2 full frames + a short remainder
    clock = VirtualClock()
    tts = make_tts(pcm, clock=clock)

    events = [e async for e in tts.synthesize("hello world", ctx(), CancellationScope())]

    audio = [e for e in events if e.kind is TtsEventKind.AUDIO]
    assert len(audio) == 3  # ceil(1380 / 640)
    assert events[-1].kind is TtsEventKind.DONE

    # Each frame is exactly FRAME_BYTES (last padded with silence).
    for e in audio:
        assert len(e.frame.pcm16) == FRAME_BYTES
        assert e.frame.media is M16

    # Timestamps monotonic from clock start + seq*20.
    t0 = audio[0].frame.timestamp_ms
    for i, e in enumerate(audio):
        assert e.frame.timestamp_ms == t0 + i * 20
        assert e.frame.seq == i

    # Concatenated AUDIO minus final pad == backend bytes.
    concatenated = b"".join(e.frame.pcm16 for e in audio)
    assert concatenated[: len(pcm)] == pcm
    assert concatenated[len(pcm):] == b"\x00" * (len(concatenated) - len(pcm))


# Scenario 2: MARKs are word-prefixes, monotonic, <= len(text)
async def test_marks_are_word_prefixes_and_monotonic() -> None:
    text = "one two three four"
    pcm = fixed_pcm(640 * 8)
    tts = make_tts(pcm)

    marks = []
    async for e in tts.synthesize(text, ctx(), CancellationScope()):
        if e.kind is TtsEventKind.MARK:
            marks.append(e.mark)

    assert marks, "expected at least one MARK"
    prev = -1
    for m in marks:
        assert m.text_so_far == text[: m.char_offset]
        assert text.startswith(m.text_so_far)
        # word-prefix: ends at a word boundary (end-of-text or just past a space)
        assert m.char_offset == len(text) or text[m.char_offset - 1] == " " or (
            m.char_offset < len(text) and text[m.char_offset] == " "
        )
        assert m.char_offset >= prev
        assert m.char_offset <= len(text)
        prev = m.char_offset
    # Final reaches end-of-text.
    assert marks[-1].char_offset == len(text)


# Scenario 3: barge-in mid-stream -> exactly one CANCELLED, then stop, no DONE
async def test_midstream_cancel_yields_cancelled_no_done() -> None:
    pcm = fixed_pcm(640 * 5)
    scope = CancellationScope()
    tts = make_tts(pcm)

    seen = []
    agen = tts.synthesize("one two three", ctx(), scope)
    async for e in agen:
        seen.append(e.kind)
        if e.kind is TtsEventKind.AUDIO and len(seen) == 1:
            scope.cancel("barge")

    assert TtsEventKind.AUDIO in seen
    assert seen[-1] is TtsEventKind.CANCELLED
    assert seen.count(TtsEventKind.CANCELLED) == 1
    assert TtsEventKind.DONE not in seen


# Scenario 4: cancel() out-of-band before iterating -> CANCELLED early
async def test_cancel_before_iteration_yields_cancelled() -> None:
    pcm = fixed_pcm(640 * 4)
    tts = make_tts(pcm)
    await tts.cancel()

    events = [e async for e in tts.synthesize("hi there", ctx(), CancellationScope())]
    assert events[0].kind is TtsEventKind.CANCELLED
    assert TtsEventKind.DONE not in [e.kind for e in events]


# Scenario 5: empty text / empty backend -> DONE only, no AUDIO
async def test_empty_text_done_only() -> None:
    tts = make_tts(b"")
    events = [e async for e in tts.synthesize("", ctx(), CancellationScope())]
    kinds = [e.kind for e in events]
    assert kinds == [TtsEventKind.DONE]


# Scenario 6: CPU-bound backend offloaded via asyncio.to_thread
async def test_synthesize_offloaded_to_thread() -> None:
    main_ident = threading.get_ident()
    recorded: dict[str, int] = {}

    def _synth(text: str) -> bytes:
        recorded["ident"] = threading.get_ident()
        return fixed_pcm(640)

    tts = StreamingTTS(
        media=M16, synthesize_pcm=_synth, clock=VirtualClock(), call_id="c1"
    )
    _ = [e async for e in tts.synthesize("hi", ctx(), CancellationScope())]

    assert recorded["ident"] != main_ident


# Scenario 7: 16k mono guard
async def test_non_16k_mono_raises_value_error() -> None:
    tts8 = make_tts(fixed_pcm(640), media=M8)
    with pytest.raises(ValueError, match="16kHz mono"):
        _ = [e async for e in tts8.synthesize("hi", ctx(media=M8), CancellationScope())]

    tts_stereo = make_tts(fixed_pcm(640), media=M16_STEREO)
    with pytest.raises(ValueError, match="16kHz mono"):
        _ = [
            e
            async for e in tts_stereo.synthesize(
                "hi", ctx(media=M16_STEREO), CancellationScope()
            )
        ]


# Scenario 8: real Piper contract (skips when extra not installed)
@pytest.mark.skipif(
    find_spec("piper") is None,
    reason="simplex-streaming-local-tts extra not installed",
)
async def test_real_piper_contract() -> None:
    tts = build_piper_tts(M16)
    kinds = []
    audio_bytes = 0
    marks = []
    async for e in tts.synthesize("Hello there friend.", ctx(), CancellationScope()):
        kinds.append(e.kind)
        if e.kind is TtsEventKind.AUDIO:
            audio_bytes += len(e.frame.pcm16)
        if e.kind is TtsEventKind.MARK:
            marks.append(e.mark)
    assert TtsEventKind.AUDIO in kinds and kinds[-1] is TtsEventKind.DONE
    assert audio_bytes > 0
    assert any(
        m.text_so_far and "Hello there friend.".startswith(m.text_so_far)
        for m in marks
    )
