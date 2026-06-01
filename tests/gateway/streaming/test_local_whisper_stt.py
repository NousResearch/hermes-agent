from __future__ import annotations

import wave
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock

import pytest

from gateway.calls.native.streaming.local_whisper_stt import (
    LocalWhisperSTT,
    build_local_whisper_stt,
)
from gateway.calls.native.streaming.types import (
    AudioFrame,
    MediaFormat,
    StreamingCallContext,
    TranscriptKind,
)

pytestmark = pytest.mark.asyncio

M16 = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)
M8 = MediaFormat(sample_rate=8000, channels=1, frame_ms=20)
M16_STEREO = MediaFormat(sample_rate=16000, channels=2, frame_ms=20)

FIX = Path(__file__).parent / "fixtures" / "turn_detection" / "speech_16k_mono.wav"


def frame(seq: int, ms: int, nbytes: int = 640, media: MediaFormat = M16) -> AudioFrame:
    return AudioFrame(pcm16=b"\x01\x02" * (nbytes // 2), media=media, timestamp_ms=ms, seq=seq)


def frame_from(seq: int, ms: int, pcm: bytes, media: MediaFormat = M16) -> AudioFrame:
    return AudioFrame(pcm16=pcm, media=media, timestamp_ms=ms, seq=seq)


def ctx(call_id: str = "c1") -> StreamingCallContext:
    return StreamingCallContext(call_id=call_id, contact_id="x", session_id="s", media=M16)


# Scenario 1: buffer + finalize -> one FINAL
async def test_finalize_produces_single_final_event_from_buffered_audio() -> None:
    transcribe = Mock(return_value="hello hermes")
    stt = LocalWhisperSTT(media=M16, transcribe=transcribe, call_id="c1")
    await stt.start(ctx())
    await stt.push(frame(0, 100))
    await stt.push(frame(1, 120))
    await stt.push(frame(2, 140))

    final = await stt.finalize()

    transcribe.assert_called_once()
    audio = transcribe.call_args.args[0]
    # 3 frames * 640 bytes = 1920 bytes int16 -> 960 samples
    assert audio.shape == (960,)
    assert str(audio.dtype) == "float32"
    assert audio.min() >= -1.0 and audio.max() <= 1.0

    assert final is not None
    assert final.kind is TranscriptKind.FINAL
    assert final.text == "hello hermes"
    assert final.provider == "faster-whisper"
    assert final.call_id == "c1"
    assert final.start_ms == 100
    assert final.end_ms == 140


# Scenario 2: empty buffer -> None
async def test_finalize_with_no_audio_returns_none() -> None:
    transcribe = Mock(return_value="should not be called")
    stt = LocalWhisperSTT(media=M16, transcribe=transcribe, call_id="c1")
    await stt.start(ctx())

    final = await stt.finalize()

    assert final is None
    transcribe.assert_not_called()


# Scenario 3: events() is empty
async def test_events_yields_nothing() -> None:
    stt = LocalWhisperSTT(media=M16, transcribe=Mock(return_value=""), call_id="c1")
    await stt.start(ctx())
    async for _event in stt.events():
        pytest.fail("events() must not yield any transcript this slice")


# Scenario 4: cancel()/close() clear state
async def test_cancel_clears_buffer_then_finalize_returns_none() -> None:
    transcribe = Mock(return_value="x")
    stt = LocalWhisperSTT(media=M16, transcribe=transcribe, call_id="c1")
    await stt.start(ctx())
    await stt.push(frame(0, 100))
    await stt.cancel()

    final = await stt.finalize()

    assert final is None
    transcribe.assert_not_called()
    await stt.close()  # must not raise


# Scenario 5: absent extra -> clear RuntimeError
async def test_build_raises_when_faster_whisper_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import gateway.calls.native.streaming.local_whisper_stt as mod

    monkeypatch.setattr(mod, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="simplex-streaming-local-stt"):
        build_local_whisper_stt(M16)


# Scenario 7: 16k-mono guard
async def test_push_rejects_non_16k_frame() -> None:
    stt = LocalWhisperSTT(media=M16, transcribe=Mock(return_value=""), call_id="c1")
    await stt.start(ctx())
    with pytest.raises(ValueError, match="16kHz mono"):
        await stt.push(frame(0, 0, media=M8))


async def test_push_rejects_non_mono_frame() -> None:
    stt = LocalWhisperSTT(media=M16, transcribe=Mock(return_value=""), call_id="c1")
    await stt.start(ctx())
    with pytest.raises(ValueError, match="16kHz mono"):
        await stt.push(frame(0, 0, media=M16_STEREO))


# Scenario 6: real-ASR contract (local only; skips when faster-whisper absent)
@pytest.mark.skipif(
    find_spec("faster_whisper") is None,
    reason="simplex-streaming-local-stt extra not installed",
)
async def test_real_asr_contract_transcribes_fixture() -> None:
    w = wave.open(str(FIX), "rb")
    pcm = w.readframes(w.getnframes())
    w.close()

    stt = build_local_whisper_stt(M16, call_id="real")
    await stt.start(ctx("real"))
    for seq, off in enumerate(range(0, len(pcm) - 640, 640)):
        await stt.push(frame_from(seq, seq * 20, pcm[off:off + 640]))

    final = await stt.finalize()

    assert final is not None
    assert final.kind is TranscriptKind.FINAL
    assert final.provider == "faster-whisper"
    norm = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in final.text)
    norm = " ".join(norm.split())
    assert "weather" in norm and "today" in norm and ("hermes" in norm or "forecast" in norm)
