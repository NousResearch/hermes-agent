from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.calls.native.streaming.local_turn_detection import (
    LocalTurnDetector,
    build_local_turn_detector,
)
from gateway.calls.native.streaming.pipecat_runtime import pipecat_available
from gateway.calls.native.streaming.types import (
    AudioFrame,
    MediaFormat,
    TurnEventKind,
)

pytestmark = pytest.mark.asyncio

M16 = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)
FIX = Path(__file__).parent / "fixtures" / "turn_detection" / "speech_16k_mono.wav"


def frame(seq: int, ms: int, pcm: bytes | None = None, nbytes: int = 640) -> AudioFrame:
    pcm16 = pcm if pcm is not None else b"\x01\x02" * (nbytes // 2)
    return AudioFrame(pcm16=pcm16, media=M16, timestamp_ms=ms, seq=seq)


def make(vad_states, eot=None, conf=0.9):
    """Build a LocalTurnDetector with mocked VAD + SmartTurn.

    ``vad_states`` is the sequence of VADState values returned per window.
    The mock reports ``num_frames_required()==256`` so the detector slices
    512-byte windows (256 samples * 2 bytes), matching the plan's assertions.
    """
    from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
    from pipecat.audio.vad.vad_analyzer import VADState  # noqa: F401  (real enum for tests)

    vad = Mock()
    vad.num_frames_required = Mock(return_value=256)
    seq = iter(vad_states)
    vad.analyze_audio = AsyncMock(side_effect=lambda buf: next(seq))
    vad.voice_confidence = Mock(return_value=conf)

    st = Mock()
    st.append_audio = Mock(return_value=(eot or EndOfTurnState.INCOMPLETE))
    st.clear = Mock()

    det = LocalTurnDetector(media=M16, vad=vad, smart_turn=st, call_id="call-1")
    return det, vad, st


# ---- Scenario 1: re-chunking -------------------------------------------------


async def test_rechunks_640_byte_frames_into_512_byte_windows():
    from pipecat.audio.vad.vad_analyzer import VADState

    det, vad, st = make([VADState.QUIET, VADState.QUIET])
    # Two 640-byte frames = 1280 bytes -> two 512-byte windows, 256 remain.
    await det.observe(frame(0, 0))
    await det.observe(frame(1, 20))

    assert vad.analyze_audio.await_count == 2
    for call in vad.analyze_audio.await_args_list:
        assert len(call.args[0]) == 512
    assert st.append_audio.call_count == 2
    for call in st.append_audio.call_args_list:
        assert len(call.args[0]) == 512
    # 256 leftover bytes still buffered (no third window yet).
    assert len(det._buffer) == 256


# ---- Scenario 2: speech started/stopped edges --------------------------------


async def test_started_and_stopped_edges_emit_once():
    from pipecat.audio.vad.vad_analyzer import VADState

    states = [
        VADState.QUIET,
        VADState.STARTING,
        VADState.SPEAKING,
        VADState.SPEAKING,
        VADState.STOPPING,
        VADState.QUIET,
    ]
    det, vad, st = make(states, conf=0.77)
    # Each 640-byte frame yields one 512-byte window plus 128 leftover; two
    # frames give exactly one extra window from accumulated leftovers. Feed
    # enough frames to surface all six scripted states.
    kinds: list[TurnEventKind] = []
    started_at = None
    started_conf = None
    seq = 0
    ms = 0
    # 6 windows of 512 bytes = 3072 bytes; 640-byte frames -> need ceil(3072/640)=5 frames.
    while vad.analyze_audio.await_count < 6:
        evs = await det.observe(frame(seq, ms))
        for e in evs:
            kinds.append(e.kind)
            if e.kind is TurnEventKind.USER_SPEECH_STARTED:
                started_at = e.at_ms
                started_conf = e.vad_confidence
        seq += 1
        ms += 20

    assert kinds.count(TurnEventKind.USER_SPEECH_STARTED) == 1
    assert kinds.count(TurnEventKind.USER_SPEECH_STOPPED) == 1
    # STARTED comes before STOPPED.
    assert kinds.index(TurnEventKind.USER_SPEECH_STARTED) < kinds.index(
        TurnEventKind.USER_SPEECH_STOPPED
    )
    assert started_at is not None
    assert started_conf == pytest.approx(0.77)


# ---- Scenario 3: endpoint detected -------------------------------------------


async def test_endpoint_detected_when_smartturn_complete():
    from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
    from pipecat.audio.vad.vad_analyzer import VADState

    det, vad, st = make([VADState.SPEAKING], eot=EndOfTurnState.COMPLETE)
    evs = await det.observe(frame(0, 100))
    endpoints = [e for e in evs if e.kind is TurnEventKind.ENDPOINT_DETECTED]
    assert len(endpoints) == 1
    assert endpoints[0].at_ms == 100
    assert endpoints[0].endpoint_confidence == 0.0
    assert endpoints[0].source == "silero+smartturn-v3"


# ---- Scenario 4: sample-rate guard -------------------------------------------


async def test_observe_rejects_non_16k_mono():
    from pipecat.audio.vad.vad_analyzer import VADState

    det, vad, st = make([VADState.QUIET])
    bad = AudioFrame(
        pcm16=b"\x00" * 640,
        media=MediaFormat(sample_rate=8000, channels=1, frame_ms=20),
        timestamp_ms=0,
        seq=0,
    )
    with pytest.raises(ValueError, match="16kHz mono"):
        await det.observe(bad)

    stereo = AudioFrame(
        pcm16=b"\x00" * 640,
        media=MediaFormat(sample_rate=16000, channels=2, frame_ms=20),
        timestamp_ms=0,
        seq=0,
    )
    with pytest.raises(ValueError, match="16kHz mono"):
        await det.observe(stereo)


# ---- Scenario 5: reset() clears state ----------------------------------------


async def test_reset_clears_buffer_and_state_and_reemits_started():
    from pipecat.audio.vad.vad_analyzer import VADState

    # Drive into SPEAKING, then reset, then re-enter SPEAKING.
    states = [
        VADState.SPEAKING,  # window 1 -> STARTED
        VADState.SPEAKING,  # window 2 (after reset) -> STARTED again
    ]
    det, vad, st = make(states)

    evs1 = await det.observe(frame(0, 0))
    assert any(e.kind is TurnEventKind.USER_SPEECH_STARTED for e in evs1)
    # Leave some buffered bytes (frame is 640, window 512 -> 128 leftover).
    assert len(det._buffer) == 128

    det.reset()
    assert len(det._buffer) == 0
    st.clear.assert_called_once()

    evs2 = await det.observe(frame(1, 40))
    assert any(e.kind is TurnEventKind.USER_SPEECH_STARTED for e in evs2)


# ---- Scenario 7: absent extra is safe ----------------------------------------


async def test_build_raises_clear_error_without_pipecat(monkeypatch):
    monkeypatch.setattr(
        "gateway.calls.native.streaming.local_turn_detection.pipecat_available",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="simplex-streaming"):
        build_local_turn_detector(M16)


# ---- Scenario 6: real-onnx contract ------------------------------------------


@pytest.mark.skipif(
    not pipecat_available(), reason="simplex-streaming extra not installed"
)
async def test_real_onnx_contract_emits_started_stopped_endpoint():
    w = wave.open(str(FIX), "rb")
    pcm = w.readframes(w.getnframes())
    w.close()
    pcm += b"\x00" * (16000 * 2)  # 1s trailing silence to trigger endpoint

    det = build_local_turn_detector(M16, call_id="real")
    kinds: list[TurnEventKind] = []
    seq = 0
    for off in range(0, len(pcm) - 640, 640):
        evs = await det.observe(frame(seq, seq * 20, pcm=pcm[off : off + 640]))
        kinds.extend(e.kind for e in evs)
        seq += 1

    assert TurnEventKind.USER_SPEECH_STARTED in kinds
    assert TurnEventKind.USER_SPEECH_STOPPED in kinds
    assert TurnEventKind.ENDPOINT_DETECTED in kinds
    i_start = kinds.index(TurnEventKind.USER_SPEECH_STARTED)
    i_stop = kinds.index(TurnEventKind.USER_SPEECH_STOPPED)
    i_end = kinds.index(TurnEventKind.ENDPOINT_DETECTED)
    assert i_start < i_stop < i_end
