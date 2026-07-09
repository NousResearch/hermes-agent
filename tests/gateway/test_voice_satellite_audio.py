"""Unit tests for the voice_satellite audio helpers."""

import math
import struct
import wave
from unittest.mock import patch

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("voice_satellite")
audio = _adapter._import_sibling("audio")

RATE = 16000


def make_pcm(seconds: float, amplitude: int, rate: int = RATE) -> bytes:
    """s16le mono sine wave (silence when amplitude=0)."""
    n = int(seconds * rate)
    if amplitude == 0:
        return b"\x00\x00" * n
    return b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(n)
    )


def test_rms_silence_is_zero_and_speech_is_loud():
    assert audio.rms(make_pcm(0.1, 0)) == 0
    assert audio.rms(make_pcm(0.1, 3000)) > 1000


def test_pcm_to_wav_roundtrip(tmp_path):
    pcm = make_pcm(0.25, 3000)
    out = str(tmp_path / "utt.wav")
    audio.pcm_to_wav(pcm, out, rate=RATE)
    with wave.open(out, "rb") as wf:
        assert wf.getframerate() == RATE
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.readframes(wf.getnframes()) == pcm


def test_transcode_to_pcm_invokes_ffmpeg_with_s16le_args():
    with patch.object(audio.subprocess, "run") as run:
        run.return_value.stdout = b"\x01\x02"
        result = audio.transcode_to_pcm("/tmp/reply.mp3", rate=22050)
    assert result == b"\x01\x02"
    args = run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert "/tmp/reply.mp3" in args
    for flag, value in (("-f", "s16le"), ("-ar", "22050"), ("-ac", "1")):
        assert value == args[args.index(flag) + 1]


def test_endpoint_detector_returns_utterance_after_trailing_silence():
    det = audio.EndpointDetector(
        silence_threshold=200, silence_duration=0.3, min_speech_seconds=0.2
    )
    for chunk, seconds in (
        (make_pcm(0.3, 3000), 0.3),
        (make_pcm(0.1, 0), 0.1),
        (make_pcm(0.1, 0), 0.1),
    ):
        assert det.feed(chunk, seconds) is None
    utterance = det.feed(make_pcm(0.1, 0), 0.1)
    assert utterance is not None
    assert len(utterance) == int(0.6 * RATE) * 2  # speech + all silence retained


def test_endpoint_detector_discards_silence_only_audio():
    det = audio.EndpointDetector(silence_duration=0.2, min_speech_seconds=0.2)
    assert det.feed(make_pcm(0.1, 0), 0.1) is None
    assert det.feed(make_pcm(0.1, 0), 0.1) is None  # endpoint hit, no speech -> None
    # detector reset itself: speech afterwards still produces an utterance
    assert det.feed(make_pcm(0.3, 3000), 0.3) is None
    assert det.feed(make_pcm(0.2, 0), 0.2) is not None


def test_endpoint_detector_caps_max_utterance():
    det = audio.EndpointDetector(
        silence_duration=5.0, min_speech_seconds=0.1, max_utterance_seconds=0.4
    )
    assert det.feed(make_pcm(0.3, 3000), 0.3) is None
    assert det.feed(make_pcm(0.3, 3000), 0.3) is not None  # 0.6s >= 0.4s cap
