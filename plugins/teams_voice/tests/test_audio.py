"""Tests for the PCM16 audio helpers (resample / frame / rms)."""

from __future__ import annotations

import struct

from plugins.teams_voice import audio


def _pcm(*samples: int) -> bytes:
    return struct.pack("<" + "h" * len(samples), *samples)


def test_resample_identity_when_rates_equal():
    data = _pcm(1, 2, 3, 4)
    assert audio.resample_pcm16(data, 16000, 16000) == data


def test_resample_empty_and_odd_length():
    assert audio.resample_pcm16(b"", 16000, 24000) == b""
    # odd trailing byte is dropped, single sample survives
    out = audio.resample_pcm16(_pcm(1000) + b"\x01", 16000, 24000)
    assert len(out) % 2 == 0


def test_resample_16k_to_24k_length_ratio():
    # 160 samples (10 ms @16k) -> ~240 samples (10 ms @24k)
    data = _pcm(*([0] * 160))
    out = audio.resample_pcm16(data, 16000, 24000)
    out_samples = len(out) // 2
    assert 236 <= out_samples <= 244  # ~1.5x within rounding


def test_resample_roundtrip_preserves_length_class():
    data = _pcm(*range(-100, 100))
    up = audio.resample_pcm16(data, 16000, 24000)
    down = audio.resample_pcm16(up, 24000, 16000)
    # back to ~original sample count
    assert abs(len(down) // 2 - len(data) // 2) <= 2


def test_frame_pcm16_splits_and_keeps_residual():
    data = b"\x00" * (640 + 640 + 100)
    frames, residual = audio.frame_pcm16(data, 640)
    assert len(frames) == 2
    assert all(len(f) == 640 for f in frames)
    assert len(residual) == 100


def test_pcm16_rms_silence_and_signal():
    assert audio.pcm16_rms(b"") == 0.0
    assert audio.pcm16_rms(_pcm(0, 0, 0, 0)) == 0.0
    loud = audio.pcm16_rms(_pcm(32767, -32768, 32767, -32768))
    assert 0.9 <= loud <= 1.01
