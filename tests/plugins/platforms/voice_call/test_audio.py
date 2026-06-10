"""µ-law codec, resampler, and framing tests (vendored audio.py)."""

import math
from array import array

from plugins.platforms.voice_call import audio


def _sine_pcm16(rate=8000, freq=440.0, seconds=0.05, amplitude=12000):
    n = int(rate * seconds)
    samples = array(
        "h",
        (int(amplitude * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)),
    )
    return samples.tobytes()


def test_ulaw_silence_byte():
    # µ-law encodes 0 as 0xFF.
    assert audio.pcm16_to_ulaw(b"\x00\x00") == bytes([audio.ULAW_SILENCE_BYTE])
    decoded = audio.ulaw_to_pcm16(bytes([audio.ULAW_SILENCE_BYTE]))
    assert array("h", decoded)[0] == 0


def test_ulaw_roundtrip_within_quantization_error():
    pcm = _sine_pcm16()
    restored = array("h", audio.ulaw_to_pcm16(audio.pcm16_to_ulaw(pcm)))
    original = array("h", pcm)
    assert len(restored) == len(original)
    for orig, back in zip(original, restored):
        # G.711 quantization error grows with magnitude; 8-bit µ-law keeps
        # error under ~1/16 of the sample value + a small floor.
        assert abs(orig - back) <= max(64, abs(orig) // 8), (orig, back)


def test_ulaw_known_values():
    # Reference vectors from the G.711 tables.
    assert audio.pcm16_to_ulaw(array("h", [-32635]).tobytes()) == b"\x00"
    assert audio.pcm16_to_ulaw(array("h", [32635]).tobytes()) == b"\x80"
    # Clipping beyond ±32635 maps to the extremes too.
    assert audio.pcm16_to_ulaw(array("h", [32767]).tobytes()) == b"\x80"


def test_ulaw_decode_sign_symmetry():
    for value in (100, 1000, 8000, 30000):
        pos = array("h", audio.ulaw_to_pcm16(
            audio.pcm16_to_ulaw(array("h", [value]).tobytes())))[0]
        neg = array("h", audio.ulaw_to_pcm16(
            audio.pcm16_to_ulaw(array("h", [-value]).tobytes())))[0]
        assert pos == -neg


def test_resample_length_ratio():
    pcm = _sine_pcm16(rate=8000, seconds=0.1)  # 800 samples
    up = audio.resample_pcm16(pcm, 8000, 24000)
    assert abs(len(up) // 2 - 2400) <= 2
    down = audio.resample_pcm16(up, 24000, 8000)
    assert abs(len(down) // 2 - 800) <= 2


def test_resample_preserves_constant_signal():
    pcm = array("h", [1000] * 160).tobytes()
    out = array("h", audio.resample_pcm16(pcm, 8000, 16000))
    assert all(abs(s - 1000) <= 1 for s in out)


def test_resample_identity_and_empty():
    pcm = _sine_pcm16()
    assert audio.resample_pcm16(pcm, 8000, 8000) == pcm
    assert audio.resample_pcm16(b"", 8000, 24000) == b""


def test_chunk_frames_pads_tail_with_silence():
    frames = audio.chunk_frames(bytes([0x55]) * 250)
    assert [len(f) for f in frames] == [160, 160]
    assert frames[1][90:] == bytes([audio.ULAW_SILENCE_BYTE]) * 70
    assert audio.silence_frame() == bytes([audio.ULAW_SILENCE_BYTE]) * 160
