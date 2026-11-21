"""Discord voice-message waveform: the envelope must follow the audio's loudness.

A uniform 256-byte array is what makes a bot's voice message look like a loading
bar instead of a recording, so these assert the mapping itself: silence reads as
the floor, loud audio reads high, and the bars track loudness changes across the
clip. No ffmpeg involved — the PCM path is a pure function.
"""
from array import array

import pytest

from plugins.platforms.discord.adapter_waveform import (
    FLAT_WAVEFORM,
    WAVEFORM_POINTS,
    envelope,
    waveform_from_pcm,
)


def _constant(amplitude: int, samples: int = 2048) -> array:
    return array("h", [amplitude] * samples)


def test_envelope_is_one_unsigned_value_per_bar():
    bars = envelope(_constant(12000))

    assert len(bars) == WAVEFORM_POINTS == 256
    assert all(0 <= value <= 255 for value in bars)


def test_silence_reads_as_the_floor_not_mid_scale():
    """Digital silence is 0, not 128: a quiet stretch has to look empty."""
    assert set(envelope(_constant(0))) == {0}


def test_full_scale_reads_as_the_top_of_the_scale():
    assert min(envelope(_constant(32767))) > 240


def test_bars_follow_loudness_across_the_clip():
    """The quiet half must render below the loud half, with real variation between bars."""
    quiet_then_loud = array("h", [300] * 2048 + [20000] * 2048)

    bars = envelope(quiet_then_loud)

    assert max(bars[:128]) < min(bars[128:])
    assert len(set(bars)) > 1, "a featureless array is the bug this exists to prevent"


def test_a_single_loud_window_stays_visible_among_silence():
    """Envelope resolution: one loud bucket in 256 is a spike, not a lost sample."""
    samples = array("h", [0] * 1024 * 255 + [26000] * 1024)

    bars = envelope(samples)

    assert bars[-1] > 200
    assert max(bars[:-1]) == 0


def test_pcm_decoding_ignores_a_trailing_partial_frame():
    """ffmpeg streams whole frames, but a half sample would corrupt every later value."""
    pcm = array("h", [16384] * 512).tobytes() + b"\x01"

    bars = waveform_from_pcm(pcm)

    assert bars is not None and len(bars) == WAVEFORM_POINTS


@pytest.mark.parametrize("pcm", [b"", b"\x01"], ids=["empty", "half-a-sample"])
def test_undecodable_pcm_reports_unmeasurable(pcm):
    assert waveform_from_pcm(pcm) is None


def test_flat_waveform_is_the_documented_fallback():
    assert FLAT_WAVEFORM == bytes([128] * WAVEFORM_POINTS)
