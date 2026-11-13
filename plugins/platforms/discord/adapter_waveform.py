"""Loudness envelope for Discord voice-message previews.

Discord renders a voice message's ``waveform`` field as the bars beside the play
button: a base64-encoded ``uint8`` bytearray of at most 256 datapoints, one per
window of audio, sampled at most once per 100 ms. Clients draw real speech as
jagged bars; a uniform array renders as a featureless block that reads as a
loading bar rather than a recording.

This module turns the audible audio into that envelope: decode once to 8 kHz
mono PCM (an envelope needs no fidelity, and 8 kHz keeps the decode cheap),
split into 256 equal buckets, and map each bucket's RMS to dBFS on a 0-255
scale with a -60 dBFS floor. Silence is the floor, full scale is 255, so pauses
between words produce the gaps that make the waveform readable.

Everything here is best-effort: no ffmpeg, an undecodable file or empty audio
returns ``None`` and the caller keeps the legacy flat array. Nothing about the
voice message itself depends on the envelope, so a missing waveform must never
fail or delay a delivery.
"""

from __future__ import annotations

import array
import logging
import math
import shutil
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Discord's documented cap; also the number of bars the client draws.
WAVEFORM_POINTS = 256

# The flat array every voice message shipped before the envelope existed: 256
# samples at mid-scale. Kept as the fallback so a decode failure is invisible.
FLAT_WAVEFORM = bytes([128] * WAVEFORM_POINTS)

_SAMPLE_RATE = 8000      # envelope resolution only; opus/mp3 decode fine below 24 kHz
_TIMEOUT_S = 30
_DB_FLOOR = -60.0        # quietest bucket we distinguish, matching common client meters
_FULL_SCALE = 32768.0    # s16 peak


def voice_waveform(audio_path: str) -> Optional[bytes]:
    """``WAVEFORM_POINTS`` unsigned loudness bytes for ``audio_path``, or None.

    None means "could not measure" (no ffmpeg, undecodable, empty audio) and is
    never an error: the caller falls back to :data:`FLAT_WAVEFORM`.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.debug("No ffmpeg on PATH; sending the flat voice-message waveform")
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", audio_path,
             "-ac", "1", "-ar", str(_SAMPLE_RATE), "-f", "s16le", "-"],
            capture_output=True, timeout=_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Voice-message waveform decode failed for %s: %s", audio_path, exc)
        return None
    if result.returncode != 0:
        logger.debug(
            "Voice-message waveform decode exited %s for %s: %s",
            result.returncode, audio_path,
            result.stderr[-200:].decode("utf-8", "replace"),
        )
        return None
    return waveform_from_pcm(result.stdout)


def waveform_from_pcm(pcm: bytes) -> Optional[bytes]:
    """Envelope for raw little-endian s16 mono PCM (what ffmpeg streams above)."""
    # ffmpeg writes whole frames; a trailing odd byte would corrupt every sample.
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return None
    if sys.byteorder == "big":  # array('h') adopts host order; the stream is little-endian
        samples.byteswap()
    return envelope(samples)


def envelope(samples: array.array) -> bytes:
    """One unsigned loudness value per bucket, ``WAVEFORM_POINTS`` of them.

    RMS (not peak) per bucket, converted to dBFS and clamped at :data:`_DB_FLOOR`,
    so an envelope reflects perceived loudness rather than single-sample spikes.
    """
    total = len(samples)
    step = total / WAVEFORM_POINTS
    span = -_DB_FLOOR
    bars = bytearray(WAVEFORM_POINTS)
    for index in range(WAVEFORM_POINTS):
        start = int(index * step)
        end = max(start + 1, int((index + 1) * step))
        bucket = samples[start:end]
        if not bucket:
            continue
        power = 0
        for sample in bucket:
            power += sample * sample
        rms = math.sqrt(power / len(bucket))
        dbfs = 20.0 * math.log10(rms / _FULL_SCALE) if rms > 0 else _DB_FLOOR
        # A bucket quieter than the floor is indistinguishable from silence to the client.
        dbfs = max(dbfs, _DB_FLOOR)
        bars[index] = round(255.0 * (dbfs + span) / span)
    return bytes(bars)
