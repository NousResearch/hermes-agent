"""Audio helpers for Discord voice channel <-> OpenAI Realtime streaming."""

from __future__ import annotations

import logging
import struct
import threading
from typing import Any, cast

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised through tests with a fake discord module.
    import discord
except ImportError:  # pragma: no cover
    discord = None


DISCORD_SAMPLE_RATE = 48_000
REALTIME_SAMPLE_RATE = 24_000
DISCORD_CHANNELS = 2
SAMPLE_WIDTH_BYTES = 2
FRAME_DURATION_SECONDS = 0.02
DISCORD_PCM_FRAME_BYTES = int(
    DISCORD_SAMPLE_RATE
    * FRAME_DURATION_SECONDS
    * DISCORD_CHANNELS
    * SAMPLE_WIDTH_BYTES
)


_DiscordAudioSourceBase: Any = cast(Any, object)
if discord is not None:
    _maybe_audio_source = getattr(discord, "AudioSource", None)
    if isinstance(_maybe_audio_source, type):
        _DiscordAudioSourceBase = cast(Any, _maybe_audio_source)


def discord_pcm_to_realtime_pcm(pcm_48k_stereo: bytes) -> bytes:
    """Convert Discord 48 kHz stereo PCM16 into Realtime 24 kHz mono PCM16."""
    if not pcm_48k_stereo:
        return b""
    # Discord and Realtime use a fixed 2:1 sample-rate relationship.  Use the
    # deterministic integer-ratio converter so each stateless 20 ms Discord
    # frame maps to an exact 20 ms Realtime frame; audioop.ratecv can drop a
    # boundary sample without retained state, which breaks Discord frame sizing.
    return _downsample_48k_stereo_to_24k_mono(pcm_48k_stereo)


def realtime_pcm_to_discord_pcm(pcm_24k_mono: bytes) -> bytes:
    """Convert Realtime 24 kHz mono PCM16 into Discord 48 kHz stereo PCM16."""
    if not pcm_24k_mono:
        return b""
    return _upsample_24k_mono_to_48k_stereo(pcm_24k_mono)


def _downsample_48k_stereo_to_24k_mono(pcm: bytes) -> bytes:
    """Small deterministic fallback for the fixed Discord -> Realtime path."""
    usable_len = len(pcm) - (len(pcm) % 4)
    output = bytearray()
    frame_index = 0
    for left, right in struct.iter_unpack("<hh", pcm[:usable_len]):
        if frame_index % 2 == 0:
            mixed = int((left + right) / 2)
            output.extend(struct.pack("<h", mixed))
        frame_index += 1
    return bytes(output)


def _upsample_24k_mono_to_48k_stereo(pcm: bytes) -> bytes:
    """Small deterministic fallback for the fixed Realtime -> Discord path."""
    usable_len = len(pcm) - (len(pcm) % 2)
    output = bytearray()
    for (sample,) in struct.iter_unpack("<h", pcm[:usable_len]):
        stereo_sample = struct.pack("<hh", sample, sample)
        output.extend(stereo_sample)
        output.extend(stereo_sample)
    return bytes(output)


class RealtimeDiscordAudioSource(_DiscordAudioSourceBase):
    """Discord AudioSource backed by a streaming PCM queue.

    Discord's voice player asks for exactly one 20 ms frame on each ``read``.
    Returning silence while idle keeps playback alive so Realtime output can
    resume without restarting the voice player.
    """

    def __init__(self, *, input_rate: int = REALTIME_SAMPLE_RATE, discord_rate: int = DISCORD_SAMPLE_RATE):
        self.input_rate = input_rate
        self.discord_rate = discord_rate
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._closed = False

    def is_opus(self) -> bool:
        return False

    def read(self) -> bytes:
        if self._closed:
            return b""
        with self._lock:
            if len(self._buffer) >= DISCORD_PCM_FRAME_BYTES:
                frame = bytes(self._buffer[:DISCORD_PCM_FRAME_BYTES])
                del self._buffer[:DISCORD_PCM_FRAME_BYTES]
                return frame
            if self._buffer:
                frame = bytes(self._buffer)
                self._buffer.clear()
                return frame.ljust(DISCORD_PCM_FRAME_BYTES, b"\x00")
        return b"\x00" * DISCORD_PCM_FRAME_BYTES

    def push_pcm_24k_mono(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        if self.input_rate != REALTIME_SAMPLE_RATE or self.discord_rate != DISCORD_SAMPLE_RATE:
            logger.warning(
                "RealtimeDiscordAudioSource currently supports 24k input and 48k Discord output only",
            )
        converted = realtime_pcm_to_discord_pcm(pcm)
        with self._lock:
            self._buffer.extend(converted)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def close(self) -> None:
        self._closed = True
        self.clear()
