"""Telegram's buffered PCM sink for the native streaming-TTS adapter contract.

Telegram accepts complete Ogg/Opus voice messages, not a live PCM track.  The
adapter therefore buffers one spoken clause, transcodes it in memory, and sends
it before the next clause is synthesized.  No temporary audio files or second
TTS provider are involved.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from gateway.platforms.base_streaming_tts import AudioFormat, StreamingTTSHandle


@dataclass
class TelegramStreamingTTSHandle(StreamingTTSHandle):
    """PCM buffered until the consumer reaches a spoken-clause boundary."""

    defer_audible_until_flush: bool = True
    pcm: bytearray = field(default_factory=bytearray)
    metadata: Optional[dict[str, Any]] = None


def supports_pcm_format(audio_format: AudioFormat) -> bool:
    """Telegram's PCM encoder accepts the native mono signed-16-bit stream."""
    return (
        audio_format.sample_rate > 0
        and audio_format.channels == 1
        and audio_format.sample_width == 2
        and shutil.which("ffmpeg") is not None
    )


def pcm_duration_seconds(pcm: bytes | bytearray, audio_format: AudioFormat) -> int:
    """Whole-second duration for Telegram's voice-player metadata."""
    bytes_per_second = (
        audio_format.sample_rate * audio_format.channels * audio_format.sample_width
    )
    return max(1, round(len(pcm) / bytes_per_second)) if bytes_per_second else 1


def encode_pcm_ogg_opus(pcm: bytes, audio_format: AudioFormat, *, timeout: int = 30) -> bytes:
    """Encode raw PCM as an in-memory Ogg/Opus voice bubble with Hermes' ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for Telegram streaming TTS")
    result = subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "s16le",
            "-ar", str(audio_format.sample_rate),
            "-ac", str(audio_format.channels),
            "-i", "pipe:0",
            "-acodec", "libopus",
            "-ac", "1",
            "-b:a", "32k",
            "-vbr", "on",
            "-application", "voip",
            "-compression_level", "10",
            "-f", "ogg",
            "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.startswith(b"OggS"):
        detail = (result.stderr or b"").decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"PCM to Ogg/Opus conversion failed: {detail}")
    return result.stdout
