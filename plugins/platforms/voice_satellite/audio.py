"""Audio helpers for the voice satellite platform.

All functions are synchronous and CPU/subprocess-bound — callers must run
them via asyncio.to_thread (transcode) or accept the microseconds cost
inline (rms on 20ms chunks).
"""

import array
import math
import subprocess
import wave


def rms(pcm: bytes) -> int:
    """RMS amplitude of s16le mono PCM (0-32767)."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0
    return int(math.sqrt(sum(s * s for s in samples) / len(samples)))


def pcm_to_wav(
    pcm: bytes, output_path: str, rate: int = 16000, width: int = 2, channels: int = 1
) -> None:
    """Wrap raw PCM in a WAV container (STT providers accept any sample rate)."""
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def transcode_to_pcm(audio_path: str, rate: int = 22050) -> bytes:
    """Decode any audio file (mp3/wav/ogg...) to raw s16le mono PCM at `rate`."""
    from hermes_cli._subprocess_compat import windows_hide_flags

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", audio_path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(rate), "-ac", "1",
            "pipe:1",
        ],
        check=True,
        timeout=120,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        creationflags=windows_hide_flags(),
    )
    return result.stdout


class EndpointDetector:
    """Segments one utterance out of a PCM stream by RMS silence.

    Silence-only segments are discarded (leading silence rolls off); a
    segment ends when trailing silence exceeds ``silence_duration`` or the
    buffer hits ``max_utterance_seconds``.
    """

    def __init__(
        self,
        *,
        silence_threshold: int = 200,
        silence_duration: float = 1.2,
        min_speech_seconds: float = 0.5,
        max_utterance_seconds: float = 20.0,
    ):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_seconds = min_speech_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self.reset()

    def reset(self) -> None:
        self._buffer = bytearray()
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._total_seconds = 0.0

    def feed(self, pcm: bytes, seconds: float) -> bytes | None:
        self._buffer.extend(pcm)
        self._total_seconds += seconds
        if rms(pcm) >= self.silence_threshold:
            self._speech_seconds += seconds
            self._silence_seconds = 0.0
        else:
            self._silence_seconds += seconds

        if (
            self._silence_seconds >= self.silence_duration
            or self._total_seconds >= self.max_utterance_seconds
        ):
            return self._finish()
        return None

    def _finish(self) -> bytes | None:
        utterance = bytes(self._buffer)
        had_speech = self._speech_seconds >= self.min_speech_seconds
        self.reset()
        return utterance if had_speech else None
