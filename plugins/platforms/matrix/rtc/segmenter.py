"""Turn segmentation and transcription for MatrixRTC audio.

Deliberately knows nothing about LiveKit or Matrix: it takes raw PCM keyed by a
speaker label and hands back completed utterances, so it is testable with synthetic
audio and no network.

The timers are the ones Discord voice channels have been running in production
(``plugins/platforms/discord/adapter.py`` ``VoiceReceiver``): 1.5 s of silence ends an
utterance, and anything under 0.5 s is noise. Only that logic is shared — none of
Discord's RTP/SSRC/DAVE machinery applies here, because LiveKit delivers decoded PCM
already attributed to a participant identity.

One thing does not port with those timers: **what silence looks like**. Discord sends RTP
only while a user speaks, so "no frame for 1.5 s" is a real gap. A decoded WebRTC sink
never stops — LiveKit hands us comfort noise for the whole call — so frame arrival says
nothing, and taking it as speech means no turn ever ends and nothing is ever transcribed.
``feed`` therefore gates on level (``SPEECH_RMS``): quiet frames are dropped and do not
touch the clock, which turns the continuous stream back into the Discord shape the timers
were written for.

Audio arrives as 16 kHz mono s16 because ``receiver.py`` asks the LiveKit SDK for that
rate (the wire is 48 kHz; the SDK's native resampler does the conversion). That is also
what Whisper wants, so ``pcm_to_wav`` is a stdlib ``wave`` write with no ffmpeg in the
path — unlike the Discord receiver, which must shell out to convert 48 kHz stereo.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from array import array
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# Ported verbatim from Discord's VoiceReceiver — same speech, same ears.
SILENCE_THRESHOLD = 1.5     # seconds of silence -> end of utterance
MIN_SPEECH_DURATION = 0.5   # minimum seconds to process (skip noise)
# RMS a frame must clear to count as somebody talking. Silence and comfort noise sit near
# zero, speech in the hundreds. Same floor the CLI voice recorder calls silence
# (``tools/voice_mode.SILENCE_RMS_THRESHOLD``) and the barge-in gate reuses; a live room is
# the thing that retunes it, which is what ``matrix.rtc.speech_rms`` is for.
SPEECH_RMS = 200

# What we ask LiveKit to deliver, and therefore what the buffers hold.
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # s16


def _rtc_config() -> dict:
    """``matrix.rtc`` from config.yaml. Behavioural knobs live there, never in ``.env``."""
    try:
        from hermes_cli.config import read_raw_config_readonly
        matrix_cfg = (read_raw_config_readonly() or {}).get("matrix") or {}
        return matrix_cfg.get("rtc") or {}
    except Exception as exc:  # config unreadable -> ship the defaults, don't crash the call
        logger.debug("MatrixRTC: config read failed, using defaults: %s", exc)
        return {}


def _positive_float(raw, fallback: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


class TurnSegmenter:
    """Buffers PCM per speaker and releases an utterance once they stop talking.

    Thread-safe: ``feed`` runs on whatever task drains the LiveKit audio stream while
    ``check_silence`` runs on the polling loop, exactly as on Discord.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS,
                 silence_threshold: Optional[float] = None,
                 min_speech_duration: Optional[float] = None,
                 speech_rms: Optional[float] = None):
        cfg = _rtc_config()
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold = _positive_float(
            silence_threshold if silence_threshold is not None
            else cfg.get("silence_threshold"), SILENCE_THRESHOLD)
        self.min_speech_duration = _positive_float(
            min_speech_duration if min_speech_duration is not None
            else cfg.get("min_speech_duration"), MIN_SPEECH_DURATION)
        self.speech_rms = _positive_float(
            speech_rms if speech_rms is not None else cfg.get("speech_rms"), SPEECH_RMS)
        self._lock = threading.Lock()
        self._buffers: dict[str, bytearray] = defaultdict(bytearray)
        self._last_frame_time: dict[str, float] = {}

    # --- ingest ---

    def feed(self, identity: str, pcm: bytes, now: Optional[float] = None) -> None:
        """Append decoded PCM for *identity*, if anyone is actually talking in it.

        Frames below ``speech_rms`` are dropped whole rather than buffered, and — the half
        that matters — never refresh the silence clock. A stream that keeps delivering them
        is a speaker who has stopped, which is exactly what ``check_silence`` is waiting
        for. Dropping them also keeps the buffer's length equal to the *speech* in it, so
        ``min_speech_duration`` still measures a cough rather than the hour of quiet after
        it. *now* is injectable so tests need no clock.
        """
        if not pcm or pcm_rms(pcm) < self.speech_rms:
            return
        stamp = time.monotonic() if now is None else now
        with self._lock:
            self._buffers[identity].extend(pcm)
            self._last_frame_time[identity] = stamp

    def _duration(self, buf) -> float:
        return pcm_duration(buf, self.sample_rate, self.channels)

    def _release(self, identity: str, buf: bytearray) -> tuple[str, bytes]:
        """Hand an utterance back, and say so at INFO — the live call's only breadcrumb
        between "audio arrived" and "Whisper returned something"."""
        pcm = bytes(buf)
        logger.info("MatrixRTC: utterance from %s released, %.2fs rms=%.0f",
                    identity, self._duration(pcm), pcm_rms(pcm))
        return identity, pcm

    # --- release ---

    def check_silence(self, now: Optional[float] = None) -> list[tuple[str, bytes]]:
        """Return ``(identity, pcm)`` for every speaker who has gone quiet long enough."""
        stamp = time.monotonic() if now is None else now
        completed: list[tuple[str, bytes]] = []
        with self._lock:
            for identity in list(self._buffers):
                silence = stamp - self._last_frame_time.get(identity, stamp)
                buf = self._buffers[identity]
                if silence < self.silence_threshold:
                    continue
                if self._duration(buf) >= self.min_speech_duration:
                    completed.append(self._release(identity, buf))
                    self._buffers[identity] = bytearray()
                    self._last_frame_time.pop(identity, None)
                elif silence >= self.silence_threshold * 2:
                    # Too short to be speech and long since abandoned: a cough, a door,
                    # a half-frame on join. Dropping it is what stops the map growing
                    # one dead entry per noise burst for the life of the call.
                    self._buffers.pop(identity, None)
                    self._last_frame_time.pop(identity, None)
        return completed

    def flush_pending(self) -> list[tuple[str, bytes]]:
        """Drain every buffer, returning the ones long enough to be speech. Used on leave."""
        completed: list[tuple[str, bytes]] = []
        with self._lock:
            for identity, buf in list(self._buffers.items()):
                if self._duration(buf) >= self.min_speech_duration:
                    completed.append(self._release(identity, buf))
                self._buffers.pop(identity, None)
                self._last_frame_time.pop(identity, None)
        return completed

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._last_frame_time.clear()


def pcm_duration(pcm, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> float:
    """Seconds of audio in a raw s16 buffer."""
    return len(pcm) / (sample_rate * channels * SAMPLE_WIDTH)


def pcm_rms(pcm: bytes) -> float:
    """Level of a raw s16 frame, 0..32767. Silence is ~0; speech is hundreds.

    Stdlib arithmetic because ``audioop`` was removed in 3.13 and numpy is not a dependency
    of this path. ``array("h")`` reads native byte order, which is the little-endian s16 the
    LiveKit SDK hands us and ``pcm_to_wav`` writes.
    """
    samples = array("h", bytes(pcm)[:len(pcm) - len(pcm) % SAMPLE_WIDTH])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def pcm_to_wav(pcm: bytes, output_path: str, sample_rate: int = SAMPLE_RATE,
               channels: int = CHANNELS) -> str:
    """Wrap raw s16 PCM in a WAV container. Stdlib only — no ffmpeg on this path."""
    import wave
    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output_path


def transcribe_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE,
                   channels: int = CHANNELS) -> Optional[str]:
    """PCM -> WAV -> Whisper -> text, or None when there is nothing worth passing on.

    Blocking (STT is CPU-bound); callers on the event loop must use ``asyncio.to_thread``.
    """
    import os
    import tempfile
    from tools.transcription_tools import transcribe_audio
    from tools.voice_mode_transcript import is_whisper_hallucination

    handle = tempfile.NamedTemporaryFile(suffix=".wav", prefix="matrix_rtc_", delete=False)
    wav_path = handle.name
    handle.close()
    try:
        pcm_to_wav(pcm, wav_path, sample_rate, channels)
        result = transcribe_audio(wav_path, source="voice_mode")
        if not result.get("success"):
            logger.debug("MatrixRTC transcription failed: %s", result.get("error"))
            return None
        transcript = (result.get("transcript") or "").strip()
        # Whisper invents "Thank you." / subtitle credits out of near-silence; the same
        # filter the CLI and Discord voice paths use keeps that out of the session.
        if not transcript or is_whisper_hallucination(transcript):
            return None
        return transcript
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
