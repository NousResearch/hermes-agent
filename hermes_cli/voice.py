"""Process-wide voice recording + TTS API for the TUI gateway.

Wraps ``tools.voice_mode`` (recording/transcription) and ``tools.tts_tool``
(text-to-speech) behind idempotent, stateful entry points that the gateway's
``voice.record`` and ``voice.tts`` JSON-RPC handlers can call from a
dedicated thread. The gateway imports this module lazily so missing optional
audio deps (sounddevice, faster-whisper, numpy) surface as an ``ImportError``
at call time, not at startup.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from tools.voice_mode import (
    create_audio_recorder,
    is_whisper_hallucination,
    play_audio_file,
    transcribe_recording,
)

logger = logging.getLogger(__name__)

_recorder = None
_recorder_lock = threading.Lock()


def start_recording() -> None:
    """Begin capturing from the default input device.

    Idempotent — calling again while a recording is in progress is a no-op,
    which matches the TUI's toggle semantics (Ctrl+B starts, Ctrl+B stops).
    """
    global _recorder

    with _recorder_lock:
        if _recorder is not None and getattr(_recorder, "is_recording", False):
            return
        rec = create_audio_recorder()
        # No silence callback: the TUI drives start/stop explicitly via
        # the voice.record RPC. VAD auto-stop is a CLI-mode feature.
        rec.start()
        _recorder = rec


def stop_and_transcribe() -> Optional[str]:
    """Stop the active recording, transcribe it, and return the text.

    Returns ``None`` when no recording is active, when the microphone
    captured no speech, or when Whisper returned a known hallucination
    token (silence artefacts like "Thanks for watching!"). The caller
    treats ``None`` as "no speech detected" and leaves the composer
    untouched.
    """
    global _recorder

    with _recorder_lock:
        rec = _recorder
        _recorder = None

    if rec is None:
        return None

    wav_path = rec.stop()
    if not wav_path:
        return None

    try:
        result = transcribe_recording(wav_path)
    except Exception as e:
        logger.warning("voice transcription failed: %s", e)
        return None

    text = (result.get("text") or "").strip()
    if not text or is_whisper_hallucination(text):
        return None

    return text


def speak_text(text: str) -> None:
    """Synthesize ``text`` with the configured TTS provider and play it.

    The gateway spawns a daemon thread to call this so the RPC returns
    immediately. Failures are logged and swallowed — the UI already
    acknowledged "speaking" by the time we get here.
    """
    if not text or not text.strip():
        return

    # Lazy import — tts_tool pulls optional provider SDKs (OpenAI,
    # ElevenLabs, etc.) and config-reading machinery that we don't
    # want to load at module import time.
    from tools.tts_tool import text_to_speech_tool

    try:
        raw = text_to_speech_tool(text)
    except Exception as e:
        logger.warning("TTS synthesis failed: %s", e)
        return

    try:
        result = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        logger.warning("TTS returned non-JSON result")
        return

    if not isinstance(result, dict):
        return

    file_path = result.get("file_path")
    if not file_path:
        err = result.get("error") or "no file_path in TTS result"
        logger.warning("TTS succeeded but produced no audio: %s", err)
        return

    play_audio_file(file_path)
