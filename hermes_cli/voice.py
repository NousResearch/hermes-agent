"""Process-wide voice recording + TTS API for the TUI gateway.

Wraps ``tools.voice_mode`` (recording/transcription) and ``tools.tts_tool``
(text-to-speech) behind idempotent, stateful entry points that the gateway's
``voice.record``, ``voice.toggle``, and ``voice.tts`` JSON-RPC handlers can
call from a dedicated thread. The gateway imports this module lazily so that
missing optional audio deps (sounddevice, faster-whisper, numpy) surface as
an ``ImportError`` at call time, not at startup.

Two usage modes are exposed:

* **Push-to-talk** (``start_recording`` / ``stop_and_transcribe``) — single
  manually-bounded capture used when the caller drives the start/stop pair
  explicitly.
* **Continuous (VAD)** (``start_continuous`` / ``stop_continuous``) — mirrors
  the classic CLI voice mode: recording auto-stops on silence, transcribes,
  hands the result to a callback, and then auto-restarts for the next turn.
  Three consecutive no-speech cycles stop the loop and fire
  ``on_silent_limit`` so the UI can turn the mode off.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from typing import Any, Callable, Optional

from tools.voice_mode import (
    create_audio_recorder,
    is_whisper_hallucination,
    play_audio_file,
    transcribe_recording,
)

logger = logging.getLogger(__name__)


def _debug(msg: str) -> None:
    """Emit a debug breadcrumb when HERMES_VOICE_DEBUG=1.

    Goes to stderr so the TUI gateway wraps it as a gateway.stderr event,
    which createGatewayEventHandler shows as an Activity line — exactly
    what we need to diagnose "why didn't the loop auto-restart?" in the
    user's real terminal without shipping a separate debug RPC.
    """
    if os.environ.get("HERMES_VOICE_DEBUG", "").strip() == "1":
        print(f"[voice] {msg}", file=sys.stderr, flush=True)


def _beeps_enabled() -> bool:
    """CLI parity: voice.beep_enabled in config.yaml (default True)."""
    try:
        from hermes_cli.config import load_config

        voice_cfg = load_config().get("voice", {})
        if isinstance(voice_cfg, dict):
            return bool(voice_cfg.get("beep_enabled", True))
    except Exception:
        pass
    return True


def _play_beep(frequency: int, count: int = 1) -> None:
    """Audible cue matching cli.py's record/stop beeps.

    880 Hz single-beep on start (cli.py:_voice_start_recording line 7532),
    660 Hz double-beep on stop (cli.py:_voice_stop_and_transcribe line 7585).
    Best-effort — sounddevice failures are silently swallowed so the
    voice loop never breaks because a speaker was unavailable.
    """
    if not _beeps_enabled():
        return
    try:
        from tools.voice_mode import play_beep

        play_beep(frequency=frequency, count=count)
    except Exception as e:
        _debug(f"beep {frequency}Hz failed: {e}")

# ── Push-to-talk state ───────────────────────────────────────────────
_recorder = None
_recorder_lock = threading.Lock()

# ── Continuous (VAD) state ───────────────────────────────────────────
_continuous_lock = threading.Lock()
_continuous_active = False
_continuous_recorder: Any = None
_continuous_on_transcript: Optional[Callable[[str], None]] = None
_continuous_on_status: Optional[Callable[[str], None]] = None
_continuous_on_silent_limit: Optional[Callable[[], None]] = None
_continuous_no_speech_count = 0
_CONTINUOUS_NO_SPEECH_LIMIT = 3


# ── Push-to-talk API ─────────────────────────────────────────────────


def start_recording() -> None:
    """Begin capturing from the default input device (push-to-talk).

    Idempotent — calling again while a recording is in progress is a no-op.
    """
    global _recorder

    with _recorder_lock:
        if _recorder is not None and getattr(_recorder, "is_recording", False):
            return
        rec = create_audio_recorder()
        rec.start()
        _recorder = rec


def stop_and_transcribe() -> Optional[str]:
    """Stop the active push-to-talk recording, transcribe, return text.

    Returns ``None`` when no recording is active, when the microphone
    captured no speech, or when Whisper returned a known hallucination.
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
    finally:
        try:
            if os.path.isfile(wav_path):
                os.unlink(wav_path)
        except Exception:
            pass

    # transcribe_recording returns {"success": bool, "transcript": str, ...}
    # — matches cli.py:_voice_stop_and_transcribe's result.get("transcript").
    if not result.get("success"):
        return None
    text = (result.get("transcript") or "").strip()
    if not text or is_whisper_hallucination(text):
        return None

    return text


# ── Continuous (VAD) API ─────────────────────────────────────────────


def start_continuous(
    on_transcript: Callable[[str], None],
    on_status: Optional[Callable[[str], None]] = None,
    on_silent_limit: Optional[Callable[[], None]] = None,
    silence_threshold: int = 200,
    silence_duration: float = 3.0,
) -> None:
    """Start a VAD-driven continuous recording loop.

    The loop calls ``on_transcript(text)`` each time speech is detected and
    transcribed successfully, then auto-restarts. After
    ``_CONTINUOUS_NO_SPEECH_LIMIT`` consecutive silent cycles (no speech
    picked up at all) the loop stops itself and calls ``on_silent_limit``
    so the UI can reflect "voice off". Idempotent — calling while already
    active is a no-op.

    ``on_status`` is called with ``"listening"`` / ``"transcribing"`` /
    ``"idle"`` so the UI can show a live indicator.
    """
    global _continuous_active, _continuous_recorder
    global _continuous_on_transcript, _continuous_on_status, _continuous_on_silent_limit
    global _continuous_no_speech_count

    with _continuous_lock:
        if _continuous_active:
            _debug("start_continuous: already active — no-op")
            return
        _continuous_active = True
        _continuous_on_transcript = on_transcript
        _continuous_on_status = on_status
        _continuous_on_silent_limit = on_silent_limit
        _continuous_no_speech_count = 0

        if _continuous_recorder is None:
            _continuous_recorder = create_audio_recorder()

        _continuous_recorder._silence_threshold = silence_threshold
        _continuous_recorder._silence_duration = silence_duration
        rec = _continuous_recorder

    _debug(
        f"start_continuous: begin (threshold={silence_threshold}, duration={silence_duration}s)"
    )

    # CLI parity: single 880 Hz beep *before* opening the stream — placing
    # the beep after stream.start() on macOS triggers a CoreAudio conflict
    # (cli.py:7528 comment).
    _play_beep(frequency=880, count=1)

    try:
        rec.start(on_silence_stop=_continuous_on_silence)
    except Exception as e:
        logger.error("failed to start continuous recording: %s", e)
        _debug(f"start_continuous: rec.start raised {type(e).__name__}: {e}")
        with _continuous_lock:
            _continuous_active = False
        raise

    if on_status:
        try:
            on_status("listening")
        except Exception:
            pass


def stop_continuous() -> None:
    """Stop the active continuous loop and release the microphone.

    Idempotent — calling while not active is a no-op. Any in-flight
    transcription completes but its result is discarded (the callback
    checks ``_continuous_active`` before firing).
    """
    global _continuous_active, _continuous_on_transcript
    global _continuous_on_status, _continuous_on_silent_limit
    global _continuous_recorder, _continuous_no_speech_count

    with _continuous_lock:
        if not _continuous_active:
            return
        _continuous_active = False
        rec = _continuous_recorder
        on_status = _continuous_on_status
        _continuous_on_transcript = None
        _continuous_on_status = None
        _continuous_on_silent_limit = None
        _continuous_no_speech_count = 0

    if rec is not None:
        try:
            # cancel() (not stop()) discards buffered frames — the loop
            # is over, we don't want to transcribe a half-captured turn.
            rec.cancel()
        except Exception as e:
            logger.warning("failed to cancel recorder: %s", e)

    # Audible "recording stopped" cue (CLI parity: same 660 Hz × 2 the
    # silence-auto-stop path plays).
    _play_beep(frequency=660, count=2)

    if on_status:
        try:
            on_status("idle")
        except Exception:
            pass


def is_continuous_active() -> bool:
    """Whether a continuous voice loop is currently running."""
    with _continuous_lock:
        return _continuous_active


def _continuous_on_silence() -> None:
    """AudioRecorder silence callback — runs in a daemon thread.

    Stops the current capture, transcribes, delivers the text via
    ``on_transcript``, and — if the loop is still active — starts the
    next capture. Three consecutive silent cycles end the loop.
    """
    global _continuous_active, _continuous_no_speech_count

    _debug("_continuous_on_silence: fired")

    with _continuous_lock:
        if not _continuous_active:
            _debug("_continuous_on_silence: loop inactive — abort")
            return
        rec = _continuous_recorder
        on_transcript = _continuous_on_transcript
        on_status = _continuous_on_status
        on_silent_limit = _continuous_on_silent_limit

    if rec is None:
        _debug("_continuous_on_silence: no recorder — abort")
        return

    if on_status:
        try:
            on_status("transcribing")
        except Exception:
            pass

    wav_path = rec.stop()
    # Peak RMS is the critical diagnostic when stop() returns None despite
    # the VAD firing — tells us at a glance whether the mic was too quiet
    # for SILENCE_RMS_THRESHOLD (200) or the VAD + peak checks disagree.
    peak_rms = getattr(rec, "_peak_rms", -1)
    _debug(
        f"_continuous_on_silence: rec.stop -> {wav_path!r} (peak_rms={peak_rms})"
    )

    # CLI parity: double 660 Hz beep after the stream stops (safe from the
    # CoreAudio conflict that blocks pre-start beeps).
    _play_beep(frequency=660, count=2)

    transcript: Optional[str] = None

    if wav_path:
        try:
            result = transcribe_recording(wav_path)
            # transcribe_recording returns {"success": bool, "transcript": str,
            # "error": str?} — NOT {"text": str}.  Using the wrong key silently
            # produced empty transcripts even when Groq/local STT returned fine,
            # which masqueraded as "not hearing the user" to the caller.
            success = bool(result.get("success"))
            text = (result.get("transcript") or "").strip()
            err = result.get("error")
            _debug(
                f"_continuous_on_silence: transcribe -> success={success} "
                f"text={text!r} err={err!r}"
            )
            if success and text and not is_whisper_hallucination(text):
                transcript = text
        except Exception as e:
            logger.warning("continuous transcription failed: %s", e)
            _debug(f"_continuous_on_silence: transcribe raised {type(e).__name__}: {e}")
        finally:
            try:
                if os.path.isfile(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass

    with _continuous_lock:
        if not _continuous_active:
            # User stopped us while we were transcribing — discard.
            _debug("_continuous_on_silence: stopped during transcribe — no restart")
            return
        if transcript:
            _continuous_no_speech_count = 0
        else:
            _continuous_no_speech_count += 1
        should_halt = _continuous_no_speech_count >= _CONTINUOUS_NO_SPEECH_LIMIT
        no_speech = _continuous_no_speech_count

    if transcript and on_transcript:
        try:
            on_transcript(transcript)
        except Exception as e:
            logger.warning("on_transcript callback raised: %s", e)

    if should_halt:
        _debug(f"_continuous_on_silence: {no_speech} silent cycles — halting")
        with _continuous_lock:
            _continuous_active = False
            _continuous_no_speech_count = 0
        if on_silent_limit:
            try:
                on_silent_limit()
            except Exception:
                pass
        try:
            rec.cancel()
        except Exception:
            pass
        if on_status:
            try:
                on_status("idle")
            except Exception:
                pass
        return

    # Restart for the next turn.
    _debug(f"_continuous_on_silence: restarting loop (no_speech={no_speech})")
    _play_beep(frequency=880, count=1)
    try:
        rec.start(on_silence_stop=_continuous_on_silence)
    except Exception as e:
        logger.error("failed to restart continuous recording: %s", e)
        _debug(f"_continuous_on_silence: restart raised {type(e).__name__}: {e}")
        with _continuous_lock:
            _continuous_active = False
        return

    if on_status:
        try:
            on_status("listening")
        except Exception:
            pass


# ── TTS API ──────────────────────────────────────────────────────────


def speak_text(text: str) -> None:
    """Synthesize ``text`` with the configured TTS provider and play it.

    The gateway spawns a daemon thread to call this so the RPC returns
    immediately. Failures are logged and swallowed.
    """
    if not text or not text.strip():
        return

    # Lazy import — tts_tool pulls optional provider SDKs.
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
