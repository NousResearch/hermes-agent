"""One inbound media clip turned into agent-readable text.

Shared by the 1:1 inbound path (``gateway/run_inbound.py``) and hosted-room
attachment staging (``tui_gateway/hosted_room_server_rpc.py``): a voice note must
reach the model as words on both surfaces, and every STT failure must leave the
same neutral marker instead of silence.

Only audio is transcribed.  Video keeps the native contract of
``_prepend_inbound_media_file_notes``: the bytes stay on disk and the model
inspects them with the existing media tools when the request depends on them.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

# STT reads the container from the file extension, so bytes whose stored name
# carries none — a voice note uploaded as "voice", a Desktop drop renamed by the
# user — are transcribed through a temporary copy named after their declared MIME.
_MIME_SUFFIXES: Dict[str, str] = {
    "audio/aac": ".aac", "audio/flac": ".flac", "audio/x-flac": ".flac",
    "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/ogg": ".ogg", "audio/opus": ".opus", "audio/vorbis": ".ogg",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/webm": ".webm",
}

EMPTY_TRANSCRIPT_NOTE = (
    "[The user sent a voice message but it came through "
    "empty or inaudible — speech-to-text returned no "
    "words. Do not guess at the content; ask the user "
    "to resend or type it out.]"
)

UNAVAILABLE_STT_NOTE = "[voice message could not be transcribed]"


def untranscribed_note(agent_path: str) -> str:
    """One minimal neutral marker for every STT failure. Never mention "no STT provider" or setup
    steps — persisted in history they make the model keep volunteering STT-setup advice."""
    return f"[voice message could not be transcribed automatically; the audio is available at: {agent_path}]"


def untranscribed_audio_note(path: str) -> str:
    """``untranscribed_note`` for a gateway cache path, translated into the agent's view."""
    from tools.credential_files import to_agent_visible_cache_path
    return untranscribed_note(to_agent_visible_cache_path(os.path.abspath(path)))


def is_audio_mime(mime: Any) -> bool:
    """True for the media whose content this module can actually inline."""
    return str(mime or "").strip().lower().split(";")[0].strip().startswith("audio/")


def transcribe_clip(
    path: str,
    transcribe_audio: Callable[..., Dict[str, Any]],
    transcribe_audio_local_fallback: Callable[..., Dict[str, Any]],
    *,
    agent_path: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """``(transcript_or_None, note)`` for one clip via configured STT with local fallback.

    Blocking: inbound callers own the thread hop.  ``agent_path`` names the clip the way the
    reader can reach it (a room workspace ref, say); without one the gateway cache translation
    is used.
    """
    result = transcribe_audio(path, None, "gateway")
    if not result.get("success"):
        fallback = transcribe_audio_local_fallback(path)
        if fallback.get("success"):
            logger.info("Configured STT failed for %s; recovered with local STT", path)
            result = fallback
    if not result["success"]:
        logger.info("Voice transcription failed for %s: %s", path, result.get("error", "unknown error"))
        return None, (untranscribed_note(agent_path) if agent_path else untranscribed_audio_note(path))
    transcript = result["transcript"]
    # STT may return success=True with an empty/whitespace transcript (silence, cut-off);
    # empty quotes make the agent reply to nothing and can loop, so emit a sentinel note.
    # See #41603.
    if not (transcript or "").strip():
        return None, EMPTY_TRANSCRIPT_NOTE
    # Plain quoted line: a "The user sent a voice message..." wrapper read as a meta-instruction
    # and made the LLM comment on voice mode instead.
    return transcript, f'"{transcript}"'


def stt_is_enabled() -> bool:
    """Honour the same gateway switch the 1:1 voice path honours; an unreadable config means on."""
    try:
        from gateway.config import load_gateway_config
        return bool(getattr(load_gateway_config(), "stt_enabled", True))
    except Exception:
        logger.debug("STT switch unreadable; treating transcription as enabled", exc_info=True)
        return True


@contextmanager
def _decodable_clip(path: str, mime: Any) -> Iterator[str]:
    """Yield a path STT can decode: the file itself, else a temporary copy named after *mime*."""
    from tools.transcription_common import SUPPORTED_FORMATS
    suffix = _MIME_SUFFIXES.get(str(mime or "").strip().lower().split(";")[0].strip())
    if suffix is None or Path(path).suffix.lower() in SUPPORTED_FORMATS:
        yield path
        return
    work_dir = tempfile.mkdtemp(prefix="hermes-media-stt-")
    try:
        clip = os.path.join(work_dir, f"clip{suffix}")
        shutil.copyfile(path, clip)
        yield clip
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def transcribe_media_file(path: str, *, mime: Any, agent_path: str) -> str:
    """Note for one already-stored media file, or ``""`` when its content is not STT input.

    Never raises: a turn carrying a voice message must reach the model either as words or as
    the neutral marker, never as a failed turn and never as silence.
    """
    if not is_audio_mime(mime) or not stt_is_enabled():
        return ""
    try:
        from tools.transcription_tools import transcribe_audio, transcribe_audio_local_fallback
    except ModuleNotFoundError as exc:
        logger.error("Transcription module unavailable: %s", exc)
        return UNAVAILABLE_STT_NOTE
    try:
        with _decodable_clip(path, mime) as clip:
            _transcript, note = transcribe_clip(
                clip, transcribe_audio, transcribe_audio_local_fallback, agent_path=agent_path)
        return note
    except Exception as exc:
        logger.error("Transcription error: %s", exc)
        return untranscribed_note(agent_path)
