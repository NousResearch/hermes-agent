"""Post-handler auto-TTS for ``BasePlatformAdapter``: synthesis of the voice reply to a voice note,
and delivery of a spoken reply's audio together with the chat text sent beside it.

Runs after the message handler returned, so the routed profile's secret scope that wrapped the
turn is gone. Under a multiplexed gateway an unscoped provider read (``GEMINI_BASE_URL``, a plugin
provider's key in ``is_available``) raises ``UnscopedSecretError`` and the voice reply is lost, so
synthesis re-enters the scope of the profile that serves ``source`` (#145151). The spoken-text
hooks run in that scope too, so they see the routed profile's plugins.
"""

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _profile_scope(adapter, source) -> contextlib.AbstractAsyncContextManager:
    """The runner's secret-hydrating scope for ``source``; a no-op without a runner."""
    resolve = getattr(getattr(adapter, "gateway_runner", None), "_async_profile_scope_for_source", None)
    if not callable(resolve) or source is None:
        return contextlib.nullcontext()
    return resolve(source)


def _synthesize(text: str, output_path: str) -> list[str]:
    """Worker-thread half: run the TTS tool and keep the files it reports that exist."""
    from tools.tts_tool import text_to_speech_tool
    tts_data = json.loads(text_to_speech_tool(text=text, output_path=output_path))
    if not tts_data.get("success", True):
        return []
    raw_tts_paths = tts_data.get("file_paths") or [tts_data.get("file_path")]
    return [str(path) for path in raw_tts_paths if path and Path(path).exists()]


def _hook_identity(adapter, session_key: str) -> dict[str, str]:
    """``platform`` / ``session_id`` kwargs for the spoken-text hooks."""
    peek = getattr(getattr(adapter, "_session_store", None), "peek_session_id", None)
    session_id = None
    with contextlib.suppress(Exception):  # a store hiccup must not cost the voice reply
        session_id = peek(session_key) if callable(peek) and session_key else None
    return {"platform": str(getattr(adapter.platform, "value", adapter.platform)),
            "session_id": session_id if isinstance(session_id, str) else ""}


async def synthesize_auto_tts(adapter, text_content: str, source, output_path_for,
                              session_key: str = "") -> tuple[list[str], Optional[str]]:
    """``(existing_paths, requested_path)``; empty/None on failure (logged, never raised).
    ``output_path_for(platform)`` builds the platform-aware temp path (``.ogg`` for Opus platforms)."""
    from tools.tts_spoken_text_hook import apply_spoken_text_hook
    paths: list[str] = []
    requested_path = None
    try:
        from tools.tts_tool import check_tts_requirements
        async with _profile_scope(adapter, source):
            if check_tts_requirements():
                speech_text = await asyncio.to_thread(
                    apply_spoken_text_hook, adapter.prepare_tts_text(text_content), surface="auto_tts",
                    **_hook_identity(adapter, session_key))
                if not speech_text:
                    raise ValueError("Empty text after markdown cleanup")
                requested_path = output_path_for(adapter.platform)
                paths = await asyncio.to_thread(_synthesize, speech_text, requested_path)
    except Exception as tts_err:  # an auto voice reply never fails the text reply it accompanies
        logger.warning("[%s] Auto-TTS failed: %s", adapter.name, tts_err, exc_info=True)
    return paths, requested_path


class SpokenReplyText:
    """Chat text of a reply the gateway also speaks. ``display`` is the text after
    ``transform_spoken_display_text``: the Telegram caption, and the text send once the audio is
    out. ``after_delivery()`` is the original reply instead while an auto-TTS file has not
    reached the chat, so a plugin that hides the text never costs the user both audio and text."""

    def __init__(self, original: str, display: str, audio_files: int, record_delivery: Callable) -> None:
        self.display = display
        self._original, self._undelivered, self._record_delivery = original, audio_files, record_delivery

    def record_delivery(self, result: Any) -> None:
        """``record_delivery`` for the ``_play_tts_file`` calls: also counts the files that went out."""
        self._record_delivery(result)
        self._undelivered -= bool(getattr(result, "success", False))

    def after_delivery(self) -> str:
        return self._original if self._undelivered > 0 else self.display


async def reply_text_after_speech(adapter, event, session_key: str, text_content: str, tts_paths: list[str],
                                  record_delivery: Callable) -> SpokenReplyText:
    """The chat text (and Telegram caption) for a reply Hermes also spoke — auto-TTS audio, or
    the runner's delivered voice reply — after ``transform_spoken_display_text``; unchanged
    otherwise."""
    from tools.tts_spoken_text_hook import apply_spoken_display_text_hook
    display = text_content
    surface = "auto_tts" if tts_paths else "voice_reply" if getattr(event, "_voice_reply_sent", False) else ""
    if surface:
        async with _profile_scope(adapter, event.source):
            display = await asyncio.to_thread(
                apply_spoken_display_text_hook, text_content, surface=surface, **_hook_identity(adapter, session_key))
    return SpokenReplyText(text_content, display, len(tts_paths), record_delivery)


async def play_spoken_reply(adapter, event, session_key: str, text_content: str, tts_paths: list[str],
                            metadata: dict, record_delivery: Callable) -> tuple[str, bool]:
    """Play the auto-TTS files (each removed afterwards) -> ``(chat text to send, caption_delivered)``.
    ``caption_delivered`` is True when the text rode along as the first file's Telegram caption. The
    chat text is the original reply unless every file reached the chat."""
    reply = await reply_text_after_speech(adapter, event, session_key, text_content, tts_paths, record_delivery)
    caption_delivered = False
    for index, path in enumerate(tts_paths):
        try:
            caption_delivered |= await adapter._play_tts_file(
                event, reply.display, path, index == 0, metadata, reply.record_delivery)
        finally:
            with contextlib.suppress(OSError):
                os.remove(path)
    return reply.after_delivery(), caption_delivered
