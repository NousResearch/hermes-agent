"""Inbound Telegram file fetch with a bounded retry, and replied-to media attachment.

``get_file()`` + ``download_as_bytearray()`` is a pure read: replaying it can never post anything,
so unlike the SEND paths (``telegram_send_errors``) a timeout is safe to retry. A stalled getFile
socket answers in milliseconds on the next attempt (#150861), so one dropped read must not cost the
user the attachment. ``BadRequest`` (a bad file_id, a too-big file) is a permanent answer and PTB
makes it a ``NetworkError`` subclass, so it is excluded explicitly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("plugins.platforms.telegram.adapter")

# Backoff (seconds) before each further attempt of one inbound file fetch.
FETCH_RETRY_DELAYS = (2.0, 5.0)


def is_retryable_fetch_error(exc: Exception) -> bool:
    """A transport failure (PTB ``NetworkError``/``TimedOut`` or a bare httpx/OS error) other than a
    Bot API refusal or a pool timeout (the local pool is exhausted; an immediate replay stalls too)."""
    from plugins.platforms.telegram.adapter import TelegramAdapter as tg
    if tg._is_bad_request_error(exc) or tg._looks_like_auth_error(exc) or tg._looks_like_pool_timeout(exc):
        return False
    return tg._looks_like_network_error(exc) or type(exc).__name__ in {
        "ReadTimeout", "ReadError", "ConnectTimeout", "ConnectError", "RemoteProtocolError"}


async def fetch_telegram_file(source: Any, what: str) -> tuple[Any, bytes]:
    """``(file_obj, bytes)`` for one attachment. ``get_file()`` and the download retry as one unit:
    a half-read download is not resumable."""
    from plugins.platforms.telegram.adapter import _redact_telegram_error_text
    delays = list(FETCH_RETRY_DELAYS)
    attempt = 0
    while True:
        attempt += 1
        try:
            file_obj = await source.get_file()
            return file_obj, bytes(await file_obj.download_as_bytearray())
        except Exception as exc:
            if not delays or not is_retryable_fetch_error(exc):
                raise
            delay = delays.pop(0)
            logger.warning("[Telegram] %s: fetch failed on attempt %d/%d (%s), retrying in %ss",
                           what, attempt, len(FETCH_RETRY_DELAYS) + 1, _redact_telegram_error_text(exc), delay)
            await asyncio.sleep(delay)


async def cache_replied_media(adapter: Any, msg: Any, event: Any) -> None:
    """Attach the media of the message this turn replies to. An attachment that cannot be cached
    leaves a note (as the observed-media path does), so the agent never answers from the caption
    alone. A person's replied voice note keeps the reply's own type: retyped AUDIO it skipped STT."""
    reply_msg = getattr(msg, "reply_to_message", None)
    if reply_msg is None:
        return
    status, cached = await adapter._download_observed_media(reply_msg, "replied-to media")
    if status == "ok":
        spoken = getattr(reply_msg, "voice", None) is not None and not getattr(
            getattr(reply_msg, "from_user", None), "is_bot", False)
        note = f"[Replied-to {cached.kind} '{cached.display_name}' saved at: {cached.path}]"
        if spoken:
            # The typed reply, not the replied-to speech or this note, answers a pending clarify.
            event.metadata["replied_voice_typed_text"] = (event.text or "").strip()
            note += " [Any transcript quoted above is that replied-to voice note, not this message.]"
        adapter._attach_cached(event, cached, note, "[Telegram] Cached replied-to %s at %s", retype=not spoken)
        return
    if status == "oversized":
        limit_mb = getattr(adapter, "_max_doc_bytes", 20 * 1024 * 1024) // (1024 * 1024)
        note = f"[Replied-to Telegram attachment too large or unverifiable, not cached. Maximum: {limit_mb} MB.]"
        logger.info("[Telegram] Replied-to attachment skipped (size=%s)", cached)
    elif status == "failed":
        note = "[Replied-to Telegram attachment could not be downloaded, not cached.]"
    elif status == "unreadable":
        note = "[Replied-to Telegram attachment could not be read, not cached.]"
        logger.info("[Telegram] Replied-to attachment unreadable, not cached")
    else:
        return
    event.text = adapter._append_observed_note(event.text, note)
