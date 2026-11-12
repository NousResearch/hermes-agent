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

# Private event attributes, never ``event.metadata``: native admission refuses any metadata key
# outside its allowlist, which dropped every Telegram reply carrying them (as ITEM_NOTES_ATTR).
# TYPED_ATTR: a replying chunk's text before any replied-to note. VOICE_TYPED_ATTR: set when the
# turn types a reply to a person's voice note; that typed text, not the speech, answers a clarify.
TYPED_ATTR = "_telegram_typed_text"
VOICE_TYPED_ATTR = "_telegram_replied_voice_typed_text"


def replied_voice_typed_text(event: Any) -> str | None:
    """The typed clarify answer the adapter recorded for *event*, or ``None``."""
    return getattr(event, VOICE_TYPED_ATTR, None)


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
    # The typed text before any replied-to note: a chunk merged beside a voice-note reply keeps
    # only this, so another reply's attachment note never joins the clarify answer.
    event.__dict__.setdefault(TYPED_ATTR, (event.text or "").strip())
    status, cached = await adapter._download_observed_media(reply_msg, "replied-to media")
    if status == "ok":
        spoken = getattr(reply_msg, "voice", None) is not None and not getattr(
            getattr(reply_msg, "from_user", None), "is_bot", False)
        note = f"[Replied-to {cached.kind} '{cached.display_name}' saved at: {cached.path}]"
        if spoken:
            # The typed reply, not the replied-to speech or this note, answers a pending clarify.
            setattr(event, VOICE_TYPED_ATTR, getattr(event, TYPED_ATTR))
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


def _typed_text(event: Any) -> str:
    for attr in (VOICE_TYPED_ATTR, TYPED_ATTR):  # "" is a recorded value: a reply with no typed text
        if (typed := getattr(event, attr, None)) is not None:
            return typed
    return (event.text or "").strip()


def merged_typed_text(existing: Any, event: Any) -> tuple[str, bool] | None:
    """For text chunk *event* about to merge into the pending batch *existing* (read BEFORE the merge
    appends its text): the batch's typed text afterwards, every chunk's in order without replied-to
    notes, and whether any chunk typed a reply to a voice note. ``None``: nothing to record."""
    if existing is None or not (event.text or "").strip():
        return None
    voice = replied_voice_typed_text(existing) is not None or replied_voice_typed_text(event) is not None
    return "\n".join(p for p in (_typed_text(existing), _typed_text(event)) if p), voice


def record_typed_text(existing: Any, merged: tuple[str, bool]) -> None:
    typed, voice = merged
    existing.__dict__[TYPED_ATTR] = typed
    if voice:
        setattr(existing, VOICE_TYPED_ATTR, typed)
