"""Flood-control handling for ``sendMessageDraft`` frames (``TelegramAdapter.send_draft``).

A short wait is slept inline and the frame retried once, so a single hiccup is invisible to the
caller. A longer one comes back as a retryable ``flood_control:<seconds>`` result, so the stream
consumer cools down instead of counting it toward its consecutive-failure fallback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.telegram.adapter")

_INLINE_FLOOD_WAIT_MAX = 5.0


def draft_flood_result(wait: float) -> SendResult:
    """Retryable flood-control result carrying the wait Telegram asked for."""
    return SendResult(success=False, error=f"flood_control:{wait:.0f}", retryable=True, retry_after=wait)


async def retry_draft_after_flood(
    name: str, kwargs: dict[str, Any], wait: float,
    send: Callable[[dict[str, Any]], Awaitable[Any]], redact: Callable[[object], str],
) -> SendResult:
    """Handle a flood-control refusal of one draft frame (``kwargs`` as sent to ``sendMessageDraft``)."""
    chat_id, draft_id = kwargs.get("chat_id"), kwargs.get("draft_id")
    if wait > _INLINE_FLOOD_WAIT_MAX:
        logger.debug("[%s] sendMessageDraft flood control %.1fs (chat=%s draft_id=%s)", name, wait, chat_id, draft_id)
        return draft_flood_result(wait)
    logger.debug("[%s] sendMessageDraft flood control %.1fs, retrying (chat=%s draft_id=%s)", name, wait, chat_id, draft_id)
    await asyncio.sleep(wait)
    try:
        ok = await send(kwargs)
    except Exception as exc:  # health: allow BLE001 -- any error ends this ephemeral frame; logged redacted (a traceback would carry the bot token)
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            logger.debug("[%s] sendMessageDraft flood control %.1fs on retry (chat=%s draft_id=%s)",
                         name, float(retry_after), chat_id, draft_id)
            return draft_flood_result(float(retry_after))
        # A non-flood failure is a normal miss, so repeated hard failures still trip the
        # consecutive-failure fallback.
        logger.debug("[%s] sendMessageDraft retry failed (chat=%s draft_id=%s): %s", name, chat_id, draft_id, redact(exc))
        return SendResult(success=False, error=redact(exc))
    if ok:
        return SendResult(success=True, message_id=None)
    logger.debug("[%s] sendMessageDraft ok=False after retry (chat=%s draft_id=%s)", name, chat_id, draft_id)
    return SendResult(success=False, error="rejected")
