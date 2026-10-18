"""Chunked ``chat.postMessage`` for the Slack standalone (out-of-process) senders.

The split runs on the mrkdwn-formatted text, as the live adapter's ``_post_chunks`` does:
formatting grows text (``&`` becomes ``&amp;``), so only that split bounds every post by
``SlackAdapter.MAX_MESSAGE_LENGTH``.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)


def standalone_post_kwargs(
    chat_id: str, text: Any, unfurl_kwargs: dict[str, Any], thread_id: Optional[str]
) -> dict[str, Any]:
    """``chat.postMessage`` kwargs for the standalone senders (key order is the wire order)."""
    kwargs: dict[str, Any] = {"channel": chat_id, "text": text, "mrkdwn": True, **unfurl_kwargs}
    if thread_id:
        kwargs["thread_ts"] = thread_id
    return kwargs


async def post_chunks(
    post: Callable[[str], Awaitable[dict[str, Any]]], formatted: str, limit: int,
) -> dict[str, Any]:
    """``post`` each chunk in order; returns the last response, or the first failed one. A failure
    after the first chunk carries ``delivered_chunks`` / ``total_chunks``: the head is already on
    screen, so the caller must report the loss rather than resend the whole message. A response
    without ``ok`` counts as delivered, as the SDK senders always treated it."""
    chunks = BasePlatformAdapter.truncate_message(formatted, limit)
    data: dict[str, Any] = {}
    for index, chunk in enumerate(chunks):
        try:
            data = await post(chunk)
        except Exception as exc:
            if not index:
                raise
            logger.warning("[Slack] standalone chunk %d/%d failed", index + 1, len(chunks),
                           exc_info=True)
            data = {"ok": False, "error": str(exc)}
        if not data.get("ok", True):
            return {**data, "delivered_chunks": index, "total_chunks": len(chunks)} if index else data
    return data
