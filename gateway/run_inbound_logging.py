"""Inbound gateway diagnostics shared by idle and queued turns."""

from __future__ import annotations

import logging

from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


def log_inbound_reply_context(source: SessionSource, message_text: str | None, event: MessageEvent | None = None, *, queued: bool = False) -> str:
    _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
    logger.info(
        "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r queued=%s",
        _platform_name, source.user_name or source.user_id or "unknown",
        source.chat_id or "unknown", (message_text or "")[:80].replace("\n", " "),
        getattr(event, "reply_to_message_id", None),
        (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " "), queued,
    )
    return _platform_name
