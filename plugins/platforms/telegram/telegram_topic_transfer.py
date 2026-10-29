"""Opt-in root-DM -> fresh topic transfer for Telegram ``/topic`` mode (``dm_topic_auto_transfer``).

Off by default: a root-DM message keeps the v0.22 lobby recovery and is answered in the user's
most recently bound topic (#136567). With the key on, the runner's transfer hook creates a new
topic and the message (and its reply) move there before session keying; if creating the topic
fails, the hook answers with the lobby-recovery topic instead (#139136).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from gateway.platforms.base import coerce_plaintext_gateway_command
from gateway.platforms.event import MessageEvent
from gateway.session_identity import replace_source

logger = logging.getLogger(__name__)

# ``(thread_id, created)``: ``created`` = a fresh topic; otherwise the lobby-recovery fallback topic.
TopicTransferFn = Callable[[Any], Awaitable[Optional[tuple[str, bool]]]]


async def apply_root_dm_topic_transfer(adapter: Any, event: MessageEvent) -> None:
    """Rewrite ``event.source`` onto the topic the transfer hook picks; no-op when it declines.

    Runs at adapter ingress, before session keying, so the turn, its session and its reply all use
    the new topic. Commands, internal wakes, routed events and non-DM traffic are never moved.
    """
    transfer: Optional[TopicTransferFn] = getattr(adapter, "_topic_transfer_fn", None)
    source = getattr(event, "source", None)
    if transfer is None or source is None or getattr(source, "chat_type", None) != "dm":
        return
    if getattr(event, "allow_gateway_control", False):
        coerce_plaintext_gateway_command(event)  # "restart gateway" is a lobby command too (idempotent)
    if getattr(event, "internal", False) or event.get_command():
        return
    if str((event.metadata or {}).get("gateway_session_key") or "").strip():
        return
    if adapter._is_sender_authorized(source.user_id, source.chat_type, source.chat_id) is False:
        return
    try:
        decided = await transfer(source)
    except Exception:
        logger.debug("telegram topic transfer hook failed", exc_info=True)
        return
    if not decided:
        return
    thread_id, created = decided
    if not created:
        event.source = replace_source(source, thread_id=str(thread_id))
        return
    # The root-DM message lives outside the new topic, so it cannot anchor the reply there: an empty
    # override plus no source.message_id sends the first reply by message_thread_id alone.
    event.source = replace_source(source, thread_id=str(thread_id), message_id=None)
    event.reply_anchor_override = ""
