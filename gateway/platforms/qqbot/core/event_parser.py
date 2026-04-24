# -*- coding: utf-8 -*-
"""QQ Bot inbound event parser — produces platform-agnostic InboundEvent.

Replaces the scattered ``_extract_context()`` branches in the old
``QQMessageHandler`` with a single, testable :class:`EventParser`.

Zero hermes dependencies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .dto import (
    EventType,
    Message,
    MessageAttachment,
    MsgElement,
    parse_message,
)

logger = logging.getLogger(__name__)

_AT_MENTION_RE = re.compile(r"^@\S+\s*")


# ── InboundEvent ──────────────────────────────────────────────────────

@dataclass
class InboundEvent:
    """Platform-agnostic inbound event produced by :class:`EventParser`.

    Consumed by the hermes adapter layer to construct a ``MessageEvent``.
    Contains no hermes types.
    """

    event_type: str
    """Original QQ event type string, e.g. ``'C2C_MESSAGE_CREATE'``."""

    chat_id: str
    """Conversation identifier (user openid / group openid / channel id)."""

    user_id: str
    """Sender identifier."""

    chat_scope: str
    """Logical scope: ``'c2c'`` | ``'group'`` | ``'guild'`` | ``'dm'``."""

    content: str
    """Cleaned text content (@ mentions stripped where applicable)."""

    message_id: str
    timestamp: str
    message_type: int

    attachments: List[MessageAttachment] = field(default_factory=list)
    msg_elements: List[MsgElement] = field(default_factory=list)
    user_name: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


# ── EventParser ───────────────────────────────────────────────────────

class EventParser:
    """Parse a raw QQ Bot dispatch payload into an :class:`InboundEvent`.

    Usage::

        parser = EventParser()
        event = parser.parse(event_type, raw_dict)
        if event is None:
            return  # unknown / unsupported event type
    """

    def parse(
        self,
        event_type: str,
        raw: Dict[str, Any],
    ) -> Optional[InboundEvent]:
        """Parse a raw dispatch payload.

        :param event_type: QQ dispatch event type string.
        :param raw: Raw event dict from the WebSocket frame's ``d`` field.
        :returns: :class:`InboundEvent` or ``None`` if the event type is
            unsupported or required fields are missing.
        """
        if not isinstance(raw, dict):
            return None

        msg = parse_message(raw)
        content = msg.content.strip()

        handler = self._EVENT_HANDLERS.get(event_type)
        if handler is None:
            logger.debug("[EventParser] Unsupported event type: %s", event_type)
            return None

        return handler(self, event_type, msg, content, raw)

    # ------------------------------------------------------------------
    # Per-event-type handlers
    # ------------------------------------------------------------------

    def _parse_c2c(
        self,
        event_type: str,
        msg: Message,
        content: str,
        raw: Dict[str, Any],
    ) -> Optional[InboundEvent]:
        user_openid = msg.author.user_openid
        if not user_openid:
            return None
        return InboundEvent(
            event_type=event_type,
            chat_id=user_openid,
            user_id=user_openid,
            chat_scope="c2c",
            content=content,
            message_id=msg.id,
            timestamp=msg.timestamp,
            message_type=msg.message_type,
            attachments=msg.attachments,
            msg_elements=msg.msg_elements,
            raw=raw,
        )

    def _parse_group(
        self,
        event_type: str,
        msg: Message,
        content: str,
        raw: Dict[str, Any],
    ) -> Optional[InboundEvent]:
        if not msg.group_openid:
            return None
        return InboundEvent(
            event_type=event_type,
            chat_id=msg.group_openid,
            user_id=msg.author.member_openid,
            chat_scope="group",
            content=_strip_at_mention(content),
            message_id=msg.id,
            timestamp=msg.timestamp,
            message_type=msg.message_type,
            attachments=msg.attachments,
            msg_elements=msg.msg_elements,
            raw=raw,
        )

    def _parse_guild(
        self,
        event_type: str,
        msg: Message,
        content: str,
        raw: Dict[str, Any],
    ) -> Optional[InboundEvent]:
        if not msg.channel_id:
            return None
        nick = (msg.member.nick if msg.member else "") or msg.author.username
        return InboundEvent(
            event_type=event_type,
            chat_id=msg.channel_id,
            user_id=msg.author.id,
            chat_scope="guild",
            content=content,
            message_id=msg.id,
            timestamp=msg.timestamp,
            message_type=msg.message_type,
            attachments=msg.attachments,
            msg_elements=msg.msg_elements,
            user_name=nick or None,
            raw=raw,
        )

    def _parse_dm(
        self,
        event_type: str,
        msg: Message,
        content: str,
        raw: Dict[str, Any],
    ) -> Optional[InboundEvent]:
        if not msg.guild_id:
            return None
        return InboundEvent(
            event_type=event_type,
            chat_id=msg.guild_id,
            user_id=msg.author.id,
            chat_scope="dm",
            content=content,
            message_id=msg.id,
            timestamp=msg.timestamp,
            message_type=msg.message_type,
            attachments=msg.attachments,
            msg_elements=msg.msg_elements,
            raw=raw,
        )

    # Map event type strings to handler methods.
    _EVENT_HANDLERS = {
        EventType.C2C_MESSAGE_CREATE: _parse_c2c,
        EventType.GROUP_AT_MESSAGE_CREATE: _parse_group,
        EventType.GUILD_MESSAGE_CREATE: _parse_guild,
        EventType.GUILD_AT_MESSAGE_CREATE: _parse_guild,
        EventType.DIRECT_MESSAGE_CREATE: _parse_dm,
    }


# ── Helpers ───────────────────────────────────────────────────────────

def _strip_at_mention(content: str) -> str:
    """Strip the leading @bot mention from group message content."""
    return _AT_MENTION_RE.sub("", content.strip())
