"""Canonical Matrix inbound event construction and server timestamp conversion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Optional

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType, QuotedMediaDependency
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache, MatrixReplyContext


def _matrix_event_timestamp_seconds(event: Any) -> float:
    """Return a Matrix event timestamp in seconds, accepting ms or sec values."""
    try:
        ts = float(getattr(event, "timestamp", None) or getattr(event, "server_timestamp", None) or 0)
    except (TypeError, ValueError):
        return 0.0
    # origin_server_ts is ms; some SDK objects/fakes expose seconds — keep both sane.
    return ts / 1000.0 if ts > 10_000_000_000 else ts


class MatrixInboundEventMixin(BasePlatformAdapter):
    _event_context_cache: MatrixEventContextCache
    _resolve_message_context: Callable[..., Awaitable[tuple | None]]
    _extract_reply_context: Callable[..., Awaitable[MatrixReplyContext]]
    _retain_rich_content: Callable[[MessageEvent, dict, str, str], None]

    async def _build_inbound_event(
        self, room_id: str, sender: str, event_id: str, body: str, source_content: dict, relates_to: dict,
        ctx: Optional[tuple] = None, *, reply_parent: MatrixEventContext | None = None,
        event_ts: float = 0.0, **extra) -> Optional[MessageEvent]:
        """Gate + normalise an inbound event into a MessageEvent (None => drop). Text body may
        still change (reply-fallback strip); ``extra`` carries media fields / message_type.
        ``ctx`` is a pre-resolved ``_resolve_message_context`` result (media path gates before
        downloading); resolving it twice would double the read receipt / thread mark."""
        from .adapter import _normalize_matrix_bang_command, _inbound_media_caption, _label_body

        reply_target = MatrixRelation.from_content(relates_to).reply_target
        retained_parent = reply_parent or (self._event_context_cache.retain(room_id, reply_target) if reply_target else None)
        if ctx is None:
            ctx = await self._resolve_message_context(room_id, sender, event_id, body, source_content, relates_to)
        if ctx is None:
            return None
        body, _is_dm, chat_type, _thread_id, display_name, requires_mention, source = ctx
        if requires_mention:
            extra["metadata"] = {**(extra.get("metadata") or {}), "matrix_requires_mention": True}
        reply = await self._extract_reply_context(
            room_id, body, source_content, relates_to, sender=sender, chat_type=chat_type,
        )
        body = reply.body
        if reply.media_path and reply.media_content_id:
            extra["media_urls"] = [*(extra.get("media_urls") or []), reply.media_path]
            extra["media_types"] = [*(extra.get("media_types") or []), reply.media_type or "image/png"]
        media_msgtype = extra.pop("media_msgtype", None)
        if source_content.get("msgtype") == "m.emote":
            body = _label_body("m.emote", body, sender)
            extra["message_type"] = MessageType.TEXT
        elif media_msgtype == "m.sticker":
            body = _label_body("m.sticker", body, sender)
        elif media_msgtype is None:
            # Re-normalize after reply stripping so ``> quoted\n\n!model`` is still a command.
            body = _normalize_matrix_bang_command(body)
            extra["message_type"] = MessageType.COMMAND if body.startswith("/") else MessageType.TEXT
        else:
            body = _inbound_media_caption(media_msgtype, body, source_content, relates_to)
        timestamp = datetime.fromtimestamp(event_ts, tz=timezone.utc) if event_ts else datetime.now(timezone.utc)
        event = MessageEvent(
            text=body, source=source, raw_message=source_content, message_id=event_id,
            reply_to_message_id=reply.event_id, reply_to_text=reply.text, reply_to_author_id=reply.author_id,
            reply_to_author_name=reply.author_name,
            reply_to_is_own_message=reply.is_own_message,
            reply_to_author_authorized=reply.author_authorized,
            # Top-level sender fields mirror source.* — downstream prompt code reads them.
            user_id=sender, user_name=display_name, timestamp=timestamp, **extra)
        if reply.media_path and reply.event_id and reply.media_content_id:
            event._quoted_media_dependencies = (
                QuotedMediaDependency(
                    room_id, reply.event_id, len(event.media_urls) - 1,
                    reply.media_content_id,
                ),
            )
        if reply.event_id:
            from plugins.platforms.matrix.turn_context import MatrixTurnContext

            event._inbound_context_dependencies = (
                MatrixTurnContext.capture(self, event, reply.parent or retained_parent),
            )
        if source_content.get("msgtype") in {"m.emote", "m.sticker"}:
            event.media_urls = event.media_urls or []
            event.media_types = event.media_types or []
            self._retain_rich_content(event, source_content, event_id, sender)
        return event
