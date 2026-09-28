"""Native Matrix action attribution and authored attachment lifetime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable

from gateway.platforms.event import MessageEvent
from plugins.platforms.matrix.reply_context import (
    MatrixEventContext,
    MatrixEventContextCache,
)
from plugins.platforms.matrix.turn_context import MatrixTurnContext


@dataclass(frozen=True)
class _MatrixInboundEvent:
    room_id: str
    sender: str
    event_id: str
    timestamp: Any
    type: str
    content: dict[str, Any] | None


def inbound_event(event: Any) -> Any:
    if not isinstance(event, dict):
        return event
    content = event.get("content")
    return _MatrixInboundEvent(
        room_id=str(event.get("room_id") or ""),
        sender=str(event.get("sender") or ""),
        event_id=str(event.get("event_id") or ""),
        timestamp=event.get("origin_server_ts", 0),
        type=str(event.get("type") or ""),
        content=content if isinstance(content, dict) else None,
    )


@dataclass
class MatrixRichContentSnapshot:
    context: MatrixTurnContext
    authored: MatrixEventContext
    original_text: str
    original_media_identity: str

    @classmethod
    async def prepare(
        cls, adapter: Any, event: MessageEvent, *, include_thread_history: bool
    ) -> MatrixRichContentSnapshot:
        previous = next(
            (
                snapshot
                for snapshot in event._inbound_context_dependencies
                if isinstance(snapshot, cls) and snapshot.context.adapter is adapter
            ),
            None,
        )
        context = await MatrixTurnContext.prepare(
            adapter, event, include_thread_history=include_thread_history
        )
        authored = (
            previous.authored
            if previous is not None
            else adapter._event_context_cache.retain(
                event.source.chat_id,
                event.message_id,
            )
        )
        return cls(
            context,
            authored,
            previous.original_text if previous is not None else event.text,
            previous.original_media_identity
            if previous is not None
            else authored.attachment_identity,
        )

    async def refresh(self) -> None:
        await self.context.refresh()
        self.authored = await self.context.adapter._event_context_cache.refresh(
            self.context.adapter._client,
            self.context.room_id,
            self.authored,
        )

    def _current(self) -> MatrixEventContext:
        return self.context.adapter._event_context_cache.recheck(
            self.context.room_id, self.authored
        )

    def media_event(self, event: MessageEvent) -> MessageEvent:
        authored = event.authored_media()
        current = self._current()
        if (
            current.redacted
            or current.state_error
            or current.attachment_identity != self.original_media_identity
        ):
            return replace(authored, media_urls=[], media_types=[])
        return authored

    def prepend_history(self, text: str) -> str:
        current = self._current()
        if current.redacted or current.state_error:
            updated = (
                "[redacted]" if current.redacted else "[event content unavailable]"
            )
        else:
            updated = current.text
        if self.original_text and updated != self.original_text:
            text = text.replace(self.original_text, updated, 1)
        return self.context.prepend_history(text)

    def reply_event(self, event: MessageEvent) -> MessageEvent:
        return self.context.reply_event(event)

    def reply_image_paths(self) -> list[str]:
        return self.context.reply_image_paths()


class MatrixRichContentMixin:
    _event_context_cache: MatrixEventContextCache
    _text_batch_delay_seconds: float
    _build_inbound_event: Callable[..., Awaitable[MessageEvent | None]]
    _enqueue_text_event: Callable[[MessageEvent], None]
    handle_message: Callable[[MessageEvent], Awaitable[Any]]

    async def _handle_emote_message(
        self,
        room_id: str,
        sender: str,
        event_id: str,
        event_ts: float,
        source_content: dict[str, Any],
        relates_to: dict[str, Any],
        *,
        reply_parent: MatrixEventContext | None = None,
    ) -> None:
        body = source_content.get("body")
        if not isinstance(body, str) or not body:
            return
        event = await self._build_inbound_event(
            room_id,
            sender,
            event_id,
            body,
            source_content,
            relates_to,
            reply_parent=reply_parent,
        )
        if event is not None:
            if self._text_batch_delay_seconds > 0:
                self._enqueue_text_event(event)
            else:
                await self.handle_message(event)

    def _retain_rich_content(
        self, event: MessageEvent, content: dict, event_id: str, sender: str
    ) -> None:
        room_id = event.source.chat_id
        media = event.authored_media()
        authored = MatrixEventContext(
            sender,
            event.text,
            media_path=media.media_urls[0] if media.media_urls else None,
            media_type=media.media_types[0] if media.media_types else None,
            is_image=content.get("msgtype") == "m.sticker",
            media_content=MatrixEventContext.image_content(content),
        )
        entry = self._event_context_cache.retain(room_id, event_id)
        if not entry.redacted and not entry.replacement_id and not entry.state_error:
            self._event_context_cache.store(room_id, event_id, authored)
            entry = self._event_context_cache.retain(room_id, event_id)
        context = MatrixTurnContext.capture(self, event)
        event._inbound_context_dependencies = (
            *event._inbound_context_dependencies,
            context,
            MatrixRichContentSnapshot(
                context, entry, event.text, authored.attachment_identity
            ),
        )
