"""Matrix event snapshots retained until a new model input is prepared."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from gateway.platforms.event import MessageEvent, QuotedMediaDependency
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import MatrixEventContext
from plugins.platforms.matrix.room_context import (
    MatrixHistoryContext,
    fetch_room_entries,
)
from plugins.platforms.matrix.thread_context import fetch_thread_entries


@dataclass
class MatrixQuotedAttachment:
    dependency: QuotedMediaDependency
    parent: MatrixEventContext | None

    async def refresh(self, adapter: Any) -> None:
        cache = adapter._event_context_cache
        parent = self.parent or cache.history_entry(
            self.dependency.room_id, self.dependency.event_id
        )
        if parent is not None:
            self.parent = await cache.refresh(
                adapter._client, self.dependency.room_id, parent
            )

    def image_path(self, adapter: Any) -> str | None:
        if self.parent is None:
            return None
        parent = adapter._event_context_cache.recheck(
            self.dependency.room_id, self.parent
        )
        if (
            parent.redacted
            or parent.state_error
            or parent.attachment_identity != self.dependency.content_id
        ):
            return None
        return parent.media_path


@dataclass
class MatrixTurnContext:
    adapter: Any
    room_id: str
    reply: MessageEvent
    parent: MatrixEventContext | None
    history: MatrixHistoryContext | None = None
    mention: bool = False
    attachments: tuple[MatrixQuotedAttachment, ...] = ()

    @classmethod
    async def prepare(
        cls,
        adapter: Any,
        event: MessageEvent,
        *,
        include_thread_history: bool,
    ) -> MatrixTurnContext:
        room_id = event.source.chat_id
        parent = (
            adapter._event_context_cache.history_entry(
                room_id, event.reply_to_message_id
            )
            if event.reply_to_message_id
            else None
        )
        snapshot = cls(
            adapter,
            room_id,
            replace(event),
            parent,
            attachments=tuple(
                MatrixQuotedAttachment(
                    dependency,
                    adapter._event_context_cache.history_entry(
                        dependency.room_id, dependency.event_id
                    ),
                )
                for dependency in event._quoted_media_dependencies
            ),
        )
        content = event.raw_message
        mention = (
            not event.internal
            and bool(event.message_id)
            and event.source.chat_type != "dm"
            and isinstance(content, dict)
            and (
                event.metadata.get("matrix_mention_claimed")
                or adapter._content_mentions_bot(
                    str(content.get("body") or ""),
                    content,
                )
            )
        )
        snapshot.mention = bool(mention)
        thread_id = (
            MatrixRelation.from_content(content.get("m.relates_to")).thread_root
            if mention
            else None
        )
        if not mention and include_thread_history and not event.internal:
            thread_id = event.source.thread_id
        if thread_id and (mention or thread_id != event.message_id):
            entries = await fetch_thread_entries(
                adapter._client,
                adapter._event_context_cache,
                room_id,
                thread_id,
                limit=adapter._thread_backfill_limit,
                before_event_id=event.message_id,
            )
            heading = "Earlier messages in this thread"
        elif mention and event.message_id:
            entries = await fetch_room_entries(
                adapter._client,
                adapter._event_context_cache,
                room_id,
                event.message_id,
                limit=adapter._room_backfill_limit,
            )
            heading = "Recent room messages"
        else:
            return snapshot
        if entries:
            snapshot.history = await MatrixHistoryContext.prepare(
                adapter, room_id, entries, heading
            )
        return snapshot

    async def refresh(self) -> None:
        if self.history is not None:
            await self.history.refresh()
        for attachment in self.attachments:
            await attachment.refresh(self.adapter)
        event_id = self.reply.reply_to_message_id
        if not event_id:
            return
        current = self.adapter._event_context_cache.history_entry(
            self.room_id, event_id
        )
        parent = self.parent or current
        if parent is not None:
            self.parent = await self.adapter._event_context_cache.refresh(
                self.adapter._client, self.room_id, parent
            )
            sender = self.parent.sender
            if sender and sender != self.reply.reply_to_author_id:
                self.reply.reply_to_author_name = await self.adapter._get_display_name(
                    self.room_id, sender
                )
                self.reply.reply_to_author_id = sender

    def prepend_history(self, text: str) -> str:
        history = self.history.render() if self.history is not None else None
        if not history:
            return text
        separator = "\n\n[New message]\n" if self.mention else "\n\n"
        return f"{history}{separator}{text}"

    def _current_parent(self) -> MatrixEventContext | None:
        event_id = self.reply.reply_to_message_id
        parent = self.parent or (
            self.adapter._event_context_cache.history_entry(self.room_id, event_id)
            if event_id
            else None
        )
        return (
            self.adapter._event_context_cache.recheck(self.room_id, parent)
            if parent is not None
            else None
        )

    def reply_event(self, event: MessageEvent) -> MessageEvent:
        parent = self._current_parent()
        if parent is None:
            return replace(event, reply_to_text=self.reply.reply_to_text)
        sender = parent.sender or None
        own = sender == self.adapter._user_id
        authorized = (
            None
            if own or not sender
            else self.adapter._is_sender_authorized(
                sender,
                chat_type=event.source.chat_type,
                chat_id=self.room_id,
            )
        )
        return replace(
            event,
            reply_to_text=None
            if parent.redacted or parent.state_error
            else parent.text,
            reply_to_author_id=sender,
            reply_to_author_name=self.reply.reply_to_author_name,
            reply_to_is_own_message=own,
            reply_to_author_authorized=authorized,
        )

    def reply_image_paths(self) -> list[str]:
        return list(
            dict.fromkeys(
                path
                for attachment in self.attachments
                if (path := attachment.image_path(self.adapter)) is not None
            )
        )
