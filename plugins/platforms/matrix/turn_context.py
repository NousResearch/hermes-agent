"""Matrix event snapshots retained until a new model input is prepared."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from gateway.platforms.event import MessageEvent, QuotedMediaDependency, TurnContextUpdate
from plugins.platforms.matrix.reply_context import MatrixEventContext
from plugins.platforms.matrix.room_context import MatrixHistoryContext

_UNAVAILABLE = "[event content unavailable]"


@dataclass(frozen=True)
class MatrixTurnContextUpdate(TurnContextUpdate):
    """A turn-context update whose earlier messages are read again from the event cache
    each time the model input is rendered."""

    room_note: str | None = None
    history: MatrixHistoryContext | None = field(default=None, compare=False, repr=False)

    def render(self) -> str | None:
        history = self.history.render() if self.history is not None else None
        return "\n\n".join(block for block in (self.room_note, history) if block) or None


@dataclass
class MatrixQuotedAttachment:
    dependency: QuotedMediaDependency
    parent: MatrixEventContext | None
    path: str | None

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
        return self.path


@dataclass
class MatrixTurnContext:
    adapter: Any
    room_id: str
    reply: MessageEvent
    parent: MatrixEventContext | None
    attachments: tuple[MatrixQuotedAttachment, ...] = ()
    turn_context: TurnContextUpdate | None = None

    @classmethod
    def capture(
        cls,
        adapter: Any,
        event: MessageEvent,
        parent: MatrixEventContext | None = None,
    ) -> MatrixTurnContext:
        """Capture the reply context of *event*. The gateway's snapshot comes from
        ``fetch_inbound_context``, which runs in the routed profile's scope after
        ``rehome_inbound_media``, so each quoted image uses the event's own attachment entry.
        The parent's cached ``media_path`` can refer to another profile's cache or to a file
        that has since been moved. An entry that is still in the launch profile's cache is
        dropped, because the gateway could not move it into the routed profile."""
        from gateway.run_inbound import rehomed_media_path

        def turn_path(dependency: QuotedMediaDependency) -> str | None:
            path = event.media_urls[dependency.media_index]
            return path if rehomed_media_path(path) == path else None

        room_id = event.source.chat_id
        dependencies = tuple(
            dependency
            for dependency in event._inbound_context_dependencies
            if isinstance(dependency, cls) and dependency.adapter is adapter
        )
        if parent is None and event.reply_to_message_id:
            parent = next(
                (
                    dependency.parent
                    for dependency in dependencies
                    if dependency.room_id == room_id
                    and dependency.reply.reply_to_message_id
                    == event.reply_to_message_id
                ),
                None,
            ) or adapter._event_context_cache.retain(room_id, event.reply_to_message_id)
        attachments: list[MatrixQuotedAttachment] = []
        for dependency in event._quoted_media_dependencies:
            retained = next(
                (
                    attachment.parent
                    for snapshot in dependencies
                    for attachment in snapshot.attachments
                    if attachment.dependency.room_id == dependency.room_id
                    and attachment.dependency.event_id == dependency.event_id
                    and attachment.dependency.content_id == dependency.content_id
                ),
                None,
            ) or adapter._event_context_cache.retain(
                dependency.room_id, dependency.event_id
            )
            attachments.append(
                MatrixQuotedAttachment(dependency, retained, turn_path(dependency))
            )
        return cls(
            adapter, room_id, replace(event), parent, attachments=tuple(attachments)
        )

    def use_turn_context(self, update: TurnContextUpdate | None) -> None:
        self.turn_context = update

    async def refresh(self) -> None:
        update = self.turn_context
        if isinstance(update, MatrixTurnContextUpdate) and update.history is not None:
            await update.history.refresh()
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

    def prepend_turn_context(self, text: str) -> str:
        update = self.turn_context
        if isinstance(update, MatrixTurnContextUpdate):
            note = update.render()
        else:
            note = update.note if update is not None else None
        return f"{note}\n\n[New message]\n{text}" if note else text

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
        if not (parent.sender or parent.text or parent.redacted or parent.state_error):
            return replace(
                event,
                reply_to_text=self.reply.reply_to_text or _UNAVAILABLE,
                reply_to_author_id=self.reply.reply_to_author_id,
                reply_to_author_name=self.reply.reply_to_author_name,
                reply_to_is_own_message=self.reply.reply_to_is_own_message,
                reply_to_author_authorized=self.reply.reply_to_author_authorized,
            )
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
            reply_to_text="[redacted]"
            if parent.redacted
            else _UNAVAILABLE
            if parent.state_error
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
