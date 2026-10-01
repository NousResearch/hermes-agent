"""Native Matrix action attribution and authored attachment lifetime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from gateway.platforms.event import MessageEvent, TurnContextUpdate
from plugins.platforms.matrix.reply_context import (
    MatrixEventContext,
    MatrixEventContextCache,
    _label_body,
    _own_text,
)
from plugins.platforms.matrix.turn_context import MatrixTurnContext


def has_media_url(content: dict[str, Any]) -> bool:
    encrypted = content.get("file")
    return bool(content.get("url") or (isinstance(encrypted, dict) and encrypted.get("url")))


def native_event_context(
    content: dict[str, Any],
    sender: str,
    *,
    media_path: str | None = None,
    media_type: str | None = None,
) -> MatrixEventContext:
    msgtype = str(content.get("msgtype") or "")
    return MatrixEventContext(
        sender,
        _label_body(msgtype, _own_text(str(content.get("body") or "").strip(), content), sender),
        media_path=media_path,
        media_type=media_type,
        is_image=msgtype == "m.sticker",
        media_content=MatrixEventContext.image_content(content),
    )


@dataclass
class _MatrixAuthoredContent:
    authored: MatrixEventContext
    original_text: str
    original_content_text: str
    original_media_identity: str
    media_paths: tuple[str, ...]

    def current(self, context: MatrixTurnContext) -> MatrixEventContext:
        return context.adapter._event_context_cache.recheck(
            context.room_id, self.authored
        )

    def text(self, context: MatrixTurnContext) -> str:
        current = self.current(context)
        if current.redacted:
            return "[redacted]"
        if current.state_error:
            return "[event content unavailable]"
        return (
            self.original_text
            if current.text == self.original_content_text
            else current.text
        )

    def media_available(self, context: MatrixTurnContext) -> bool:
        current = self.current(context)
        return (
            not current.redacted
            and not current.state_error
            and current.attachment_identity == self.original_media_identity
        )


@dataclass
class MatrixRichContentSnapshot:
    context: MatrixTurnContext
    contributions: tuple[_MatrixAuthoredContent, ...]

    @classmethod
    def capture(cls, adapter: Any, event: MessageEvent) -> MatrixRichContentSnapshot:
        """Capture the event's contributions after the gateway has moved its attachments into
        the routed profile. Call this in that profile's scope: each contribution records its
        attachments at the paths where ``rehome_inbound_media`` moved them."""
        from gateway.run_inbound_media import rehomed_media_path

        def current_path(path: str) -> str:
            rehomed = rehomed_media_path(path)
            return rehomed if rehomed in event.media_urls else path

        contributions = tuple(
            replace(
                contribution,
                media_paths=tuple(map(current_path, contribution.media_paths)),
            )
            for snapshot in event._inbound_context_dependencies
            if isinstance(snapshot, cls) and snapshot.context.adapter is adapter
            for contribution in snapshot.contributions
        )
        return cls(MatrixTurnContext.capture(adapter, event), contributions)

    def pending_state(self) -> dict[str, Any]:
        return {
            "kind": "matrix_rich_content",
            "context": self.context.pending_state(),
            "contributions": [
                {
                    "event_id": contribution.authored.event_id,
                    "sender": contribution.authored.sender,
                    "original_text": contribution.original_text,
                    "original_content_text": contribution.original_content_text,
                    "original_media_identity": contribution.original_media_identity,
                    "media_paths": list(contribution.media_paths),
                }
                for contribution in self.contributions
            ],
        }

    def use_turn_context(self, update: TurnContextUpdate | None) -> None:
        self.context.use_turn_context(update)

    async def refresh(self) -> None:
        await self.context.refresh()
        for contribution in self.contributions:
            contribution.authored = (
                await self.context.adapter._event_context_cache.refresh(
                    self.context.adapter._client,
                    self.context.room_id,
                    contribution.authored,
                )
            )

    def media_event(self, event: MessageEvent) -> MessageEvent:
        authored = event.authored_media()
        withdrawn = {
            path
            for contribution in self.contributions
            if not contribution.media_available(self.context)
            for path in contribution.media_paths
        }
        indices = [
            index
            for index, path in enumerate(authored.media_urls)
            if path not in withdrawn
        ]
        return replace(
            authored,
            media_urls=[authored.media_urls[index] for index in indices],
            media_types=[
                authored.media_types[index]
                for index in indices
                if index < len(authored.media_types)
            ],
            media_text_inlined=[
                authored.media_text_inlined[index] for index in indices
            ],
        )

    def authored_text(self, text: str) -> str:
        groups: dict[str, list[_MatrixAuthoredContent]] = {}
        for contribution in self.contributions:
            groups.setdefault(contribution.original_text, []).append(contribution)
        collapsed = {
            original
            for original, contributions in groups.items()
            if original and self.context.reply.text.count(original) < len(contributions)
        }
        processed: set[str] = set()
        offset = 0
        for contribution in self.contributions:
            original = contribution.original_text
            if not original or original in processed:
                continue
            start = text.find(original, offset)
            if start < 0:
                continue
            if original in collapsed:
                updated = "\n\n".join(
                    dict.fromkeys(
                        entry.text(self.context) for entry in groups[original]
                    )
                )
                processed.add(original)
            else:
                updated = contribution.text(self.context)
            text = text[:start] + updated + text[start + len(original) :]
            offset = start + len(updated)
        return text

    def prepend_turn_context(self, text: str) -> str:
        return self.context.prepend_turn_context(text)

    def reply_event(self, event: MessageEvent) -> MessageEvent:
        return self.context.reply_event(event)

    def reply_image_paths(self) -> list[str]:
        return self.context.reply_image_paths()


class MatrixRichContentMixin:
    _event_context_cache: MatrixEventContextCache
    _text_batch_delay_seconds: float
    _build_inbound_event: Callable[..., Awaitable[MessageEvent | None]]
    _enqueue_text_event: Callable[[MessageEvent], None]
    if TYPE_CHECKING:
        async def handle_message(self, event: MessageEvent) -> None: ...

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
        authored = native_event_context(
            content,
            sender,
            media_path=media.media_urls[0] if media.media_urls else None,
            media_type=media.media_types[0] if media.media_types else None,
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
                context,
                (
                    _MatrixAuthoredContent(
                        entry,
                        event.text,
                        authored.text,
                        authored.attachment_identity,
                        tuple(media.media_urls),
                    ),
                ),
            ),
        )
