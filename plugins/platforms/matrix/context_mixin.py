"""Matrix room and thread context for gateway turns."""

from __future__ import annotations

from typing import Any, Callable

from gateway.inbound_context import InboundContextSnapshot
from gateway.platforms.event import MessageEvent
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import (
    MatrixEventContext,
    MatrixEventContextCache,
)
from plugins.platforms.matrix.room_context import (
    fetch_room_entries,
    format_history_context,
)
from plugins.platforms.matrix.thread_context import fetch_thread_entries
from plugins.platforms.matrix.turn_context import MatrixTurnContext


class MatrixContextMixin:
    _client: Any
    _event_context_cache: MatrixEventContextCache
    _thread_backfill_limit: int
    _room_backfill_limit: int
    _content_mentions_bot: Callable[[str, dict], bool]

    async def fetch_inbound_context(
        self,
        event: MessageEvent,
        *,
        include_thread_history: bool,
    ) -> InboundContextSnapshot:
        return await MatrixTurnContext.prepare(
            self, event, include_thread_history=include_thread_history
        )

    async def fetch_thread_context(
        self,
        chat_id: str,
        thread_id: str,
        *,
        before_event_id: str | None = None,
    ) -> str | None:
        entries = await fetch_thread_entries(
            self._client,
            self._event_context_cache,
            chat_id,
            thread_id,
            limit=self._thread_backfill_limit,
            before_event_id=before_event_id,
        )
        return await self._format_history_context(
            chat_id, entries, "Earlier messages in this thread"
        )

    async def fetch_room_context(self, chat_id: str, event_id: str) -> str | None:
        entries = await fetch_room_entries(
            self._client,
            self._event_context_cache,
            chat_id,
            event_id,
            limit=self._room_backfill_limit,
        )
        return await self._format_history_context(
            chat_id, entries, "Recent room messages"
        )

    async def fetch_mention_context(self, event: MessageEvent) -> str | None:
        source = event.source
        content = event.raw_message
        if event.internal or source.chat_type == "dm" or not isinstance(content, dict):
            return None
        if not event.metadata.get(
            "matrix_mention_claimed"
        ) and not self._content_mentions_bot(
            str(content.get("body") or ""),
            content,
        ):
            return None
        if not event.message_id:
            return None

        relation = MatrixRelation.from_content(content.get("m.relates_to"))
        if relation.thread_root:
            return await self.fetch_thread_context(
                source.chat_id,
                relation.thread_root,
                before_event_id=event.message_id,
            )
        return await self.fetch_room_context(source.chat_id, event.message_id)

    async def _format_history_context(
        self,
        chat_id: str,
        entries: list[MatrixEventContext],
        heading: str,
    ) -> str | None:
        return await format_history_context(self, chat_id, entries, heading)
