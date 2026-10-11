"""Matrix room and thread context for gateway turns."""

from __future__ import annotations

from typing import Any, Callable, Collection

from gateway.inbound_context import InboundContextSnapshot
from gateway.platforms.event import MessageEvent
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import MatrixEventContextCache
from plugins.platforms.matrix.room_context import MatrixHistoryContext, fetch_room_entries
from plugins.platforms.matrix.thread_context import PreviousTurnCheck, fetch_thread_entries
from plugins.platforms.matrix.turn_context import MatrixTurnContext


class MatrixContextMixin:
    _client: Any
    _user_id: str
    _event_context_cache: MatrixEventContextCache
    _thread_backfill_limit: int
    _room_backfill_limit: int
    _content_mentions_bot: Callable[[str, dict], bool]
    _is_sender_authorized: Callable[..., bool | None]

    async def fetch_inbound_context(self, event: MessageEvent) -> InboundContextSnapshot:
        from plugins.platforms.matrix.rich_content import MatrixRichContentSnapshot

        if any(
            isinstance(dependency, MatrixRichContentSnapshot)
            and dependency.context.adapter is self
            for dependency in event._inbound_context_dependencies
        ):
            return MatrixRichContentSnapshot.capture(self, event)
        return MatrixTurnContext.capture(self, event)

    async def fetch_thread_history(
        self,
        chat_id: str,
        thread_id: str,
        *,
        before_event_id: str | None = None,
        exclude_event_ids: Collection[str] = (),
        is_previous_turn: PreviousTurnCheck | None = None,
    ) -> MatrixHistoryContext | None:
        entries = await fetch_thread_entries(
            self._client,
            self._event_context_cache,
            chat_id,
            thread_id,
            limit=self._thread_backfill_limit,
            before_event_id=before_event_id,
            exclude_event_ids=exclude_event_ids,
            is_previous_turn=is_previous_turn,
        )
        if not entries:
            return None
        return await MatrixHistoryContext.prepare(
            self, chat_id, entries, "Earlier messages in this thread"
        )

    async def fetch_room_history(
        self,
        chat_id: str,
        event_id: str,
        *,
        is_previous_turn: PreviousTurnCheck | None = None,
        exclude_event_ids: Collection[str] = (),
    ) -> MatrixHistoryContext | None:
        entries = await fetch_room_entries(
            self._client,
            self._event_context_cache,
            chat_id,
            event_id,
            limit=self._room_backfill_limit,
            is_previous_turn=is_previous_turn, exclude_event_ids=exclude_event_ids,
        )
        if not entries:
            return None
        return await MatrixHistoryContext.prepare(
            self, chat_id, entries, "Recent room messages"
        )

    async def fetch_mention_history(
        self, event: MessageEvent
    ) -> MatrixHistoryContext | None:
        """Read the messages that the mention gate dropped in this room or thread since the
        previous turn. Returns None when the room or thread does not require a mention,
        because every message there has already started a turn.

        The scan stops at the bot's own last reply or the last mention that the gate
        admitted, whichever is later. That event belongs to the previous turn in this room
        or thread, and an earlier catch-up covered the messages before it. The previous turn
        can belong to another session, for example after `/new` or when each mention starts
        its own automatic thread. The scan still stops there, so in the main timeline the
        first turn after a reset does not receive the conversation that the reset discarded.
        The first turn of a thread session uses the thread history instead. The bot's status
        notices are not replies, so the scan continues past them and leaves them out."""
        source = event.source
        content = event.raw_message
        if event.internal or source.chat_type == "dm" or not isinstance(content, dict):
            return None
        if not event.metadata.get("matrix_requires_mention") or not event.message_id:
            return None
        if not event.metadata.get(
            "matrix_mention_claimed"
        ) and not self._content_mentions_bot(
            str(content.get("body") or ""),
            content,
        ):
            return None

        room_id = source.chat_id

        def is_previous_turn(sender: str, original_content: dict) -> bool:
            if sender == self._user_id:
                return True
            return self._content_mentions_bot(
                str(original_content.get("body") or ""), original_content,
            ) and self._is_sender_authorized(sender, chat_type="group", chat_id=room_id) is not False

        relation = MatrixRelation.from_content(content.get("m.relates_to"))
        if relation.thread_root:
            return await self.fetch_thread_history(
                room_id,
                relation.thread_root,
                before_event_id=event.message_id,
                is_previous_turn=is_previous_turn, exclude_event_ids=event.merged_message_ids,
            )
        return await self.fetch_room_history(
            room_id, event.message_id, is_previous_turn=is_previous_turn, exclude_event_ids=event.merged_message_ids
        )
