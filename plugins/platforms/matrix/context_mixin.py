"""Matrix room and thread context for gateway turns."""

from __future__ import annotations

from typing import Any, Callable, Collection

from gateway.inbound_context import InboundContextSnapshot
from gateway.platforms.event import MessageEvent
from gateway.platforms.helpers import ThreadParticipationTracker
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import (
    MatrixEventContextCache,
    _has_reply_fallback,
    _split_reply_fallback,
)
from plugins.platforms.matrix.room_context import (
    MatrixHistoryContext,
    fetch_room_entries,
)
from plugins.platforms.matrix.thread_context import (
    PreviousTurnCheck,
    fetch_thread_entries,
)
from plugins.platforms.matrix.turn_context import MatrixTurnContext


class MatrixContextMixin:
    _client: Any
    _user_id: str
    _event_context_cache: MatrixEventContextCache
    _thread_backfill_limit: int
    _room_backfill_limit: int
    _threads: ThreadParticipationTracker
    _thread_require_mention: bool
    _content_mentions_bot: Callable[[str, dict], bool]
    _is_sender_authorized: Callable[..., bool | None]

    async def fetch_inbound_context(
        self, event: MessageEvent
    ) -> InboundContextSnapshot:
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
            is_previous_turn=is_previous_turn,
            exclude_event_ids=exclude_event_ids,
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

        The scan stops at the latest bot reply or authorised input that passed the
        mention gate, including a mention or a command. The boundary can belong to
        another session after `/new` or when each mention starts an automatic thread.
        Earlier context does not cross that boundary. A command which the thread's
        mention gate dropped remains background context. The first turn of a thread
        session uses the thread history instead. Bot status notices are excluded and
        do not end the scan."""
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
            from plugins.platforms.matrix.adapter import _normalize_matrix_bang_command

            body = str(original_content.get("body") or "")
            relation = MatrixRelation.from_content(original_content.get("m.relates_to"))
            if relation.thread_fallback_target and _has_reply_fallback(
                body, original_content
            ):
                _, body = _split_reply_fallback(body)
            body = _normalize_matrix_bang_command(body)
            in_bot_thread = bool(
                relation.thread_root and relation.thread_root in self._threads
            )
            command_admitted = body.startswith("/") and not (
                in_bot_thread and self._thread_require_mention
            )
            return (
                command_admitted or self._content_mentions_bot(body, original_content)
            ) and self._is_sender_authorized(
                sender, chat_type="group", chat_id=room_id
            ) is not False

        relation = MatrixRelation.from_content(content.get("m.relates_to"))
        if relation.thread_root:
            return await self.fetch_thread_history(
                room_id,
                relation.thread_root,
                before_event_id=event.message_id,
                is_previous_turn=is_previous_turn,
                exclude_event_ids=event.merged_message_ids,
            )
        return await self.fetch_room_history(
            room_id,
            event.message_id,
            is_previous_turn=is_previous_turn,
            exclude_event_ids=event.merged_message_ids,
        )
