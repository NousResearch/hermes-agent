"""Shared Matrix room sends, reactions and redactions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Dict, Optional, TYPE_CHECKING

from gateway.platforms.base import BasePlatformAdapter

if TYPE_CHECKING:
    from plugins.platforms.matrix.outbound_relations import ThreadFallbackTracker
    from plugins.platforms.matrix.reply_context import MatrixEventContextCache


class MatrixSendRetryMixin(BasePlatformAdapter):
    if TYPE_CHECKING:
        _client: Any
        _user_id: str
        _event_context_cache: MatrixEventContextCache
        _thread_fallbacks: ThreadFallbackTracker

        def _remember_followup_delivery(
            self, room_id: str, event_id: str, content: Dict[str, Any], *, finalize: bool,
        ) -> None: ...

        async def _client_op(
            self, coro_factory: Callable[[], Awaitable[Any]], ok_msg: tuple, err_msg: str,
            *, level: str = "warning",
        ) -> bool: ...

    async def _send_room_message(
        self, chat_id: str, msg_content: dict[str, Any], *, finalize: bool = True, notice: bool = False,
    ) -> str:
        """Send one m.room.message event (45s cap) and return its event ID as str."""
        from .adapter import asyncio, RoomID, EventType, MatrixEventContext

        event_id = await asyncio.wait_for(
            self._client.send_message_event(RoomID(chat_id), EventType.ROOM_MESSAGE, msg_content), timeout=45)
        event_id = str(event_id)
        self._event_context_cache.store(
            chat_id, event_id, MatrixEventContext(self._user_id or "", msg_content["body"])
        )
        self._thread_fallbacks.remember_sent(chat_id, msg_content, event_id, notice=notice)
        self._remember_followup_delivery(chat_id, event_id, msg_content, finalize=finalize)
        return event_id


    async def _send_reaction(self, room_id: str, event_id: str, emoji: str) -> Optional[str]:
        """Send an emoji reaction; returns the reaction event_id, or None on failure."""
        from .adapter import RoomID, EventType, logger

        if not self._client:
            return None
        content = {"m.relates_to": {"rel_type": "m.annotation", "event_id": event_id, "key": emoji}}
        try:
            resp_event_id = await self._client.send_message_event(RoomID(room_id), EventType.REACTION, content)
            logger.debug("Matrix: sent reaction %s to %s", emoji, event_id)
            return str(resp_event_id)
        except Exception as exc:
            logger.debug("Matrix: reaction send error: %s", exc)
            return None


    async def redact_message(self, room_id: str, event_id: str, reason: str = "") -> bool:
        from .adapter import RoomID, EventID

        return await self._client_op(
            lambda: self._client.redact(RoomID(room_id), EventID(event_id), reason=reason or None),
            ("Matrix: redacted %s in %s", event_id, room_id), "Matrix: redact error: %s")
