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

    async def _call_with_rate_limit_backoff(self, op: Callable[[], Awaitable[Any]], *, label: str, retries: int = 3,
                                            base_delay: float = 1.5) -> Any:
        """Retry rate-limited Matrix writes with capped exponential backoff."""
        from .adapter import asyncio, logger
        from mautrix.errors.request import MLimitExceeded, MatrixRequestError
        for attempt in range(retries + 1):
            try:
                return await op()
            except MatrixRequestError as exc:
                if not isinstance(exc, MLimitExceeded) and getattr(exc, "http_status", None) != 429:
                    raise
                if attempt >= retries:
                    raise
                delay = min(base_delay * (2 ** attempt), 15.0)
                logger.warning("Matrix: %s rate limited (M_LIMIT_EXCEEDED); retry %d/%d in %.1fs",
                               label, attempt + 1, retries, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _send_room_message(
        self, chat_id: str, msg_content: Dict[str, Any], *, finalize: bool = True,
    ) -> str:
        """Send one m.room.message event (45s cap) and return its event ID as str."""
        from .adapter import asyncio, RoomID, EventType, MatrixEventContext

        event_id = await asyncio.wait_for(
            self._call_with_rate_limit_backoff(
                lambda: self._client.send_message_event(RoomID(chat_id), EventType.ROOM_MESSAGE, msg_content),
                label="send"),
            timeout=45)
        event_id = str(event_id)
        self._event_context_cache.store(
            chat_id, event_id, MatrixEventContext(self._user_id or "", msg_content["body"])
        )
        self._thread_fallbacks.remember_sent(chat_id, msg_content, event_id)
        self._remember_followup_delivery(chat_id, event_id, msg_content, finalize=finalize)
        return event_id

    async def _send_reaction(self, room_id: str, event_id: str, emoji: str) -> Optional[str]:
        """Send an emoji reaction; returns the reaction event_id, or None on failure."""
        from .adapter import RoomID, EventType, logger

        if not self._client:
            return None
        content = {"m.relates_to": {"rel_type": "m.annotation", "event_id": event_id, "key": emoji}}
        try:
            resp_event_id = await self._call_with_rate_limit_backoff(
                lambda: self._client.send_message_event(RoomID(room_id), EventType.REACTION, content),
                label="reaction")
            logger.debug("Matrix: sent reaction %s to %s", emoji, event_id)
            return str(resp_event_id)
        except Exception as exc:
            logger.debug("Matrix: reaction send error: %s", exc)
            return None

    async def redact_message(self, room_id: str, event_id: str, reason: str = "") -> bool:
        from .adapter import RoomID, EventID

        async def _redact_with_backoff():
            await self._call_with_rate_limit_backoff(
                lambda: self._client.redact(RoomID(room_id), EventID(event_id), reason=reason or None),
                label="redact")
        return await self._client_op(
            _redact_with_backoff,
            ("Matrix: redacted %s in %s", event_id, room_id), "Matrix: redact error: %s")
