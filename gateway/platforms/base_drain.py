"""Adapter drain handoff after a pending notification reaches its turn."""
import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from gateway.platforms.event import MessageEvent

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.platforms.base")


async def drain_after(self: "BasePlatformAdapter", pending_event: MessageEvent, session_key: str, delay: float,
                       guard: Optional[asyncio.Event]) -> None:
    if delay > 0:
        await asyncio.sleep(delay)
        await self._flush_text_debounce_now(session_key)  # as every other task exit does
        pending_event = self._pending_messages.pop(session_key, None)
        if pending_event is None:  # consumed elsewhere during the back-off
            self._cleanup_finished_session_task(session_key, guard)
            return
    if self._drain_event_has_consumed_process_completion(pending_event):
        logger.debug("[%s] Dropping consumed process completion before drain", self.name)
        self._finish_session_task(session_key, guard or asyncio.Event())
        return
    await self._process_message_background(pending_event, session_key)
