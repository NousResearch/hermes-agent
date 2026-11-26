"""Transfer a pending message to its next session task."""

import asyncio
from typing import Optional

from gateway.platforms.event import MessageEvent


def spawn_drain_task(self, pending_event: MessageEvent, session_key: str,
                     delay: float = 0.0) -> None:
    """Hand the session to a fresh task for a queued follow-up — never recurse (chained
    follow-ups grew the C stack to SIGSEGV). Clearing (not deleting) the Event keeps the guard
    live for concurrent inbound; ownership moves so stale-lock detection works. With ``delay``
    the event stays in ``_pending_messages`` and the new owner task pops the slot only after
    sleeping, so a cancel/discard during the back-off needs no put-back and can't drop a
    newer message."""
    self._clear_session_guard(session_key)
    # Capture the guard this drain owns now: a /stop//new guard swapped in during the
    # back-off must survive the slot-empty exit (#48300).
    guard = self._active_sessions.get(session_key)
    task = asyncio.create_task(_drain_after(self, pending_event, session_key, delay, guard))
    owner = asyncio.current_task()
    task._goal_drain = getattr(owner, "_goal_drain", None)
    if owner is not None:
        owner._goal_drain = None
    self._track_session_task(session_key, task)


async def _drain_after(self, pending_event: MessageEvent, session_key: str, delay: float,
                       guard: Optional[asyncio.Event]) -> None:
    if delay > 0:
        await asyncio.sleep(delay)
        await self._flush_text_debounce_now(session_key)  # as every other task exit does
        pending_event = self._pending_messages.pop(session_key, None)
        if pending_event is None:  # consumed elsewhere during the back-off
            self._cleanup_finished_session_task(session_key, guard)
            return
        metadata = pending_event.metadata or {}
        if metadata.get("goal_created_at") is not None:
            asyncio.current_task()._goal_drain = (session_key, metadata.get("gateway_session_id"),
                                                  metadata["goal_created_at"])
    await self._process_message_background(pending_event, session_key)
