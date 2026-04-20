"""Asyncio event bus for the office.

Single fan-out hub used by both runtimes (simulated + Hermes-bridge) and
consumed by the ``/ws/office`` WebSocket endpoint and the persistent activity
log.

* Each subscriber gets its own bounded :class:`asyncio.Queue` (size 512).
* On overflow we drop the *oldest* item, not the newest — UX prefers fresh data
  to a stalled feed when the user is offline / browser is throttled.
* All event text passes through :func:`hermes_office.store.redact_secrets`
  before being handed to subscribers, defending in depth.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from .models import ActivityEvent
from .store import redact_secrets

logger = logging.getLogger(__name__)


class EventBus:
    """Simple async fan-out queue."""

    def __init__(self, *, queue_size: int = 512) -> None:
        self._subs: set[asyncio.Queue[ActivityEvent]] = set()
        self._queue_size = queue_size
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[ActivityEvent]:
        q: asyncio.Queue[ActivityEvent] = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subs.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[ActivityEvent]) -> None:
        async with self._lock:
            self._subs.discard(q)

    async def publish(self, evt: ActivityEvent) -> None:
        # Defensive redaction (the runtime should already redact, but never
        # trust upstream alone).
        redacted = evt.model_copy(update={"text": redact_secrets(evt.text)})
        for q in list(self._subs):
            if q.full():
                # Drop oldest to keep stream fresh.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(redacted)
            except asyncio.QueueFull:
                # Best-effort; subscriber will eventually drain.
                logger.debug("EventBus subscriber queue full; dropped event")

    async def stream(self) -> AsyncIterator[ActivityEvent]:
        """Convenience helper; yields events from a fresh subscription until
        the consumer breaks out of the loop."""
        q = await self.subscribe()
        try:
            while True:
                evt = await q.get()
                yield evt
        finally:
            await self.unsubscribe(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)
