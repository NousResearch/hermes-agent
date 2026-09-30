"""SSE event queue for one ``/api/sessions/{id}/chat/stream`` run."""
import asyncio
import time
from contextlib import suppress
from typing import Any, Optional


class _SessionEventQueue:
    """Ordered SSE event queue for one /api/sessions/{id}/chat/stream run. ``payload`` stamps
    session_id/run_id/seq/ts; ``enqueue`` is executor-thread safe (hops onto the owning loop)."""

    def __init__(self, session_id: str, run_id: str):
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[Optional[tuple[str, dict[str, Any]]]] = asyncio.Queue()
        self.session_id = session_id
        self.run_id = run_id
        self.seq = 0

    def payload(self, name: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        self.seq += 1
        payload.setdefault("session_id", self.session_id)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("seq", self.seq)
        payload.setdefault("ts", time.time())
        return name, payload

    def enqueue(self, name: str, payload: dict[str, Any]) -> None:
        event = self.payload(name, payload)
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        with suppress(RuntimeError):
            if running_loop is self.loop:
                self.queue.put_nowait(event)
            else:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
