"""SSE event queue for one ``/api/sessions/{id}/chat/stream`` run."""
import asyncio
import threading
import time
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
        # Cross-thread puts handed to the loop and not yet applied. The loop does not order them
        # against the worker future's completion, so the turn's terminal frames wait on flush().
        self._inflight = 0
        self._inflight_lock = threading.Lock()

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
        if running_loop is self.loop:
            self.queue.put_nowait(event)
            return
        with self._inflight_lock:
            self._inflight += 1

        def _apply() -> None:
            try:
                self.queue.put_nowait(event)
            finally:
                self._settle_one()
        try:
            self.loop.call_soon_threadsafe(_apply)
        except RuntimeError:  # loop closed: the stream is gone, nothing left to deliver to
            self._settle_one()

    def _settle_one(self) -> None:
        with self._inflight_lock:
            self._inflight -= 1

    async def flush(self) -> None:
        """Wait until every event a worker thread enqueued has reached the queue. Call it before
        the terminal frames: a commentary/delta/approval handed over before the worker returned
        otherwise lands after ``done``/``None`` and the reader never sees it."""
        while True:
            with self._inflight_lock:
                if not self._inflight:
                    return
            await asyncio.sleep(0)
