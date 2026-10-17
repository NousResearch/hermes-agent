"""Background ingest worker for the MemorySync Hermes provider.

``sync_turn`` must never block the conversation ("Should be
non-blocking — queue for background processing", per the Hermes
MemoryProvider contract), so every write lands on this single daemon
thread. The queue is bounded: if the backend stalls long enough to fill
it, new jobs are dropped silently — a memoryless stretch, never a
blocked or crashing agent.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

_SENTINEL = object()


class IngestWorker:
    """Runs queued zero-argument callables on one daemon thread."""

    def __init__(
        self,
        *,
        on_failure: Optional[Callable[[BaseException], None]] = None,
        max_queue: int = 256,
    ) -> None:
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max_queue)
        self._on_failure = on_failure
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="memorysync-ingest"
        )
        self._thread.start()

    def submit(self, job: Callable[[], None]) -> bool:
        """Queue a job. Returns False (job dropped) when the queue is full."""
        if not self._started:
            return False
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def shutdown(self, *, timeout: float = 2.0) -> None:
        """Stop the thread, draining briefly. Never blocks beyond timeout."""
        if not self._started:
            return
        self._started = False
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is _SENTINEL:
                return
            try:
                job()  # type: ignore[operator]
            except BaseException as exc:  # noqa: BLE001 — worker never dies
                if self._on_failure is not None:
                    try:
                        self._on_failure(exc)
                    except BaseException:
                        pass
