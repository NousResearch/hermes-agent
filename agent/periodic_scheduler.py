"""One process-wide timer thread for periodic maintenance callbacks.

Replaces the per-child ``while not stop.wait(interval): body()`` daemon
threads (delegate heartbeat, durable turn-lease refresher, turn-liveness
watchdog).  With ~130 in-process subagents those added 2-3 sleeping OS
threads per child; this module runs every periodic body on ONE daemon
thread ordered by a heap of due times.

Semantics match the loop they replace: the first call happens ``interval``
seconds after :func:`schedule`, and each following call ``interval`` seconds
after the previous body *returned* (drift-free wrt. body duration was never
a property of the old loops either).  A body that returns ``False`` stops
itself; a body that raises is logged at debug and rescheduled — one bad
callback must never kill the shared thread.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_THREAD_NAME = "hermes-periodic-scheduler"


class ScheduledHandle:
    """Cancel token for one scheduled periodic callback."""

    __slots__ = ("_fn", "_interval", "_cancelled", "_scheduler")

    def __init__(self, scheduler: "PeriodicScheduler", fn: Callable[[], object], interval: float):
        self._scheduler = scheduler
        self._fn = fn
        self._interval = interval
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self, wait: Optional[float] = None) -> None:
        """Stop future runs.  ``wait`` (seconds) additionally blocks until an
        in-flight run of this callback finishes — the analogue of
        ``thread.join(timeout=wait)`` on the old per-child thread."""
        self._scheduler._cancel(self, wait)


class PeriodicScheduler:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._heap: list = []  # (due, seq, handle)
        self._seq = itertools.count()
        self._thread: Optional[threading.Thread] = None
        self._running: Optional[ScheduledHandle] = None

    def schedule(self, fn: Callable[[], object], interval: float) -> ScheduledHandle:
        handle = ScheduledHandle(self, fn, float(interval))
        with self._cond:
            heapq.heappush(self._heap, (time.monotonic() + handle._interval, next(self._seq), handle))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name=_THREAD_NAME, daemon=True)
                self._thread.start()
            self._cond.notify()
        return handle

    def _cancel(self, handle: ScheduledHandle, wait: Optional[float]) -> None:
        with self._cond:
            handle._cancelled = True
            self._cond.notify()
            if wait and self._running is handle and threading.current_thread() is not self._thread:
                self._cond.wait_for(lambda: self._running is not handle, timeout=wait)

    def _run(self) -> None:
        while True:
            with self._cond:
                while True:
                    if not self._heap:
                        self._cond.wait()
                        continue
                    due, _, handle = self._heap[0]
                    if handle._cancelled:
                        heapq.heappop(self._heap)
                        continue
                    delay = due - time.monotonic()
                    if delay > 0:
                        self._cond.wait(delay)
                        continue
                    heapq.heappop(self._heap)
                    self._running = handle
                    break
            stop = False
            try:
                stop = handle._fn() is False
            except Exception:
                logger.debug("periodic callback %r raised", handle._fn, exc_info=True)
            with self._cond:
                self._running = None
                if stop:
                    handle._cancelled = True
                elif not handle._cancelled:
                    heapq.heappush(
                        self._heap,
                        (time.monotonic() + handle._interval, next(self._seq), handle),
                    )
                self._cond.notify_all()


_DEFAULT = PeriodicScheduler()


def schedule(fn: Callable[[], object], interval: float) -> ScheduledHandle:
    """Run ``fn()`` every ``interval`` seconds on the shared scheduler thread."""
    return _DEFAULT.schedule(fn, interval)
