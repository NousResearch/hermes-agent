"""One process-wide timer thread + bounded worker pool for periodic maintenance callbacks.

Replaces the per-child ``while not stop.wait(interval): body()`` daemon
threads (delegate heartbeat, durable turn-lease refresher, turn-liveness
watchdog).  With ~130 in-process subagents those added 2-3 sleeping OS
threads per child; this module runs timer management on ONE daemon
thread ordered by a heap of due times and dispatches executions to a
bounded pool of worker threads.

Semantics match the loop they replace: the first call happens ``interval``
seconds after :func:`schedule`, and each following call ``interval`` seconds
after the previous body *returned* (drift-free wrt. body duration was never
a property of the old loops either).  A body that returns ``False`` stops
itself; a body that raises is logged at debug and rescheduled — one bad
or blocked callback must never stall sibling callbacks or kill the shared
thread.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import queue
import threading
import time
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

_THREAD_NAME = "hermes-periodic-scheduler"
_WORKER_THREAD_NAME_PREFIX = "hermes-periodic-worker"
_DEFAULT_MAX_WORKERS = 8


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
    def __init__(self, max_workers: int = _DEFAULT_MAX_WORKERS) -> None:
        self._cond = threading.Condition()
        self._heap: list = []  # (due, seq, handle)
        self._seq = itertools.count()
        self._thread: Optional[threading.Thread] = None
        self._max_workers = max(1, max_workers)
        self._work_queue: queue.Queue[Optional[ScheduledHandle]] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._running_handles: Set[ScheduledHandle] = set()
        self._handle_threads: Dict[ScheduledHandle, threading.Thread] = {}

    @property
    def _running(self) -> Optional[ScheduledHandle]:
        """Backwards-compatibility accessor for tests checking in-flight execution."""
        with self._cond:
            return next(iter(self._running_handles), None) if self._running_handles else None

    def _ensure_worker_under_lock(self) -> None:
        if len(self._workers) < self._max_workers:
            idx = len(self._workers) + 1
            w = threading.Thread(
                target=self._worker_loop,
                name=f"{_WORKER_THREAD_NAME_PREFIX}-{idx}",
                daemon=True,
            )
            self._workers.append(w)
            w.start()

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
            self._cond.notify_all()
            if wait and handle in self._running_handles:
                current = threading.current_thread()
                if (
                    current is not self._thread
                    and current is not self._handle_threads.get(handle)
                ):
                    self._cond.wait_for(
                        lambda: handle not in self._running_handles, timeout=wait
                    )

    def _worker_loop(self) -> None:
        while True:
            handle = self._work_queue.get()
            if handle is None:
                break
            try:
                self._execute(handle)
            finally:
                self._work_queue.task_done()

    def _execute(self, handle: ScheduledHandle) -> None:
        with self._cond:
            self._handle_threads[handle] = threading.current_thread()
        stop = False
        try:
            if not handle._cancelled:
                stop = handle._fn() is False
        except Exception:
            logger.debug("periodic callback %r raised", handle._fn, exc_info=True)
        with self._cond:
            self._handle_threads.pop(handle, None)
            self._running_handles.discard(handle)
            if stop:
                handle._cancelled = True
            elif not handle._cancelled:
                heapq.heappush(
                    self._heap,
                    (time.monotonic() + handle._interval, next(self._seq), handle),
                )
            self._cond.notify_all()

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
                    self._running_handles.add(handle)
                    self._ensure_worker_under_lock()
                    self._work_queue.put(handle)
                    break


_DEFAULT = PeriodicScheduler()


def schedule(fn: Callable[[], object], interval: float) -> ScheduledHandle:
    """Run ``fn()`` every ``interval`` seconds on the shared scheduler thread."""
    return _DEFAULT.schedule(fn, interval)
