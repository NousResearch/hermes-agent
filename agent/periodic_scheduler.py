"""One process-wide timer thread for periodic maintenance callbacks.

Replaces the per-child ``while not stop.wait(interval): body()`` daemon
threads (delegate heartbeat, durable turn-lease refresher, turn-liveness
watchdog).  With ~130 in-process subagents those added 2-3 sleeping OS
threads per child.  This module keeps ONE daemon thread that only orders due
times; every due body runs on its own short-lived daemon worker.  A blocked
callback therefore cannot delay unrelated lease/liveness timers, while
steady-state thread use stays near zero (workers exist only while a body is
actually running, never one per scheduled handle).

Semantics match the loop they replace: the first call happens ``interval``
seconds after :func:`schedule`, and each following call ``interval`` seconds
after the previous body *returned* (drift-free wrt. body duration was never
a property of the old loops either).  A body that returns ``False`` stops
itself; a body that raises is logged at debug and rescheduled — one bad
callback must never kill the shared thread.  A handle never overlaps itself:
it is re-queued only once its in-flight run has returned.  A worker-start
failure never retires the handle: it is re-queued and logged at warning.
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
_CALLBACK_THREAD_PREFIX = "hermes-periodic-callback"


class ScheduledHandle:
    """Cancel token for one scheduled periodic callback."""

    __slots__ = ("_fn", "_interval", "_cancelled", "_scheduler", "_runner")

    def __init__(self, scheduler: "PeriodicScheduler", fn: Callable[[], object], interval: float):
        self._scheduler = scheduler
        self._fn = fn
        self._interval = interval
        self._cancelled = False
        self._runner: Optional[threading.Thread] = None

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

    def schedule(self, fn: Callable[[], object], interval: float) -> ScheduledHandle:
        handle = ScheduledHandle(self, fn, float(interval))
        with self._cond:
            self._requeue(handle)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name=_THREAD_NAME, daemon=True)
                self._thread.start()
            self._cond.notify()
        return handle

    def _cancel(self, handle: ScheduledHandle, wait: Optional[float]) -> None:
        with self._cond:
            handle._cancelled = True
            self._cond.notify()
            runner = handle._runner
        if wait and runner is not None and threading.current_thread() is not runner:
            runner.join(wait)

    def _dispatch(self, handle: ScheduledHandle) -> None:
        """Start ``handle``'s body on its own worker.  Called with ``_cond`` held
        so ``cancel`` can never observe a half-set runner."""
        runner = threading.Thread(
            target=self._run_callback,
            args=(handle,),
            name=f"{_CALLBACK_THREAD_PREFIX}-{id(handle):x}",
            daemon=True,
        )
        handle._runner = runner
        try:
            runner.start()
        except Exception:
            handle._runner = None
            logger.warning(
                "failed to start periodic callback worker %r; retrying in %s s",
                handle._fn,
                handle._interval,
                exc_info=True,
            )
            if not handle._cancelled:
                self._requeue(handle)
                self._cond.notify()

    def _requeue(self, handle: ScheduledHandle) -> None:
        """Push ``handle``'s next due time (``_cond`` held)."""
        heapq.heappush(self._heap, (time.monotonic() + handle._interval, next(self._seq), handle))

    def _run_callback(self, handle: ScheduledHandle) -> None:
        stop = False
        try:
            stop = handle._fn() is False
        except Exception:
            logger.debug("periodic callback %r raised", handle._fn, exc_info=True)
        finally:
            with self._cond:
                handle._runner = None
                if stop:
                    handle._cancelled = True
                elif not handle._cancelled:
                    self._requeue(handle)
                self._cond.notify()

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
                    self._dispatch(handle)
                    break


_DEFAULT = PeriodicScheduler()


def schedule(fn: Callable[[], object], interval: float) -> ScheduledHandle:
    """Run ``fn()`` every ``interval`` seconds via the shared scheduler."""
    return _DEFAULT.schedule(fn, interval)
