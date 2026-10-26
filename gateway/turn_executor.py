"""Unbounded executor for gateway turn bodies (see ``GatewayRunner._get_executor`` in gateway/run.py)."""

from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import queue
import threading

logger = logging.getLogger(__name__)

# A parked worker exits after this long without work, so a burst's threads do not outlive it by much.
_IDLE_TIMEOUT_S = 60.0


class _Worker:
    __slots__ = ("inbox", "thread")

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()


class _UnboundedThreadExecutor(concurrent.futures.Executor):
    """Every submitted item starts at once on its own thread; no queue, no cap.

    ``ThreadPoolExecutor(max_workers=None)`` is NOT unbounded (it is ``min(32, cpu_count + 4)``),
    which is the same silent queue at a larger number. An item goes to a parked (idle) worker when
    one exists, otherwise to a new thread. Reuse matters because ``submit`` runs on the event loop
    and ``Thread.start()`` waits there until the new thread gets the GIL: with many turns running,
    every submit (the handoff watcher's 2 s idle gates included) froze the loop (#142061). Handing an
    item to a parked worker is a queue put. A worker parked for ``idle_timeout`` exits.

    A reused thread must look fresh to its next item: each item runs in an empty
    ``contextvars.Context`` (what a new thread starts with), and ``reset_reused_thread_state`` drops
    the thread-keyed state a turn leaves behind before the worker parks. A worker whose reset fails
    is retired, never reused.

    ``_threads`` holds the BUSY threads only, as before: ``_stop_pool`` / ``_shutdown_executor`` join
    and count those. A parked worker holds no work; shutdown wakes it to exit.
    """

    def __init__(self, thread_name_prefix: str = "", idle_timeout: float = _IDLE_TIMEOUT_S):
        self._prefix = thread_name_prefix
        self._threads: set = set()
        self._idle: list[_Worker] = []
        self._shutdown = False
        self._lock = threading.Lock()
        self._n = 0
        self._idle_timeout = idle_timeout

    def submit(self, fn, /, *args, **kwargs):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        item = (fut, fn, args, kwargs)
        # One critical section for check + hand-off/start + register (as ThreadPoolExecutor.submit does), so a
        # concurrent shutdown() either refuses this item or sees its thread; never a live, uncounted one.
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if self._idle:
                worker = self._idle.pop()  # LIFO: the warmest thread; the coldest time out
                self._threads.add(worker.thread)
                worker.inbox.put(item)
                return fut
            self._n += 1
            worker = _Worker()
            # health: allow HX012 -- a worker serves many items; each runs in the context its caller captured (ctx.run)
            t = threading.Thread(target=self._work, args=(worker, item), name=f"{self._prefix}_{self._n}", daemon=True)
            worker.thread = t
            # Start BEFORE registering: at the OS thread limit start() raises, and an unstarted thread
            # left in _threads would make shutdown's join() raise and skip the quiesce decision.
            t.start()
            self._threads.add(t)
        return fut

    def _work(self, worker: _Worker, item) -> None:
        while item is not None:
            fut, fn, args, kwargs = item
            del item
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(contextvars.Context().run(fn, *args, **kwargs))
                except BaseException as exc:  # health: allow BLE001 -- delivered to the item's future, as ThreadPoolExecutor does
                    fut.set_exception(exc)
            del fut, fn, args, kwargs  # a parked worker must not pin the last item's arguments
            item = self._park(worker)

    def _park(self, worker: _Worker):
        """Reset this thread, then wait for its next item; None means exit."""
        from gateway.turn_thread_reset import reset_reused_thread_state

        try:
            reset_reused_thread_state()
            reusable = True
        except Exception:
            logger.warning("turn worker state reset failed; retiring the thread", exc_info=True)
            reusable = False
        # Taking the lock also waits for submit() to finish registering this thread.
        with self._lock:
            self._threads.discard(worker.thread)
            if self._shutdown or not reusable:
                return None
            self._idle.append(worker)
        try:
            return worker.inbox.get(timeout=self._idle_timeout)
        except queue.Empty:
            with self._lock:
                if worker in self._idle:
                    self._idle.remove(worker)
                    return None
            # submit() claimed this worker as the timeout fired: its item is already on the way.
            return worker.inbox.get()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        # cancel_futures is accepted for API parity only: there is no queue, so nothing is pending.
        with self._lock:
            self._shutdown = True
            threads = list(self._threads)
            idle, self._idle = self._idle, []
        for worker in idle:
            worker.inbox.put(None)
        if wait:
            for t in threads:
                t.join()
