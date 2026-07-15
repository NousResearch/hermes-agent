"""Bounded asynchronous delivery for observer-only plugin hooks.

Observer hooks deliberately trade guaranteed delivery for isolation from the
live agent path. Each callback owns a bounded daemon queue, so a slow or hung
listener cannot delay a result or head-of-line block another listener.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(__name__)

ASYNC_OBSERVER_HOOKS = frozenset({"post_agent_result"})
OBSERVER_QUEUE_MAX = 256
OBSERVER_RELOAD_RETIRE_SECONDS = 0.1


@dataclass(eq=False)
class _ObserverCallbackWorker:
    """One generation-bound queue and daemon worker for one callback."""

    hook_name: str
    callback: Callable[..., Any]
    generation: int
    listener_id: int
    on_drop: Callable[[], None] = field(repr=False)
    on_failure: Callable[[], None] = field(repr=False)
    queue_max: int = OBSERVER_QUEUE_MAX
    _queue: queue.Queue[tuple[int, dict[str, Any]]] = field(init=False, repr=False)
    _retired: threading.Event = field(init=False, repr=False)
    _work_available: threading.Event = field(init=False, repr=False)
    _callback_gate: threading.Lock = field(init=False, repr=False)
    _callbacks_in_flight: int = field(init=False, default=0, repr=False)
    _thread: Optional[threading.Thread] = field(init=False, default=None, repr=False)
    _drop_observed: bool = field(init=False, default=False, repr=False)
    _failure_observed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=self.queue_max)
        self._retired = threading.Event()
        self._work_available = threading.Event()
        self._callback_gate = threading.Lock()

    def start(self) -> bool:
        thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"hermes-observer-{self.generation}-{self.listener_id}",
        )
        try:
            thread.start()
        except RuntimeError:
            logger.warning(
                "Could not start plugin observer worker; callback disabled",
                exc_info=True,
            )
            return False
        self._thread = thread
        return True

    def enqueue(self, event: dict[str, Any], generation: int) -> bool:
        if generation != self.generation or self._retired.is_set():
            return False
        try:
            # The generation rides on each entry. If retirement races this
            # lock-free put, the worker rechecks it before invoking user code.
            self._queue.put_nowait((generation, dict(event)))
            self._work_available.set()
        except queue.Full:
            self._mark_drop()
            return False
        if self._retired.is_set():
            # Purge copies that raced behind retirement so a detached
            # generation retains no queued event content.
            self._mark_drop()
            self._purge_pending()
            self._work_available.set()
            return False
        return True

    def _run(self) -> None:
        while not self._retired.is_set():
            self._work_available.wait()
            if self._retired.is_set():
                break
            try:
                generation, event = self._queue.get_nowait()
            except queue.Empty:
                # Clear only after observing an empty queue, then recheck it.
                # An enqueue racing the clear either appears in this recheck or
                # sets the event after it, so no wakeup can be lost.
                self._work_available.clear()
                if not self._queue.empty():
                    self._work_available.set()
                continue
            callback_claimed = False
            try:
                callback_claimed = self._claim_callback(generation)
                if not callback_claimed:
                    self._mark_drop()
                    continue
                try:
                    self.callback(event=dict(event))
                except BaseException as exc:
                    self._mark_failure()
                    logger.warning(
                        "Observer hook '%s' listener %d raised %s",
                        self.hook_name,
                        self.listener_id,
                        type(exc).__name__,
                    )
            finally:
                if callback_claimed:
                    self._release_callback()
                self._queue.task_done()

    def _claim_callback(self, generation: int) -> bool:
        """Atomically distinguish queued work from an in-flight callback."""
        with self._callback_gate:
            if generation != self.generation or self._retired.is_set():
                return False
            self._callbacks_in_flight += 1
            return True

    def _release_callback(self) -> None:
        with self._callback_gate:
            self._callbacks_in_flight -= 1

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        # Queue.join() has no timeout. Use its condition so enqueue/task_done
        # transitions cannot race an auxiliary idle flag.
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
            return True

    def _mark_drop(self) -> None:
        self._drop_observed = True
        self.on_drop()

    def _mark_failure(self) -> None:
        self._failure_observed = True
        self.on_failure()

    def _purge_pending(self) -> bool:
        purged = False
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                purged = True
                self._queue.task_done()
        return purged

    def retire(self, *, purge: bool) -> None:
        """Stop accepting work and optionally discard every queued entry."""
        # The claim gate is held only around a counter transition, never while
        # user code runs. Retirement therefore cannot wait on a hung callback.
        with self._callback_gate:
            self._retired.set()
        if purge and self._purge_pending():
            self._mark_drop()
        self._work_available.set()

    def join(self, timeout: float) -> bool:
        thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout=max(0.0, timeout))
        return not thread.is_alive()

    @property
    def active(self) -> bool:
        thread = self._thread
        return not self._retired.is_set() and thread is not None and thread.is_alive()

    def health(self) -> dict[str, Any]:
        """Return non-content diagnostics for coverage proof."""
        queued = self._queue.qsize()
        unfinished = self._queue.unfinished_tasks
        return {
            "listener_id": self.listener_id,
            "generation": self.generation,
            "active": self.active,
            "queue_max": self.queue_max,
            "queue_depth": queued,
            "in_flight": max(0, unfinished - queued),
            "callbacks_in_flight": self._callbacks_in_flight,
            "drop_observed": self._drop_observed,
            "failure_observed": self._failure_observed,
        }


class ObserverHookRuntime:
    """Generation-bound callback queues for asynchronous observer hooks."""

    def __init__(self, *, queue_max: int = OBSERVER_QUEUE_MAX) -> None:
        self.queue_max = queue_max
        self.generation = 0
        self.workers: dict[str, tuple[_ObserverCallbackWorker, ...]] = {}
        self._lifecycle_lock = threading.Lock()
        self._listener_serial = 0
        self._retired_degraded = False
        self._registration_failure = False
        self._drain_timeout = False
        self._drop_observed = False
        self._callback_failure = False

    def register(
        self, hook_name: str, callback: Callable[..., Any]
    ) -> Optional[_ObserverCallbackWorker]:
        """Prestart one isolated worker during plugin registration.

        Returns the started worker (truthy) so registrars can later retire
        exactly this registration — e.g. rolling back a failed plugin load —
        or ``None`` when the worker could not start.
        """
        self._validate_hook(hook_name)
        with self._lifecycle_lock:
            generation = self.generation
            self._listener_serial += 1
            worker = _ObserverCallbackWorker(
                hook_name=hook_name,
                callback=callback,
                generation=generation,
                listener_id=self._listener_serial,
                on_drop=self._mark_drop,
                on_failure=self._mark_failure,
                queue_max=self.queue_max,
            )
            if not worker.start():
                self._registration_failure = True
                return None
            current = self.workers.get(hook_name, ())
            updated = dict(self.workers)
            updated[hook_name] = (*current, worker)
            self.workers = updated
            return worker

    def _mark_drop(self) -> None:
        # Monotonic boolean writes avoid taking the lifecycle lock on an
        # overload or callback path.
        self._drop_observed = True

    def _mark_failure(self) -> None:
        self._callback_failure = True

    def _detach(
        self, *, retire_immediately: bool
    ) -> tuple[tuple[_ObserverCallbackWorker, ...], list[dict[str, Any]]]:
        with self._lifecycle_lock:
            workers = tuple(
                worker
                for hook_workers in self.workers.values()
                for worker in hook_workers
            )
            self.generation += 1
            self.workers = {}
            if retire_immediately:
                # Close each callback-claim gate before the lifecycle lock is
                # released; queued or dequeued-yet-unclaimed work cannot start.
                for worker in workers:
                    worker.retire(purge=True)
            return workers, [worker.health() for worker in workers]

    @staticmethod
    def _wait_idle(
        workers: tuple[_ObserverCallbackWorker, ...], deadline: float
    ) -> bool:
        for worker in workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not worker.wait_idle(remaining):
                return False
        return True

    def retire_workers(
        self,
        workers: "tuple[_ObserverCallbackWorker, ...] | list[_ObserverCallbackWorker]",
        timeout: float = OBSERVER_RELOAD_RETIRE_SECONDS,
    ) -> bool:
        """Retire specific workers without touching the rest of the runtime.

        Used to roll back exactly one registrar's listeners (a failed plugin
        load) while every other listener keeps serving: no generation bump, no
        shared drain. The given workers are detached from the active mapping,
        their claim gates close before the lifecycle lock is released (queued
        or dequeued-yet-unclaimed work cannot start), queued copies are purged,
        and each daemon is joined boundedly. Returns ``True`` when every
        retired worker stopped inside the deadline; a straggler is abandoned
        as a daemon and surfaced via ``health()``.
        """
        to_retire = tuple(worker for worker in workers if worker is not None)
        if not to_retire:
            return True
        retire_ids = {id(worker) for worker in to_retire}
        with self._lifecycle_lock:
            updated: dict[str, tuple[_ObserverCallbackWorker, ...]] = {}
            for hook_name, hook_workers in self.workers.items():
                kept = tuple(
                    worker for worker in hook_workers if id(worker) not in retire_ids
                )
                if kept:
                    updated[hook_name] = kept
            self.workers = updated
            for worker in to_retire:
                worker.retire(purge=True)
        deadline = time.monotonic() + max(0.0, timeout)
        stopped = True
        for worker in to_retire:
            if not worker.join(deadline - time.monotonic()):
                stopped = False
        if not stopped:
            self._retired_degraded = True
        return stopped

    def drain(self, timeout: float = 0.25) -> bool:
        """Wait boundedly for currently accepted events without retiring."""
        workers = tuple(
            worker for hook_workers in self.workers.values() for worker in hook_workers
        )
        drained = self._wait_idle(workers, time.monotonic() + max(0.0, timeout))
        if not drained:
            self._drain_timeout = True
        return drained

    def shutdown(self, timeout: float = 0.25, *, drain: bool = True) -> bool:
        """Detach a generation, then drain-or-purge and retire it boundedly."""
        deadline = time.monotonic() + max(0.0, timeout)
        workers, health_snapshot = self._detach(retire_immediately=not drain)
        if any(
            item["drop_observed"] or item["failure_observed"]
            for item in health_snapshot
        ):
            self._retired_degraded = True
        drained = True
        if drain:
            drained = self._wait_idle(workers, deadline)
            if not drained:
                self._drain_timeout = True
                self._retired_degraded = True
        elif any(item["queue_depth"] or item["in_flight"] for item in health_snapshot):
            self._retired_degraded = True
        if drain:
            for worker in workers:
                worker.retire(purge=True)
        stopped = True
        for worker in workers:
            remaining = deadline - time.monotonic()
            if not worker.join(remaining):
                stopped = False
        if any(
            item["drop_observed"] or item["failure_observed"]
            for item in (worker.health() for worker in workers)
        ):
            self._retired_degraded = True
        if not stopped:
            self._retired_degraded = True
        return drained and stopped

    def emit(self, hook_name: str, event: dict[str, Any]) -> bool:
        """Enqueue without waiting; return whether any listener accepted."""
        self._validate_hook(hook_name)
        generation = self.generation
        accepted = False
        for worker in self.workers.get(hook_name, ()):
            if worker.enqueue(event, generation):
                accepted = True
        return accepted

    def has_active(self, hook_name: str) -> bool:
        """Return whether a prestarted worker can currently accept events."""
        generation = self.generation
        return any(
            worker.generation == generation and worker.active
            for worker in self.workers.get(hook_name, ())
        )

    def health(self) -> dict[str, Any]:
        """Expose content-free queue and lifecycle health."""
        listeners = {
            hook_name: [worker.health() for worker in workers]
            for hook_name, workers in self.workers.items()
        }
        degraded = (
            self._registration_failure
            or self._retired_degraded
            or self._drain_timeout
            or self._drop_observed
            or self._callback_failure
        )
        degraded = degraded or any(
            item["drop_observed"] or item["failure_observed"]
            for items in listeners.values()
            for item in items
        )
        return {
            "generation": self.generation,
            "degraded": degraded,
            "registration_failure": self._registration_failure,
            "drain_timeout_observed": self._drain_timeout,
            "drop_observed": self._drop_observed,
            "callback_failure_observed": self._callback_failure,
            "retired_generation_degraded": self._retired_degraded,
            "listeners": listeners,
        }

    @staticmethod
    def _validate_hook(hook_name: str) -> None:
        if hook_name not in ASYNC_OBSERVER_HOOKS:
            raise ValueError(f"Hook '{hook_name}' is not an async observer hook")
