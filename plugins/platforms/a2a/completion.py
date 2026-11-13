"""Reserved local recording capacity and independent best-effort push delivery."""
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
import logging
import threading
from typing import Literal, TypeVar, cast

from . import protocol

logger = logging.getLogger(__name__)
Record = TypeVar("Record", bound=Mapping[str, object])


class CompletionIO:
    def __init__(self) -> None:
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="a2a-completion")
        self._slots: threading.BoundedSemaphore = threading.BoundedSemaphore(128)
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._records: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="a2a-records")
        self._record_slots: threading.BoundedSemaphore = threading.BoundedSemaphore(512)
        self._reserved: set[str] = set()

    def reserve(self, task_id: str) -> bool:
        with self._lock:
            if self._closed:
                return False
            if task_id in self._reserved:
                return True
            if not self._record_slots.acquire(blocking=False):
                return False
            self._reserved.add(task_id)
            return True

    def release_reservation(self, attempt_id: str) -> None:
        with self._lock:
            if attempt_id in self._reserved:
                self._reserved.remove(attempt_id)
                self._record_slots.release()

    def record(self, callback: Callable[[Record], object], record: Record) -> Literal[False] | Callable[[], None]:
        task_id = cast(str, record.get("recording_id", record["task_id"]))
        with self._lock:
            if self._closed:
                logger.error("A2A: recording closed for task %s", task_id)
                return False
            if task_id not in self._reserved:
                if not self._record_slots.acquire(blocking=False):
                    logger.error("A2A: recording capacity exhausted for task %s", task_id)
                    return False
                self._reserved.add(task_id)
            self._reserved.remove(task_id)
            committed = threading.Event()
            self._records.submit(copy_context().run, self._record, callback, record, committed)
        return committed.set

    def _record(self, callback: Callable[[Record], object], record: Record, committed: threading.Event) -> None:
        try:
            committed.wait()
            callback(record)
        except Exception:
            logger.exception("A2A: local recording failed")
        finally:
            self._record_slots.release()

    @property
    def closed(self) -> bool:
        return self._closed

    def submit(self, callback: Callable[[Record], object], record: Record) -> bool:
        with self._lock:
            if self._closed or not self._slots.acquire(blocking=False):
                logger.warning("A2A: completion I/O unavailable; stored task result remains queryable")
                return False
            future = self._executor.submit(copy_context().run, callback, record)
        future.add_done_callback(lambda done: self._finished(done, record))
        return True

    def _finished(self, future: Future[object], record: Mapping[str, object]) -> None:
        self._slots.release()
        if future.cancelled() and record.get("push_url"):
            protocol.metrics.push_failed += 1
        if not future.cancelled() and future.exception() is not None:
            logger.error("A2A: completion I/O failed", exc_info=future.exception())

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._records.shutdown(wait=True)
