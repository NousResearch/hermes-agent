"""Reserved local recording capacity and independent best-effort push delivery."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import logging
import threading

from . import protocol

logger = logging.getLogger(__name__)


class CompletionIO:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="a2a-completion")
        self._slots = threading.BoundedSemaphore(128)
        self._lock = threading.Lock()
        self._closed = False
        self._records = ThreadPoolExecutor(max_workers=1, thread_name_prefix="a2a-records")
        self._record_slots = threading.BoundedSemaphore(512)
        self._reserved = set()

    def reserve(self, task_id):
        with self._lock:
            if self._closed:
                return False
            if task_id in self._reserved:
                return True
            if not self._record_slots.acquire(blocking=False):
                return False
            self._reserved.add(task_id)
            return True

    def record(self, callback, record):
        task_id = record["task_id"]
        with self._lock:
            if self._closed:
                logger.error("A2A: recording closed for task %s", task_id)
                return
            if task_id not in self._reserved:
                if not self._record_slots.acquire(blocking=False):
                    logger.error("A2A: recording capacity exhausted for task %s", task_id)
                    return
                self._reserved.add(task_id)
            future = self._records.submit(copy_context().run, callback, record)
        future.add_done_callback(lambda done: self._record_finished(done, record))

    def _record_finished(self, future, record):
        if record["state"] not in protocol.INTERRUPTED_STATES:
            with self._lock:
                self._reserved.discard(record["task_id"])
            self._record_slots.release()
        if future.exception() is not None:
            logger.error("A2A: local recording failed", exc_info=future.exception())

    @property
    def closed(self):
        return self._closed

    def submit(self, callback, record):
        with self._lock:
            if self._closed or not self._slots.acquire(blocking=False):
                logger.warning("A2A: completion I/O unavailable; stored task result remains queryable")
                return False
            future = self._executor.submit(copy_context().run, callback, record)
        future.add_done_callback(lambda done: self._finished(done, record))
        return True

    def _finished(self, future, record):
        self._slots.release()
        if future.cancelled() and record.get("push_url"):
            protocol.metrics.push_failed += 1
        if not future.cancelled() and future.exception() is not None:
            logger.error("A2A: completion I/O failed", exc_info=future.exception())

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._records.shutdown(wait=True)
