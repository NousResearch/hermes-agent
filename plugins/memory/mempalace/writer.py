"""Async writer queue for MemPalace persistence."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from .store import upsert_memory_item

logger = logging.getLogger(__name__)


class WriteQueue:
    """Thread-safe async write queue for non-blocking memory persistence."""

    def __init__(
        self,
        collection: Any,
        agent_id: str,
        thread_factory=threading.Thread,
        *,
        max_queue_size: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._collection = collection
        self._agent_id = agent_id
        self._max_retries = max(0, max_retries)
        self._retry_delay = max(0.0, retry_delay)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread = thread_factory(
            target=self._loop, name="mempalace-writer", daemon=True
        )
        self._running = True
        self._thread.start()

    def enqueue(self, items: list[dict[str, Any]]) -> None:
        self._put((items, 0), context="enqueue")

    def _put(self, payload: Any, *, context: str) -> None:
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            logger.warning("MemPalace write queue full; dropping %s batch", context)

    def _flush(self, items: list[dict[str, Any]], attempt: int) -> None:
        try:
            for item in items:
                upsert_memory_item(self._collection, item, self._agent_id)
            logger.debug("MemPalace flushed %d items to ChromaDB", len(items))
        except Exception as exc:
            logger.warning("MemPalace flush failed: %s", exc)
            if self._running and attempt < self._max_retries:
                time.sleep(self._retry_delay)
                self._put((items, attempt + 1), context="retry")
            else:
                logger.error("MemPalace dropped batch after %d retries", attempt)

    def _loop(self) -> None:
        while self._running:
            try:
                payload = self._q.get(timeout=2)
                if payload is None:
                    break
                items, attempt = payload
                self._flush(items, attempt)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("MemPalace writer error: %s", exc)

    def shutdown(self) -> None:
        self._running = False
        self._put(None, context="shutdown")
        self._thread.join(timeout=10)
