"""Threading-based concurrency semaphore with priority waiter queue."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Iterator, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class _Waiter:
    priority: bool
    event: threading.Event = field(default_factory=threading.Event)


class ConcurrencySemaphore:
    """Semaphore that gates concurrent access with priority support.

    Priority waiters are served before non-priority waiters, allowing
    main-agent calls to jump ahead of auxiliary calls in the queue.
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        self._max_concurrent = max_concurrent
        self._active = 0
        self._waiters: list[_Waiter] = []
        self._lock = threading.Lock()

    @property
    def max_concurrent(self) -> int:
        with self._lock:
            return self._max_concurrent

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def waiting(self) -> int:
        with self._lock:
            return len(self._waiters)

    def acquire(self, *, priority: bool = False, timeout: float | None = None) -> bool:
        with self._lock:
            if self._active < self._max_concurrent and not self._waiters:
                self._active += 1
                return True

            if timeout == 0:
                return False

            waiter = _Waiter(priority=priority)
            # Insert priority waiters after existing priority waiters
            # but before non-priority waiters.
            if priority:
                insert_idx = 0
                for i, w in enumerate(self._waiters):
                    if w.priority:
                        insert_idx = i + 1
                    else:
                        break
                else:
                    # All existing waiters are priority (or list is empty)
                    insert_idx = len(self._waiters)
                self._waiters.insert(insert_idx, waiter)
            else:
                self._waiters.append(waiter)

        # Wait outside the lock
        signaled = waiter.event.wait(timeout)

        if not signaled:
            with self._lock:
                # Race: waiter may have been signaled between timeout and
                # lock acquisition. If so, the slot was already granted.
                if waiter.event.is_set():
                    return True
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass  # already removed
                return False

        return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            if self._waiters and self._active < self._max_concurrent:
                waiter = self._waiters.pop(0)
                self._active += 1
                waiter.event.set()

    @contextmanager
    def slot(
        self, *, priority: bool = False, timeout: float | None = None
    ) -> Iterator[bool]:
        acquired = self.acquire(priority=priority, timeout=timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()

    @asynccontextmanager
    async def async_slot(
        self, *, priority: bool = False, timeout: float | None = None
    ) -> AsyncIterator[bool]:
        acquired = await asyncio.to_thread(
            self.acquire, priority=priority, timeout=timeout
        )
        try:
            yield acquired
        finally:
            if acquired:
                self.release()


# ── Module-level registry ────────────────────────────────────────────

_registry: dict[tuple[str, str], ConcurrencySemaphore] = {}
_registry_lock = threading.Lock()


def get_semaphore(
    provider: str,
    api_key: str,
    *,
    max_concurrent: int | None = None,
    model: str | None = None,
) -> ConcurrencySemaphore:
    """Get or create a semaphore for a (provider, api_key) pair.

    On first call for a given pair, ``max_concurrent`` sets the limit.
    If omitted, the default is looked up from
    :func:`agent.model_metadata.get_default_concurrency`.  Subsequent
    calls return the same instance (``max_concurrent`` is ignored).
    """
    key = (provider or "", api_key or "")
    with _registry_lock:
        if key in _registry:
            return _registry[key]
        if max_concurrent is None:
            from agent.model_metadata import get_default_concurrency
            max_concurrent = get_default_concurrency(provider, model)
        sem = ConcurrencySemaphore(max_concurrent)
        logger.debug(
            "concurrency: created semaphore for (%s, %s…) max=%d",
            provider, (api_key or "")[:8], max_concurrent,
        )
        _registry[key] = sem
        return sem


def reset_registry() -> None:
    """Clear all semaphores.  For tests only."""
    with _registry_lock:
        _registry.clear()
