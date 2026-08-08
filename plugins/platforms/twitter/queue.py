from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class OperationNotStartedError(asyncio.CancelledError):
    """Cancellation while queued, before the operation callable starts."""


class RateQueue:
    def __init__(self, *, max_pending: int = 100, max_wait_seconds: float = 900):
        self.max_pending = max_pending
        self.max_wait_seconds = max_wait_seconds
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._pending: dict[str, int] = defaultdict(int)

    async def run(self, bucket: str, operation: Callable[[], Awaitable[T]]) -> T:
        if self._pending[bucket] >= self.max_pending:
            raise RuntimeError(f"Twitter {bucket} queue is full")
        self._pending[bucket] += 1
        lock = self._locks[bucket]
        acquired = False
        try:
            try:
                async with asyncio.timeout(self.max_wait_seconds):
                    await lock.acquire()
            except asyncio.CancelledError as exc:
                raise OperationNotStartedError() from exc
            acquired = True
            return await operation()
        finally:
            if acquired:
                lock.release()
            self._pending[bucket] -= 1
