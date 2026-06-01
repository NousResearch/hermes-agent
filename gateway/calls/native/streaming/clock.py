from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import Awaitable
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...
    def sleep(self, ms: int) -> Awaitable[None]: ...   # covers async def + Future-returning impls


class MonotonicClock:
    """Production clock backed by the event loop."""

    def now_ms(self) -> int:
        return int(time.monotonic() * 1000)

    async def sleep(self, ms: int) -> None:
        await asyncio.sleep(max(0, ms) / 1000)


class VirtualClock:
    """Deterministic test clock. Time only moves when advance() is called."""

    def __init__(self) -> None:
        self._now = 0
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._counter = 0

    def now_ms(self) -> int:
        return self._now

    def sleep(self, ms: int) -> "asyncio.Future[None]":
        # sleep() is only ever called from within a running coroutine, so
        # get_running_loop() is correct and avoids the get_event_loop()
        # deprecation / no-current-loop error on Python 3.12+.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        deadline = self._now + max(0, ms)
        self._counter += 1
        heapq.heappush(self._waiters, (deadline, self._counter, fut))
        return fut

    async def advance(self, ms: int) -> None:
        target = self._now + max(0, ms)
        while self._waiters and self._waiters[0][0] <= target:
            deadline, _, fut = heapq.heappop(self._waiters)
            self._now = deadline
            if not fut.done():
                fut.set_result(None)
            await asyncio.sleep(0)
        self._now = target
