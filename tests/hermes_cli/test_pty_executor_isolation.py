"""PTY I/O must never occupy asyncio's default executor.

Regression for the dashboard variant of #95559: keep-alive PTY drain loops (one
per open terminal, each blocking for the life of the session on Windows ConPTY)
were scheduled on the DEFAULT executor. That pool is only
``min(32, cpu_count + 4)`` threads, so a handful of dashboard terminals starved
every route that offloads blocking work with ``asyncio.to_thread`` —
``/api/sessions/{id}/messages`` and ``/api/sessions/{id}/latest-descendant``
hung forever while cheap sync routes kept answering, so the dashboard listed
sessions but could not open any of them.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

from hermes_cli.pty_executor import (
    get_pty_executor, reset_pty_executor_for_tests)
from hermes_cli.pty_session import PtySessionRegistry


class _FakeBridge:
    """Bridge whose read blocks like ConPTY's until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.reading = threading.Event()
        self.closed = False

    def read(self, _timeout: float = 0.2):
        self.reading.set()
        self.release.wait(timeout=10.0)
        return None  # EOF -> drain loop exits

    def close(self) -> None:
        self.closed = True


def _default_executor_probe(loop: asyncio.AbstractEventLoop):
    """A single-thread default executor plus a way to prove it stays free."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    return pool


def test_pty_drain_does_not_consume_default_executor():
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        pool = _default_executor_probe(loop)
        reset_pty_executor_for_tests()
        bridge = _FakeBridge()
        registry = PtySessionRegistry(
            ttl=60.0, max_sessions=4, buffer_cap=1024, read_timeout=0.05)
        try:
            session, created = await registry.attach_or_spawn("k", spawn=lambda: bridge)
            assert created
            # The blocking read is in flight...
            for _ in range(200):
                if bridge.reading.is_set():
                    break
                await asyncio.sleep(0.01)
            assert bridge.reading.is_set(), "drain loop never issued its blocking read"

            # ...and the single default-executor thread is still available, which
            # is exactly what the starved /api/sessions routes needed.
            probe = await asyncio.wait_for(asyncio.to_thread(lambda: "free"), timeout=2.0)
            assert probe == "free"
        finally:
            bridge.release.set()
            await registry.close_all()
            pool.shutdown(wait=True, cancel_futures=True)
            reset_pty_executor_for_tests()

    asyncio.run(scenario())


def test_pty_executor_is_shared_and_bounded():
    reset_pty_executor_for_tests()
    try:
        first = get_pty_executor()
        assert get_pty_executor() is first, "PTY pool must be process-wide"
        assert first._max_workers >= 16, "pool must outsize the PTY session registry"
    finally:
        reset_pty_executor_for_tests()
