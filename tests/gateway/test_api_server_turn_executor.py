"""API-server agent turns must not occupy asyncio's default executor (#144362).

The session handlers do their SessionDB work with ``asyncio.to_thread`` (the default pool). If live
turns hold that pool, every /api/sessions read queues behind a whole turn.
"""

import asyncio
import os
import threading

import pytest

from gateway.platforms import api_server_runs as api_runs


def _default_pool_size() -> int:
    return min(32, (os.cpu_count() or 1) + 4)


@pytest.mark.asyncio
async def test_live_api_turns_leave_the_default_pool_to_session_reads():
    loop = asyncio.get_running_loop()
    release = threading.Event()
    started = threading.Semaphore(0)
    n = _default_pool_size() + 2

    def _turn():
        started.release()
        release.wait(10)

    turns = [api_runs._submit_api_worker(loop, _turn) for _ in range(n)]
    try:
        session_read = asyncio.ensure_future(asyncio.to_thread(lambda: "row"))
        assert await asyncio.wait_for(session_read, 3) == "row", "session read queued behind live turns"
        await asyncio.wait_for(asyncio.to_thread(lambda: [started.acquire(timeout=5) for _ in range(n)]), 8)
        assert api_runs.api_worker_live_count() >= n
    finally:
        release.set()
        await asyncio.gather(*turns)


@pytest.mark.asyncio
async def test_cron_fire_body_keeps_contextvars_off_the_default_pool():
    import contextvars

    var = contextvars.ContextVar("probe_scope", default="unset")
    var.set("profile-a")
    seen = {}

    def _fire(job, *, adapters, loop):
        seen.update(job=job, scope=var.get(), adapters=adapters)
        return "fired"

    out = await api_runs.on_turn_executor(_fire, "job-1", adapters={"a": 1}, loop=asyncio.get_running_loop())
    assert out == "fired"
    assert seen["scope"] == "profile-a" and seen["job"] == "job-1" and seen["adapters"] == {"a": 1}
