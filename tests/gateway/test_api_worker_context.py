"""API executor jobs retain their submitting request's ContextVars."""

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.platforms import api_server_runs


def test_api_workers_snapshot_each_request_and_isolate_reused_threads():
    request = contextvars.ContextVar("api_worker_request", default="unbound")
    release = threading.Event()

    def worker():
        assert release.wait(5), "worker was not released"
        seen = request.get()
        request.set("worker mutation")
        return seen, threading.get_ident()

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        request.set("first request")
        first = api_server_runs._submit_api_worker(loop, worker)
        request.set("second request")
        second = api_server_runs._submit_api_worker(loop, worker)
        request.set("caller after submission")
        release.set()
        first_seen, first_thread = await first
        second_seen, second_thread = await second
        assert first_seen == "first request"
        assert second_seen == "second request"
        assert first_thread == second_thread
        assert request.get() == "caller after submission"
        assert await loop.run_in_executor(None, request.get) == "unbound"
        assert api_server_runs.api_worker_live_count() == 0

    asyncio.run(run())


def test_api_worker_failure_restores_context_and_releases_live_count():
    request = contextvars.ContextVar("api_worker_failing_request", default="unbound")

    def fail():
        seen = request.get()
        request.set("failed worker mutation")
        raise ValueError(seen)

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        request.set("failing request")
        with pytest.raises(ValueError, match="^failing request$"):
            await api_server_runs._submit_api_worker(loop, fail)
        assert request.get() == "failing request"
        assert await loop.run_in_executor(None, request.get) == "unbound"
        assert api_server_runs.api_worker_live_count() == 0

    asyncio.run(run())
