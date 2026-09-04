"""Physical worker completion, cancellation and context propagation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import threading
from types import SimpleNamespace

import pytest

from gateway.group_chat_work import run_group_command_work
from tests.gateway.test_hosted_room_messaging import _runner


@pytest.mark.asyncio
async def test_each_worker_inherits_its_own_context():
    value = ContextVar("group-work-profile", default="unset")
    runner = _runner()

    async def invoke(profile):
        token = value.set(profile)
        try:
            return await run_group_command_work(runner, "send", value.get)
        finally:
            value.reset(token)

    assert await asyncio.gather(invoke("alpha"), invoke("beta")) == ["alpha", "beta"]
    assert runner._active_deferred_agent_worker_count() == 0
    assert value.get() == "unset"


@pytest.mark.asyncio
async def test_cancel_before_worker_entry_never_calls_operation(monkeypatch):
    runner = _runner()
    loop = asyncio.get_running_loop()
    original = loop.run_in_executor
    entered = asyncio.Event()
    release = threading.Event()
    called = []

    def occupy():
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise AssertionError("test executor not released")

    with ThreadPoolExecutor(max_workers=1) as pool:
        blocker = original(pool, occupy)
        await asyncio.wait_for(entered.wait(), 2)
        monkeypatch.setattr(loop, "run_in_executor", lambda _pool, fn, *a: original(pool, fn, *a))
        task = asyncio.create_task(run_group_command_work(runner, "send", lambda: called.append(1)))
        try:
            await asyncio.sleep(0)
            assert runner._active_deferred_agent_worker_count() == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            await blocker
            # Drain the same executor after the cancelled queue entry.
            await original(pool, lambda: None)
            await asyncio.sleep(0)
            assert called == []
            assert runner._active_deferred_agent_worker_count() == 0
        finally:
            release.set()
            await asyncio.gather(task, blocker, return_exceptions=True)


@pytest.mark.asyncio
async def test_submission_failure_releases_registered_work(monkeypatch):
    runner = _runner()

    def reject(*_args):
        raise RuntimeError("executor unavailable")

    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", reject)
    with pytest.raises(RuntimeError, match="executor unavailable"):
        await run_group_command_work(runner, "send", lambda: pytest.fail("must not run"))
    assert runner._active_deferred_agent_worker_count() == 0


@pytest.mark.asyncio
async def test_missing_tracker_refuses_before_work():
    with pytest.raises(RuntimeError, match="tracking is unavailable"):
        await run_group_command_work(SimpleNamespace(), "send", lambda: pytest.fail("must not run"))


@pytest.mark.asyncio
async def test_unknown_action_cannot_bypass_admission():
    runner = _runner()
    runner._external_drain_active = True
    with pytest.raises(ValueError, match="Unknown Group Chat work action"):
        await run_group_command_work(runner, "unrecognized", lambda: pytest.fail("must not run"))
    assert runner._active_deferred_agent_worker_count() == 0


@pytest.mark.asyncio
async def test_worker_failure_propagates_and_releases_work():
    runner = _runner()

    def fail():
        raise ValueError("operation failed")

    with pytest.raises(ValueError, match="operation failed"):
        await run_group_command_work(runner, "send", fail)
    assert runner._active_deferred_agent_worker_count() == 0
