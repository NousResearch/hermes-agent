"""Shutdown accounting for executor work detached by hygiene timeouts."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.run import run_codex_hygiene_compaction
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_deferred_worker_remains_active_until_executor_future_finishes():
    runner, _adapter = make_restart_runner()
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    worker = asyncio.get_running_loop().create_future()
    agent = MagicMock()

    runner._defer_agent_cleanup_until_future_done(
        worker,
        agent,
        context="test hygiene timeout",
    )

    assert runner._active_deferred_agent_worker_count() == 1
    assert runner._active_work_count() == 1

    worker.set_result(([], None))
    await asyncio.gather(*runner._deferred_agent_cleanup_tasks)

    assert runner._active_deferred_agent_worker_count() == 0
    runner._cleanup_agent_resources_off_loop.assert_awaited_once_with(
        agent,
        context="test hygiene timeout",
    )


@pytest.mark.asyncio
async def test_timed_out_codex_hygiene_worker_remains_visible_to_shutdown():
    runner, _adapter = make_restart_runner()
    started = threading.Event()
    release = threading.Event()

    class BlockingCodexAgent:
        _codex_session = object()
        context_compressor = SimpleNamespace(compression_count=0)

        def _compress_context(self, *_args, **_kwargs):
            started.set()
            release.wait(timeout=5.0)

    agent = BlockingCodexAgent()
    runner._agent_cache = {"tg:123": (agent, 0.0)}
    runner._agent_cache_lock = None

    outcome = await run_codex_hygiene_compaction(
        runner,
        "tg:123",
        "sess-1",
        auto_mode="hermes",
        history=[{"role": "user", "content": "hello"}],
        approx_tokens=100,
        timeout_seconds=0.01,
        failure_cooldown_seconds=-1.0,
    )

    assert started.is_set()
    assert outcome == "failed:timeout"
    assert runner._active_deferred_agent_worker_count() == 1
    with patch("gateway.run.request_hard_interrupt") as interrupt:
        runner._interrupt_running_agents("gateway shutdown")
    interrupt.assert_called_once_with(agent, "gateway shutdown")

    release.set()
    for _ in range(100):
        if runner._active_deferred_agent_worker_count() == 0:
            break
        await asyncio.sleep(0.01)
    assert runner._active_deferred_agent_worker_count() == 0


@pytest.mark.asyncio
async def test_shutdown_drain_waits_for_deferred_hygiene_worker():
    runner, _adapter = make_restart_runner()
    worker = asyncio.get_running_loop().create_future()
    runner._deferred_agent_workers = {worker: MagicMock()}

    async def finish_worker():
        await asyncio.sleep(0.12)
        worker.set_result(([], None))

    finisher = asyncio.create_task(finish_worker())
    _snapshot, timed_out = await runner._drain_active_agents(2.0)
    await finisher

    assert timed_out is False
    assert _snapshot == {}


@pytest.mark.asyncio
async def test_deferred_hygiene_worker_times_out_and_receives_interrupt():
    runner, _adapter = make_restart_runner()
    worker = asyncio.get_running_loop().create_future()
    agent = MagicMock()
    runner._deferred_agent_workers = {worker: agent}

    _snapshot, timed_out = await runner._drain_active_agents(0.01)

    assert timed_out is True
    assert _snapshot == {}
    with patch("gateway.run.request_hard_interrupt") as interrupt:
        runner._interrupt_running_agents("gateway shutdown")
    interrupt.assert_called_once_with(agent, "gateway shutdown")

    worker.cancel()


@pytest.mark.asyncio
async def test_stop_interrupts_deferred_worker_before_teardown():
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.01
    worker = asyncio.get_running_loop().create_future()

    class DeferredAgent:
        def __init__(self):
            self.interrupts = []

        def hard_interrupt(self, reason):
            self.interrupts.append(reason)
            if not worker.done():
                worker.set_result(([], None))

    agent = DeferredAgent()
    runner._deferred_agent_workers = {worker: agent}
    adapter.disconnect = AsyncMock()

    with (
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.write_runtime_status"),
        patch("cron.scheduler.mark_job_run"),
        patch("tools.process_registry.process_registry.kill_all", return_value=0),
        patch("tools.terminal_tool.cleanup_all_environments"),
        patch("tools.browser_tool_lifecycle.cleanup_all_browsers"),
    ):
        await runner.stop()

    assert agent.interrupts == ["Gateway shutting down"]
    assert worker.done()
