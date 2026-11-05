"""Unavailable MCP servers must not hold the gateway's local readiness gate closed."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run as gateway_run
from gateway.run_bootstrap import _start_gateway_run_runner


def _runner(**extra):
    """Minimal gateway runner double carrying the real background-task bookkeeping."""
    runner = SimpleNamespace(
        config=object(), should_exit_cleanly=False, _running=True, **extra,
    )

    def _retain_background_task(task):
        tasks = getattr(runner, "_background_tasks", None)
        if tasks is None:
            tasks = runner._background_tasks = set()
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    runner._retain_background_task = _retain_background_task
    return runner


@pytest.mark.asyncio
async def test_platform_start_does_not_wait_for_mcp_discovery(monkeypatch):
    discovery_started = asyncio.Event()
    release_discovery = asyncio.Event()
    platform_started = asyncio.Event()

    async def discover(_config):
        discovery_started.set()
        await release_discovery.wait()

    async def start():
        platform_started.set()
        return True

    runner = _runner(start=start)
    monkeypatch.setattr(gateway_run, "_discover_gateway_mcp_tools", discover)
    boot = asyncio.create_task(_start_gateway_run_runner(runner, [False]))
    try:
        await asyncio.wait_for(discovery_started.wait(), timeout=5)
        await asyncio.wait_for(platform_started.wait(), timeout=5)
        assert await asyncio.wait_for(asyncio.shield(boot), timeout=5) is None
        assert runner._background_tasks, "discovery must belong to the runner's shutdown lifecycle"
        assert not release_discovery.is_set(), "discovery should still be pending, not awaited"
    finally:
        release_discovery.set()
        await asyncio.gather(boot, *runner._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_mcp_failure_is_reported_without_failing_platform_start(monkeypatch, caplog):
    discovery_finished = asyncio.Event()

    async def discover(_config):
        discovery_finished.set()
        raise ConnectionError("test MCP unavailable")

    runner = _runner(start=AsyncMock(return_value=True))
    monkeypatch.setattr(gateway_run, "_discover_gateway_mcp_tools", discover)

    with caplog.at_level("WARNING", logger="gateway.run_bootstrap"):
        assert await asyncio.wait_for(_start_gateway_run_runner(runner, [False]), timeout=5) is None
        await asyncio.wait_for(discovery_finished.wait(), timeout=5)
        await asyncio.gather(*runner._background_tasks, return_exceptions=True)

    runner.start.assert_awaited_once()
    assert any("MCP tool discovery failed in background" in record.message
               for record in caplog.records), "a failed discovery must be visible, not silent"
