"""Regression tests for #60432.

``/update`` (and other gateway shutdown paths) must drain in-flight cron jobs
before ``process_registry.kill_all()`` runs in final-cleanup.  Cron work runs on
a thread-pool worker and is tracked in ``cron.scheduler._running_job_ids``, not
in ``GatewayRunner._running_agents`` — so a zero-agent drain must still wait
for cron to finish (or time out and take the interrupt/kill path).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_drain_active_agents_waits_for_in_flight_cron_jobs():
    runner, _adapter = make_restart_runner()
    runner._running_agents = {}

    cron_count = [1]

    def _cron_in_flight():
        return frozenset(f"job-{i}" for i in range(cron_count[0]))

    async def finish_cron():
        await asyncio.sleep(0.15)
        cron_count[0] = 0

    with patch("cron.scheduler.get_running_job_ids", side_effect=_cron_in_flight):
        task = asyncio.create_task(finish_cron())
        _snapshot, timed_out = await runner._drain_active_agents(1.0)
        await task

    assert timed_out is False
    assert _snapshot == {}


@pytest.mark.asyncio
async def test_drain_active_agents_times_out_when_cron_still_running():
    runner, _adapter = make_restart_runner()
    runner._running_agents = {}

    with patch("cron.scheduler.get_running_job_ids", return_value=frozenset({"job-1"})):
        _snapshot, timed_out = await runner._drain_active_agents(0.05)

    assert timed_out is True
    assert _snapshot == {}


@pytest.mark.asyncio
async def test_gateway_stop_waits_for_cron_before_final_tool_kill():
    """Graceful cron completion must finish before final-cleanup kill_all."""
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 1.0

    cron_count = [1]
    call_order: list[str] = []

    def _cron_in_flight():
        return frozenset(f"job-{i}" for i in range(cron_count[0]))

    def _fake_kill_all(task_id=None):
        call_order.append("kill_all")
        return 0

    async def finish_cron():
        await asyncio.sleep(0.12)
        cron_count[0] = 0

    with (
        patch("cron.scheduler.get_running_job_ids", side_effect=_cron_in_flight),
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.write_runtime_status"),
        patch("agent.auxiliary_client.shutdown_cached_clients"),
        patch("tools.process_registry.process_registry") as registry_mock,
    ):
        registry_mock.kill_all.side_effect = _fake_kill_all
        adapter.disconnect = AsyncMock()

        cron_task = asyncio.create_task(finish_cron())
        await runner.stop()
        await cron_task

    assert call_order == ["kill_all"]
