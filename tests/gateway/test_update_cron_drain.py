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


