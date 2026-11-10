"""Regression tests for #60432.

``/update`` (and other gateway shutdown paths) must drain in-flight cron jobs
before ``process_registry.kill_all()`` runs in final-cleanup.  Cron work runs on
a thread-pool worker and is tracked in ``cron.scheduler._running_job_ids``, not
in ``GatewayRunner._running_agents`` — so a zero-agent drain must still wait
for cron to finish (or time out and take the interrupt/kill path).
"""
import asyncio
from unittest.mock import patch

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_drain_active_agents_waits_for_in_flight_cron_jobs():
    runner, _adapter = make_restart_runner()
    runner._running_agents = {}

    observed = asyncio.Event()
    released = asyncio.Event()

    def _cron_in_flight():
        observed.set()
        return frozenset() if released.is_set() else frozenset({"job-1"})

    with patch("cron.scheduler.get_running_job_ids", side_effect=_cron_in_flight):
        drain = asyncio.create_task(runner._drain_active_agents(10.0))
        try:
            await asyncio.wait_for(observed.wait(), timeout=5.0)
            assert not drain.done(), "drain returned while cron still owned its work"
            released.set()
            _snapshot, timed_out = await asyncio.wait_for(drain, timeout=5.0)
        finally:
            released.set()
            if not drain.done():
                drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    assert timed_out is False
    assert _snapshot == {}


