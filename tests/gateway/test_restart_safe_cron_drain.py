"""The final restart drain must honor the scheduler's profile-scoped worker split."""
import asyncio
from unittest.mock import AsyncMock

import pytest

import cron.scheduler as scheduler
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [True, False])
async def test_final_drain_only_skips_isolated_worker_on_restart(tmp_path, monkeypatch, restart):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: [])
    runner, _ = make_restart_runner()
    runner._restart_requested = restart
    assert scheduler.try_register_running_job("isolated")
    scheduler._record_external_cron_worker("isolated", 4321, scope_isolated=True)

    async def finish_on_poll(_delay):
        scheduler.release_running_job("isolated")

    poll = AsyncMock(side_effect=finish_on_poll)
    monkeypatch.setattr("gateway.run_shutdown.asyncio.sleep", poll)
    try:
        _, timed_out = await runner._drain_active_agents(timeout=0, cron_timeout=180)
        assert not timed_out
        assert poll.await_count == (0 if restart else 1)
        # Restart must leave ownership intact for the worker and its durable delivery.
        assert scheduler.get_restart_wait_cron_counts()["restart_safe"] == (1 if restart else 0)
    finally:
        scheduler.release_running_job("isolated")


@pytest.mark.asyncio
async def test_final_restart_drain_waits_for_same_id_in_another_profile(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: [])
    runner, _ = make_restart_runner()
    runner._restart_requested = True
    homes = [tmp_path / "a", tmp_path / "b"]
    for index, home in enumerate(homes):
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert scheduler.try_register_running_job("shared-id")
        scheduler._record_external_cron_worker("shared-id", 4321 + index, scope_isolated=index == 0)
    polls = []

    async def finish_dependent_worker(_delay):
        polls.append(True)
        assert len(polls) == 1, "the surviving isolated worker must not hold the drain"
        scheduler.release_running_job("shared-id", homes[1])

    monkeypatch.setattr("gateway.run_shutdown.asyncio.sleep", finish_dependent_worker)
    try:
        _, timed_out = await asyncio.wait_for(runner._drain_active_agents(0, 180), timeout=5)
        assert not timed_out
        assert len(polls) == 1, "the worker sharing the gateway lifetime must finish before restart"
        assert scheduler.get_restart_wait_cron_counts()["restart_safe"] == 1
    finally:
        for home in homes:
            scheduler.release_running_job("shared-id", home)
