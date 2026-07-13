"""Tests for #63529: the gateway shutdown drain was structurally blind to
in-flight api_server (desk/API) agent runs.

API-server runs are tracked only inside ``APIServerAdapter``
(``_inflight_agent_runs`` + ``_active_run_agents``) and never enter
``GatewayRunner._running_agents``. Without folding them into the drain,
stop/restart reported ``active_at_start=0`` and let systemd SIGKILL mid-tool.
Mirrors tests/gateway/test_cron_active_work_drain.py for cron (#60432).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


def _make_api_adapter(*, inflight: int = 0, active_ids=None):
    from gateway.config import Platform

    active = {rid: MagicMock() for rid in (active_ids or [])}
    adapter = SimpleNamespace(
        platform=Platform.API_SERVER,
        _inflight_agent_runs=inflight,
        _active_run_agents=active,
    )

    def active_agent_work_count() -> int:
        return int(adapter._inflight_agent_runs) + len(adapter._active_run_agents)

    adapter.active_agent_work_count = active_agent_work_count
    return adapter


class TestActiveApiRunCount:
    def test_zero_when_no_api_adapters(self):
        runner, _adapter = make_restart_runner()
        runner.adapters = {}
        runner._profile_adapters = {}
        assert runner._active_api_run_count() == 0

    def test_sums_inflight_and_active_run_agents(self):
        runner, _adapter = make_restart_runner()
        adapter = _make_api_adapter(inflight=2, active_ids=["r1"])
        runner.adapters = {"api": adapter}
        runner._profile_adapters = {}
        assert runner._active_api_run_count() == 3

    def test_includes_profile_adapters(self):
        runner, _adapter = make_restart_runner()
        runner.adapters = {"api": _make_api_adapter(inflight=1)}
        runner._profile_adapters = {"p1": _make_api_adapter(active_ids=["a", "b"])}
        assert runner._active_api_run_count() == 3

    def test_ignores_non_api_platforms(self):
        from gateway.config import Platform

        runner, _adapter = make_restart_runner()
        other = SimpleNamespace(
            platform=Platform.DISCORD,
            _inflight_agent_runs=99,
            _active_run_agents={"x": MagicMock()},
            active_agent_work_count=lambda: 99,
        )
        runner.adapters = {"discord": other}
        runner._profile_adapters = {}
        assert runner._active_api_run_count() == 0

    def test_never_raises_on_broken_adapters(self):
        runner, _adapter = make_restart_runner()

        class Bad:
            platform = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

        runner.adapters = {"bad": Bad()}
        runner._profile_adapters = None
        assert runner._active_api_run_count() == 0


class TestAPIServerAdapterWorkCount:
    def test_active_agent_work_count_on_real_class_method(self):
        from gateway.platforms.api_server import APIServerAdapter

        # Instantiate unbound helpers via object.__new__ to avoid full connect.
        adapter = object.__new__(APIServerAdapter)
        adapter._inflight_agent_runs = 2
        adapter._active_run_agents = {"r1": object(), "r2": object()}
        assert adapter.active_agent_work_count() == 4


class TestDrainWaitsForApiWork:
    @pytest.mark.asyncio
    async def test_drain_returns_immediately_when_nothing_active(self):
        runner, _adapter = make_restart_runner()
        runner.adapters = {}
        runner._profile_adapters = {}

        _snapshot, timed_out = await runner._drain_active_agents(5.0)

        assert timed_out is False

    @pytest.mark.asyncio
    async def test_drain_waits_for_in_flight_api_run(self):
        runner, _adapter = make_restart_runner()
        api = _make_api_adapter(inflight=1)
        runner.adapters = {"api": api}
        runner._profile_adapters = {}

        async def finish_run():
            await asyncio.sleep(0.12)
            api._inflight_agent_runs = 0

        task = asyncio.create_task(finish_run())
        _snapshot, timed_out = await runner._drain_active_agents(2.0)
        await task

        assert timed_out is False, (
            "drain must wait for api_server work, not report active_at_start=0"
        )

    @pytest.mark.asyncio
    async def test_drain_times_out_if_api_run_outlives_the_window(self):
        runner, _adapter = make_restart_runner()
        runner.adapters = {"api": _make_api_adapter(inflight=1)}
        runner._profile_adapters = {}

        _snapshot, timed_out = await runner._drain_active_agents(0.1)

        assert timed_out is True

    @pytest.mark.asyncio
    async def test_drain_still_waits_for_chat_and_cron(self):
        import cron.scheduler as sched

        runner, _adapter = make_restart_runner()
        runner._running_agents = {"session-1": MagicMock()}
        sched._running_job_ids.add("job-1")
        runner.adapters = {"api": _make_api_adapter(inflight=1)}
        runner._profile_adapters = {}

        async def finish_all():
            await asyncio.sleep(0.12)
            runner._running_agents.clear()
            sched._running_job_ids.discard("job-1")
            runner.adapters["api"]._inflight_agent_runs = 0

        task = asyncio.create_task(finish_all())
        try:
            _snapshot, timed_out = await runner._drain_active_agents(2.0)
        finally:
            await task
            sched._running_job_ids.discard("job-1")

        assert timed_out is False
