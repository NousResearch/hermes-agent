"""Regression: the gateway graceful-shutdown drain was structurally blind to in-flight
background subagents (``delegate_task(background=true)``). They run as daemon threads in the
owning process, tracked by ``tools.async_delegation`` and therefore outside
``GatewayRunner._running_agents`` -- the dict every other active-work check reads. A graceful
restart (an upgrade) could report ``active_at_start=0`` and exit, killing a near-finished
subagent mid-write; the next boot's ``recover_abandoned_delegations`` then delivers only
``outcome unknown`` instead of the real result.

Mirror of the #60432 cron fix: background delegations now ride the drain (the cron floor), so a
graceful restart waits for them (bounded) before the process exits.

  - _active_delegation_count() reads async_delegation.active_count()
  - _active_work_count() / _drain_work_counts() include them
  - _drain_active_agents() waits for in-flight delegations
"""

import asyncio

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


def _reset_async_delegation_records():
    import tools.async_delegation as ad
    with ad._records_lock:
        ad._records.clear()


@pytest.fixture(autouse=True)
def _cleanup_async_delegations():
    _reset_async_delegation_records()
    yield
    _reset_async_delegation_records()


class TestDrainWaitsForBackgroundDelegations:
    @pytest.mark.asyncio
    async def test_drain_waits_for_in_flight_delegation(self, monkeypatch):
        """A delegation-only workload must make the drain wait (not report idle and return
        instantly) -- the exact repro from the upgrade-killed-subagent incident."""
        import tools.async_delegation as ad

        runner, _adapter = make_restart_runner()
        state = {"n": 1}
        monkeypatch.setattr(ad, "active_count", lambda: state["n"])

        async def finish_delegation():
            await asyncio.sleep(0.12)
            state["n"] = 0

        task = asyncio.create_task(finish_delegation())
        _snapshot, timed_out = await runner._drain_active_agents(2.0)
        await task

        assert timed_out is False, (
            "drain must wait for the in-flight background delegation to finish, "
            "not report active_at_start=0 and return instantly"
        )

    def test_active_work_count_includes_delegations(self, monkeypatch):
        import tools.async_delegation as ad

        runner, _ = make_restart_runner()
        monkeypatch.setattr(ad, "active_count", lambda: 0)
        base = runner._active_work_count()
        monkeypatch.setattr(ad, "active_count", lambda: 3)
        assert runner._active_work_count() == base + 3, (
            "_active_work_count must reflect in-flight background delegations"
        )

    def test_drain_work_counts_exposes_delegation_slot(self, monkeypatch):
        import tools.async_delegation as ad

        runner, _ = make_restart_runner()
        monkeypatch.setattr(ad, "active_count", lambda: 2)
        counts = runner._drain_work_counts()
        assert len(counts) == 5, "_drain_work_counts must expose the delegation slot"
        assert counts[4] == 2, "delegations is the 5th slot of _drain_work_counts"


class TestManagedWorkerDelegationDrain:
    """The orphan/owner-death path in agent/managed_worker.py must drain in-flight
    background subagents before the process-group SIGKILL."""

    def test_drain_returns_zero_when_idle(self, monkeypatch):
        import agent.managed_worker as mw
        import tools.async_delegation as ad

        monkeypatch.setattr(ad, "active_count", lambda: 0)
        assert mw._drain_background_delegations(5.0) == 0

    def test_drain_zero_budget_returns_remaining(self, monkeypatch):
        import agent.managed_worker as mw
        import tools.async_delegation as ad

        monkeypatch.setattr(ad, "active_count", lambda: 2)
        # budget 0 -> the first liveness check already past the deadline -> returns the count.
        assert mw._drain_background_delegations(0.0) == 2
