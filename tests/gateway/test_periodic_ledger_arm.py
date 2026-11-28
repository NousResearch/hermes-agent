"""The runner keeps re-arming redelivery timers for ledger rows that are still waiting.

Rows that failed while their platform session was not ready never re-arm by themselves: without the
periodic arm they wait for the next restart even after the session has silently revived.
"""

import asyncio

import pytest

from gateway import run_startup
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.run_startup import GatewayStartupMixin
from tests.waiting import async_wait_until


class _Runner(GatewayStartupMixin, GatewayAdapterLifecycleMixin):
    def __init__(self):
        self._running = True
        self.arms = 0

    async def _arm_flood_timers_for_waiting_rows(self) -> None:
        self.arms += 1


def _live_arms() -> int:
    return sum(1 for t in asyncio.all_tasks() if t.get_name() == "periodic-ledger-arm" and not t.done())


async def _stop(runner: _Runner) -> None:
    """Request shutdown and wait for the loop to notice it (one interval at most)."""
    runner._running = False
    await runner._periodic_ledger_arm_task


@pytest.fixture
def fast_interval(monkeypatch):
    monkeypatch.setattr(run_startup, "_LEDGER_ARM_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda config=None: True)


@pytest.mark.asyncio
async def test_re_arms_while_running_and_stops_when_not_running(fast_interval):
    runner = _Runner()
    runner._start_periodic_ledger_arm()

    await async_wait_until(lambda: runner.arms >= 2, what="the periodic arm fired more than once")
    assert runner._periodic_ledger_arm_task in runner._background_tasks, "shutdown must cancel it"

    stopped_at = runner.arms
    await _stop(runner)
    assert runner.arms == stopped_at, "no arm once shutdown was requested"


@pytest.mark.asyncio
async def test_second_start_does_not_duplicate_the_loop(fast_interval):
    runner = _Runner()
    runner._start_periodic_ledger_arm()
    runner._start_periodic_ledger_arm()

    await async_wait_until(lambda: runner.arms >= 1, what="the periodic arm fired")
    assert _live_arms() == 1
    await _stop(runner)


@pytest.mark.asyncio
async def test_noop_when_the_ledger_is_off(monkeypatch):
    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda config=None: False)

    runner = _Runner()
    runner._start_periodic_ledger_arm()
    await asyncio.sleep(0)  # yield: a loop would have been scheduled by now

    assert runner.arms == 0
    assert getattr(runner, "_periodic_ledger_arm_task", None) is None
    assert _live_arms() == 0
