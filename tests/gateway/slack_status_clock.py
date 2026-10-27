"""A simulated clock for the Slack status line over a multi-minute turn.

It drives ONLY the code under test: ``plugins/platforms/slack/adapter_status``'s ``time`` and
``asyncio.sleep`` bindings and the base ``_keep_typing`` loop's ``asyncio.sleep``. The event loop's
own clock stays real (``tests/module_clock.py``), so no loop timeout moves with simulated time."""
import asyncio
from unittest.mock import AsyncMock

import gateway.platforms.base as _base
import plugins.platforms.slack.adapter_status as _status
from tests.module_clock import ModuleClock

_real_sleep = asyncio.sleep


class _ModuleAsyncio:
    """``asyncio`` for one module, except that ``sleep`` is the simulated one."""

    def __init__(self, sleep):
        self.sleep = sleep

    def __getattr__(self, name):
        return getattr(asyncio, name)


class SimulatedTurnClock(ModuleClock):
    """``sleep`` advances ``now`` instead of blocking (each delay recorded in ``sleeps``). With
    ``turn_seconds`` set, the first sleep at or past it cancels the sleeping task, so a
    ``_keep_typing`` loop ends right after its tick at that instant — by iteration count, never
    by a timeout."""

    def __init__(self, turn_seconds: float | None = None):
        super().__init__(0.0)
        self.turn_seconds = turn_seconds
        self.sleeps: list[float] = []

    async def sleep(self, delay=0, result=None):
        if self.turn_seconds is not None and self.now >= self.turn_seconds:
            asyncio.current_task().cancel()
        else:
            self.sleeps.append(delay)
            self.now += delay
        await _real_sleep(0)  # still yield, so concurrent tasks interleave
        return result

    def install(self, monkeypatch) -> "SimulatedTurnClock":
        monkeypatch.setattr(_status, "time", self)
        module_asyncio = _ModuleAsyncio(self.sleep)
        monkeypatch.setattr(_status, "asyncio", module_asyncio)
        monkeypatch.setattr(_base, "asyncio", module_asyncio)
        return self

    def record(self, setter: AsyncMock) -> list[tuple[float, str]]:
        """``(simulated time, status)`` of every call to ``setter`` from now on."""
        calls: list[tuple[float, str]] = []
        setter.side_effect = lambda **kwargs: calls.append((self.now, kwargs["status"]))
        return calls
