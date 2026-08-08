"""Regression tests for ``GatewaySupervisionMixin._spawn_supervised``.

Wave 1 god-file extraction (shard s3, cluster c2): the task-level
supervision logic moved verbatim from ``GatewayRunner`` into
``gateway/supervision_mixin.py``. These tests pin the supervision contract
that must survive the move:

1. Tasks are tracked in ``self._background_tasks`` (lazily initialized) and
   discarded when they finish.
2. A crashed task that the gateway is still running is respawned with capped
   exponential backoff, up to ``_MAX_SUPERVISED_RESTARTS`` rapid failures.
3. ``on_spawn`` is invoked on EVERY spawn (initial + internal respawns) so
   external live-handle trackers (e.g. ``_reconnect_watcher_task``) never go
   stale.
4. A run that lasted at least ``_SUPERVISED_HEALTHY_SECS`` resets the
   consecutive-failure counter (a healthy-then-crashed daemon is never
   abandoned).
5. Clean returns and crashes while the gateway is stopping never respawn.

The mixin is exercised on a bare ``object.__new__`` instance with stub
state, following the pattern used by the original ``_compile_mention_patterns``
test seam — no ``GatewayRunner`` construction needed.
"""
from __future__ import annotations

import asyncio

import pytest

import gateway.supervision_mixin as supervision_mixin
from gateway.supervision_mixin import GatewaySupervisionMixin

# Real asyncio.sleep captured before the tests patch module-level asyncio.sleep
# to collapse backoff delays (the mixin references asyncio.sleep at call time).
_REAL_SLEEP = asyncio.sleep


def _make_runner(*, running=True, max_restarts=2, healthy_secs=300.0):
    runner = object.__new__(GatewaySupervisionMixin)
    runner._background_tasks = None
    runner._running = running
    runner._MAX_SUPERVISED_RESTARTS = max_restarts
    runner._SUPERVISED_HEALTHY_SECS = healthy_secs
    return runner


async def _drive_until_idle(runner, max_iterations=400):
    """Yield to the loop until the supervisor's task set drains."""
    for _ in range(max_iterations):
        if not runner._background_tasks:
            return
        await _REAL_SLEEP(0.01)
    raise AssertionError("supervisor task set did not drain")


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Collapse the backoff sleep to a loop-yielding no-op."""

    async def _fast_sleep(delay):
        await _REAL_SLEEP(0)

    monkeypatch.setattr(supervision_mixin.asyncio, "sleep", _fast_sleep)
    yield


def test_spawn_tracks_task_and_discards_on_clean_return():
    """A clean-returning watcher is tracked, then discarded, never respawned."""

    async def main():
        runner = _make_runner()
        spawns = []

        async def watcher():
            return None

        task = runner._spawn_supervised(watcher, "clean_watcher", on_spawn=spawns.append)
        assert task in runner._background_tasks  # tracked immediately
        await _drive_until_idle(runner)
        assert len(spawns) == 1  # initial spawn only, no respawn on clean exit
        assert task.done() and not task.cancelled()
        assert runner._background_tasks == set()

    asyncio.run(main())


def test_crash_respawns_with_backoff_and_gives_up_at_cap():
    """Rapid crash-loop respawns until _MAX_SUPERVISED_RESTARTS, then stops."""

    async def main():
        runner = _make_runner(max_restarts=2)
        spawns = []

        async def always_crash():
            raise RuntimeError("watcher exploded")

        runner._spawn_supervised(always_crash, "crash_watcher", on_spawn=spawns.append)
        await _drive_until_idle(runner)
        # initial + respawn at _attempt=1 + respawn at _attempt=2, then give up
        assert len(spawns) == 3
        assert runner._background_tasks == set()

    asyncio.run(main())


def test_healthy_run_resets_restart_counter():
    """A run lasting >= _SUPERVISED_HEALTHY_SECS resets the rapid-failure count."""

    async def main():
        runner = _make_runner(max_restarts=1, healthy_secs=0.01)
        spawns = []
        calls = {"n": 0}

        async def crash_after_healthy_delay():
            calls["n"] += 1
            if calls["n"] == 2:  # second run is "healthy" (runs >= 0.01s)
                await _REAL_SLEEP(0.05)
            raise RuntimeError("watcher exploded")

        runner._spawn_supervised(
            crash_after_healthy_delay, "healthy_crash_watcher", on_spawn=spawns.append
        )
        await _drive_until_idle(runner)
        # Without the health reset: crash(0) -> respawn(1) -> crash(1) gives up
        # at cap=1 (2 spawns). With the reset, spawn(1) is treated as fresh
        # (attempt 0 again), so the give-up is deferred one more cycle (3 spawns).
        assert len(spawns) == 3
        assert runner._background_tasks == set()

    asyncio.run(main())


def test_no_respawn_when_gateway_stopped():
    """A crash while self._running is False never respawns."""

    async def main():
        runner = _make_runner(running=False)
        spawns = []

        async def always_crash():
            raise RuntimeError("watcher exploded")

        runner._spawn_supervised(always_crash, "stopped_watcher", on_spawn=spawns.append)
        await _drive_until_idle(runner)
        assert len(spawns) == 1  # initial spawn only
        assert runner._background_tasks == set()

    asyncio.run(main())


def test_on_spawn_receives_live_task_handle_on_every_spawn():
    """on_spawn fires on internal respawns too, with a live (not done) task."""

    async def main():
        runner = _make_runner(max_restarts=1)
        seen = []

        async def flaky():
            raise RuntimeError("flaky watcher")

        def record(handle):
            # The mixin calls on_spawn right after create_task, before the
            # loop ever runs the coroutine — the handle must still be live.
            seen.append(handle.done())

        runner._spawn_supervised(flaky, "flaky_watcher", on_spawn=record)
        await _drive_until_idle(runner)
        assert len(seen) == 2  # initial + one respawn (cap=1)
        assert seen == [False, False]  # every handle live at spawn time

    asyncio.run(main())


def test_spawn_supervised_resolves_via_gateway_runner_mro():
    """GatewayRunner must still expose _spawn_supervised through the mixin."""
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewaySupervisionMixin)
    assert callable(GatewayRunner._spawn_supervised)
