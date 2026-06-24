"""Tests for GatewayRunner._spawn_supervised — crash supervision for the
long-lived gateway watcher tasks.

The 6 long-lived watchers (session_expiry, kanban notifier, kanban
dispatcher, platform_reconnect, handoff, async_delegation) were previously
launched with bare ``asyncio.create_task`` — a raise in a watcher's OUTER
``while self._running:`` loop died silently. ``_spawn_supervised`` tracks the
task, logs crashes, and restarts with bounded exponential backoff.

Two invariants matter most:
  * A coroutine that returns CLEANLY is NOT respawned (respawning a
    synchronously-returning watcher would busy-spin the event loop).
  * Restarts are bounded by ``_MAX_SUPERVISED_RESTARTS`` so a permanently
    failing watcher can't restart-storm forever.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

import gateway.run
from gateway.run import GatewayRunner


def _make_runner():
    """Minimal GatewayRunner via object.__new__ to skip __init__."""
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._background_tasks = set()
    return runner


@pytest.mark.asyncio
async def test_clean_synchronous_return_is_not_respawned():
    """A supervised coro that returns immediately must be spawned exactly once.

    A clean return signals shutdown/disabled — respawning it would busy-spin.
    """
    runner = _make_runner()
    calls = 0

    async def quick_return():
        nonlocal calls
        calls += 1
        return  # clean return, no exception

    runner._spawn_supervised(quick_return, "quick")

    # Let the task (and any erroneously-scheduled respawn) run.
    await asyncio.sleep(0.05)

    assert calls == 1, (
        f"Expected a cleanly-returning watcher to run exactly once, ran {calls} times"
    )


@pytest.mark.asyncio
async def test_exception_restart_bounded_by_ceiling(monkeypatch):
    """A coro that always raises is invoked exactly _MAX_SUPERVISED_RESTARTS+1 times.

    The first spawn + _MAX_SUPERVISED_RESTARTS restarts, then the supervisor
    gives up. asyncio.sleep is monkeypatched to instant so backoff doesn't
    slow the test.
    """
    runner = _make_runner()
    calls = 0

    async def always_raises():
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    # Make the supervisor's backoff sleeps instant, but still yield to the
    # loop so respawn tasks actually run. Capture the real sleep first so the
    # drain loop below can still hand control to the event loop (the patch
    # below replaces the module-global asyncio.sleep everywhere).
    real_sleep = asyncio.sleep

    async def _instant_sleep(_delay):
        # Yield once so chained respawns get scheduled, but skip the backoff.
        await real_sleep(0)

    monkeypatch.setattr(gateway.run.asyncio, "sleep", _instant_sleep)

    runner._spawn_supervised(always_raises, "boomer")

    # Drain the chain of respawns. Each death schedules a respawn task which
    # spawns the next attempt; yield repeatedly until both the call count has
    # reached the ceiling AND no supervised tasks remain pending. Keep going a
    # few extra iterations after that to prove no over-restart occurs.
    expected = GatewayRunner._MAX_SUPERVISED_RESTARTS + 1
    stable = 0
    for _ in range(500):
        await asyncio.sleep(0)
        pending = [t for t in runner._background_tasks if not t.done()]
        if calls >= expected and not pending:
            stable += 1
            if stable >= 5:
                break
        else:
            stable = 0

    assert calls == expected, (
        f"Expected exactly {expected} invocations "
        f"(1 spawn + {GatewayRunner._MAX_SUPERVISED_RESTARTS} restarts), got {calls}"
    )


@pytest.mark.asyncio
async def test_create_task_called_without_name_kwarg(monkeypatch):
    """_spawn_supervised must NOT pass name= to create_task.

    A real test elsewhere mocks create_task with a signature that rejects the
    name kwarg; guard that contract here.
    """
    runner = _make_runner()
    captured = {}
    real_create_task = asyncio.create_task

    def _spy_create_task(coro, *args, **kwargs):
        captured["kwargs"] = kwargs
        return real_create_task(coro)

    monkeypatch.setattr(gateway.run.asyncio, "create_task", _spy_create_task)

    async def noop():
        return

    runner._spawn_supervised(noop, "noop")
    await asyncio.sleep(0.01)

    assert "name" not in captured.get("kwargs", {})
