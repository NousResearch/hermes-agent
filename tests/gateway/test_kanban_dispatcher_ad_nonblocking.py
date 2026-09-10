"""Regression tests for #106985: the embedded dispatcher's auto-decompose pass
must not block the spawn step of the same tick.

``auto_decompose_tick`` used to be awaited inline before ``tick_once``, so one
slow triage LLM call delayed every unrelated ready-task spawn on every board
sharing the dispatcher. It now runs as a single in-flight background task: the
spawn step proceeds immediately, and a new decompose pass only starts once the
previous one has finished.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import gateway.kanban_watchers as kww
from gateway.kanban_watchers import GatewayKanbanWatchersMixin


class _FakeDispatcher:
    """Records call times; ``auto_decompose_tick`` blocks for ``ad_delay``.

    ``ad_crash`` simulates a decompose pass that raises instead of returning.
    """

    def __init__(self, state: dict) -> None:
        self.state = state

    def auto_decompose_tick(self, per_tick: int) -> int:
        self.state["ad_calls"] = self.state.get("ad_calls", 0) + 1
        if self.state.get("ad_crash"):
            raise RuntimeError("boom")
        time.sleep(self.state["ad_delay"])
        return 0

    def tick_once(self):
        self.state["spawn_times"] = self.state.get("spawn_times", []) + [time.monotonic()]
        if len(self.state["spawn_times"]) >= self.state["stop_after"]:
            self.state["runner"]._running = False
        return []

    def ready_nonempty(self) -> bool:
        return False


class _StubRunner(GatewayKanbanWatchersMixin):
    def __init__(self) -> None:
        self._running = True
        self.released = False

    def _kanban_dispatcher_boot(self):
        return (lambda: {}, SimpleNamespace(), {})

    def _release_kanban_dispatcher_lock(self) -> None:
        self.released = True


@pytest.fixture
def fast_sleep(monkeypatch):
    """Shrink every ``asyncio.sleep`` the watcher takes (boot delay, tick slices)."""
    real_sleep = asyncio.sleep

    async def _fast(delay, *args, **kwargs):
        return await real_sleep(min(delay, 0.01))

    monkeypatch.setattr(asyncio, "sleep", _fast)


def _patch_watcher(monkeypatch, state: dict) -> None:
    from hermes_cli import kanban_db_dispatch

    monkeypatch.setattr(kanban_db_dispatch, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kww, "_resolve_dispatcher_settings", lambda cfg, kb: SimpleNamespace(interval=0.05))
    monkeypatch.setattr(kww, "_resolve_auto_decompose_settings", lambda loader: (state.get("ad_enabled", True), 3))
    monkeypatch.setattr(kww, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kww, "_KanbanDispatcher", lambda kb, settings: _FakeDispatcher(state))
    monkeypatch.setattr(
        kww,
        "_to_thread_process_service",
        staticmethod(lambda func, *args: asyncio.to_thread(func, *args)),
    )


def _run_watcher_sync(monkeypatch, state: dict):
    runner = _StubRunner()
    state["runner"] = runner
    _patch_watcher(monkeypatch, state)

    async def _drive() -> None:
        await asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=10.0)

    asyncio.run(_drive())
    assert runner.released, "dispatcher lock must be released on exit"
    return runner


def test_slow_decompose_does_not_block_spawns(monkeypatch, fast_sleep):
    state = {"ad_delay": 0.5, "stop_after": 3}
    _run_watcher_sync(monkeypatch, state)

    spawn_times = state["spawn_times"]
    assert len(spawn_times) == 3
    # Inline await would serialize 3 ticks behind two 0.5s decompose calls;
    # unblocked spawns finish well inside one decompose delay.
    assert spawn_times[-1] - spawn_times[0] < 0.4
    # Single-flight guard: the first decompose is still in flight, so no retry.
    assert state["ad_calls"] == 1


def test_new_decompose_starts_after_previous_finishes(monkeypatch, fast_sleep):
    state = {"ad_delay": 0.02, "stop_after": 6}
    _run_watcher_sync(monkeypatch, state)

    assert len(state["spawn_times"]) == 6
    assert state["ad_calls"] >= 2, "single-flight guard must lift once a pass finishes"


def test_decompose_disabled_skips_pass(monkeypatch, fast_sleep):
    state = {"ad_delay": 0.0, "ad_enabled": False, "stop_after": 2}
    _run_watcher_sync(monkeypatch, state)

    assert len(state["spawn_times"]) == 2
    assert state.get("ad_calls", 0) == 0


def test_crashed_background_pass_never_stops_the_loop(monkeypatch, fast_sleep):
    state = {"ad_delay": 0.0, "ad_crash": True, "stop_after": 3}
    _run_watcher_sync(monkeypatch, state)

    # A crashing decompose pass is contained to its background task; the
    # spawn step still runs every tick and the loop shuts down cleanly.
    assert state["ad_calls"] >= 1
    assert len(state["spawn_times"]) == 3
