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
import threading
import time
from types import SimpleNamespace

import pytest

import gateway.kanban_watchers as kww
from gateway.kanban_watchers import GatewayKanbanWatchersMixin


class _FakeDispatcher:
    """Records call times; ``auto_decompose_tick`` blocks for ``ad_delay``.

    ``ad_crash`` simulates a decompose pass that raises instead of returning.
    ``ad_gate`` is a ``(started, unblock)`` event pair that parks the pass on
    the unblock event, for shutdown-ownership tests.
    """

    def __init__(self, state: dict) -> None:
        self.state = state

    def auto_decompose_tick(self, per_tick: int) -> int:
        self.state["ad_calls"] = self.state.get("ad_calls", 0) + 1
        gate = self.state.get("ad_gate")
        if gate is not None:
            started, unblock = gate
            started.set()
            assert unblock.wait(timeout=30.0), "test harness failed to unblock the pass"
            self.state["ad_finished_at"] = time.monotonic()
            return 0
        if self.state.get("ad_crash"):
            raise RuntimeError("boom")
        time.sleep(self.state["ad_delay"])
        return 0

    def tick_once(self):
        self.state["spawn_times"] = self.state.get("spawn_times", []) + [
            time.monotonic()
        ]
        if len(self.state["spawn_times"]) >= self.state["stop_after"]:
            self.state["runner"]._running = False
        return []

    def ready_nonempty(self) -> bool:
        return False


class _StubRunner(GatewayKanbanWatchersMixin):
    def __init__(self) -> None:
        self._running = True
        self.released = False
        self.released_at: float | None = None

    def _kanban_dispatcher_boot(self):
        return (lambda: {}, SimpleNamespace(), {})

    def _release_kanban_dispatcher_lock(self) -> None:
        self.released = True
        self.released_at = time.monotonic()


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
    monkeypatch.setattr(
        kww,
        "_resolve_dispatcher_settings",
        lambda cfg, kb: SimpleNamespace(interval=0.05),
    )
    monkeypatch.setattr(
        kww,
        "_resolve_auto_decompose_settings",
        lambda loader: (state.get("ad_enabled", True), 3),
    )
    monkeypatch.setattr(kww, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(
        kww, "_KanbanDispatcher", lambda kb, settings: _FakeDispatcher(state)
    )
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


def _run_watcher_thread(
    monkeypatch, state: dict, runner: _StubRunner, drive
) -> threading.Thread:
    _patch_watcher(monkeypatch, state)
    thread = threading.Thread(target=asyncio.run, args=(drive(),), daemon=True)
    thread.start()
    return thread


def test_stop_holds_dispatcher_lock_until_decompose_quiesces(monkeypatch, fast_sleep):
    """Normal loop exit must not release the lease while the pass is in flight.

    Cancelling the wrapper task would not stop a ``to_thread`` worker that is
    already running (#106998): releasing the machine-global dispatcher lock at
    that point lets a replacement gateway start a second concurrent decompose
    pass. The lock may only be released after the callable has returned.
    """
    started, unblock = threading.Event(), threading.Event()
    state = {"ad_delay": 0.0, "stop_after": 1, "ad_gate": (started, unblock)}
    runner = _StubRunner()
    state["runner"] = runner

    async def _drive() -> None:
        await asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=30.0)

    thread = _run_watcher_thread(monkeypatch, state, runner, _drive)
    assert started.wait(timeout=5.0), "decompose pass must be in flight"
    time.sleep(0.3)  # let the loop observe _running=False and park in the drain
    assert not runner.released, (
        "lock must stay held while the decompose callable is in flight"
    )
    unblock.set()
    thread.join(timeout=15.0)

    assert not thread.is_alive(), "watcher thread must finish after the pass unblocks"
    assert runner.released
    assert state["ad_finished_at"] < runner.released_at, (
        "lease release must happen strictly after the decompose callable quiesces"
    )


def test_cancel_holds_dispatcher_lock_until_decompose_quiesces(monkeypatch, fast_sleep):
    """Cancelling the watcher (gateway shutdown) must drain the pass first too.

    The replacement-gateway window from #106998 is most reachable on this
    path: the old gateway is cancelled while a slow decompose is running, and
    the next gateway would immediately re-acquire the released lease.
    """
    started, unblock = threading.Event(), threading.Event()
    state = {"ad_delay": 0.0, "stop_after": 99, "ad_gate": (started, unblock)}
    runner = _StubRunner()
    state["runner"] = runner

    async def _drive() -> None:
        task = asyncio.create_task(runner._kanban_dispatcher_watcher())
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    thread = _run_watcher_thread(monkeypatch, state, runner, _drive)
    assert started.wait(timeout=5.0), "decompose pass must be in flight"
    time.sleep(0.3)  # let the CancelledError handler park in the drain
    assert not runner.released, (
        "cancelled watcher must still hold the lease during the pass"
    )
    unblock.set()
    thread.join(timeout=15.0)

    assert not thread.is_alive(), "watcher thread must finish after the pass unblocks"
    assert runner.released
    assert state["ad_finished_at"] < runner.released_at, (
        "lease release must happen strictly after the decompose callable quiesces"
    )
