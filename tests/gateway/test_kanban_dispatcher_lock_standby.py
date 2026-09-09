"""A gateway that loses the dispatcher-lock race must stand by, not give up.

Before this, losing the race at boot was a verdict for the whole process
life: the watcher returned. When the holder later exited, the lock was free
but every other gateway had already stopped asking, so the board was not
dispatched by anyone until a human restarted a gateway.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers as kw
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.kanban_watchers_common import (
    _acquire_singleton_lock,
    _release_singleton_lock,
)


class _Runner(GatewayKanbanWatchersMixin):
    def __init__(self):
        self._running = True
        self._kanban_dispatcher_lock_handle = None


def _lock_path(tmp_path):
    path = tmp_path / "kanban" / ".dispatcher.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_may_dispatch_is_false_while_contended_and_true_once_free(tmp_path):
    """The per-tick question is re-asked, and its answer follows the lock."""
    lock_path = _lock_path(tmp_path)
    holder, state = _acquire_singleton_lock(lock_path)
    assert state == "held"

    runner = _Runner()
    runner._kanban_dispatcher_lock_path = lock_path
    try:
        assert runner._may_dispatch_this_tick() is False
        assert runner._owns_kanban_dispatcher_lock() is False
        # Asking twice must not corrupt state or leak a handle.
        assert runner._may_dispatch_this_tick() is False
    finally:
        _release_singleton_lock(holder)

    assert runner._may_dispatch_this_tick() is True
    assert runner._owns_kanban_dispatcher_lock() is True
    # Already the owner: no second acquisition, still True.
    assert runner._may_dispatch_this_tick() is True
    runner._release_kanban_dispatcher_lock()


def test_may_dispatch_is_true_when_locking_is_unavailable(tmp_path):
    """No advisory locking -> config-only control, never a frozen board."""
    runner = _Runner()
    runner._kanban_dispatcher_lock_path = None
    assert runner._may_dispatch_this_tick() is True
    assert runner._owns_kanban_dispatcher_lock() is False


class _FakeDispatcher:
    def __init__(self, *_args, **_kwargs):
        self.ticks = 0

    def tick_once(self):
        self.ticks += 1
        return []

    def ready_nonempty(self):
        return False

    def auto_decompose_tick(self, _per_tick):
        return None


def test_watcher_keeps_ticking_and_takes_over_when_the_holder_exits(
    tmp_path, monkeypatch,
):
    """End to end: contended at boot, dispatching after the holder releases."""
    lock_path = _lock_path(tmp_path)
    holder, state = _acquire_singleton_lock(lock_path)
    assert state == "held"

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *a, **kw_: {"kanban": {"dispatch_in_gateway": True}},
    )
    monkeypatch.setattr(kw, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(
        kw, "_resolve_auto_decompose_settings", lambda _loader: (False, 0)
    )
    monkeypatch.setattr(
        kw, "_resolve_dispatcher_settings",
        lambda _cfg, _kb: SimpleNamespace(interval=1.0),
    )
    monkeypatch.setattr(kw, "_log_spawn_results", lambda _results: False)

    dispatchers = []

    def _make_dispatcher(*args, **kwargs):
        made = _FakeDispatcher(*args, **kwargs)
        dispatchers.append(made)
        return made

    monkeypatch.setattr(kw, "_KanbanDispatcher", _make_dispatcher)

    runner = _Runner()
    ticks_at_release = {}
    real_sleep = asyncio.sleep

    async def fake_sleep(_delay):
        # Every await yields; the counter drives the scenario.
        fake_sleep.calls += 1
        if fake_sleep.calls == 5:
            # Five standby sleeps happened with the lock held elsewhere:
            # nothing may have been dispatched yet.
            ticks_at_release["before"] = sum(d.ticks for d in dispatchers)
            _release_singleton_lock(holder)
        if fake_sleep.calls >= 12:
            runner._running = False
        await real_sleep(0)

    fake_sleep.calls = 0
    monkeypatch.setattr(kw.asyncio, "sleep", fake_sleep)

    async def _drive():
        await asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=10)

    try:
        asyncio.run(_drive())
    finally:
        runner._release_kanban_dispatcher_lock()

    assert fake_sleep.calls >= 12, (
        "the watcher exited instead of standing by while the lock was contended "
        f"(only {fake_sleep.calls} sleeps happened)"
    )
    assert dispatchers, "the dispatcher loop exited before it ever built a dispatcher"
    assert ticks_at_release.get("before") == 0, (
        "the standby gateway dispatched while another one held the lock"
    )
    assert sum(d.ticks for d in dispatchers) > 0, (
        "the gateway never took over the lock after the holder released it"
    )
    assert runner._owns_kanban_dispatcher_lock() is False  # released above


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
