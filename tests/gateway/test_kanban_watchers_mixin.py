"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


@pytest.mark.asyncio
@pytest.mark.parametrize(("raw_max_spawn", "expected"), [("4", 4), (0, None)])
async def test_dispatcher_normalizes_positive_max_spawn(
    tmp_path, monkeypatch, raw_max_spawn, expected
):
    """YAML strings become ints and values below one are ignored."""
    import gateway.kanban_watchers as watchers
    from hermes_cli import kanban_db as kb

    runner = SimpleNamespace(_running=True, _kanban_dispatcher_lock_handle=None)
    captured = {}

    class _Conn:
        def close(self):
            pass

    async def no_sleep(_delay):
        pass

    def dispatch_once(_conn, **kwargs):
        captured.update(kwargs)
        runner._running = False
        return SimpleNamespace(spawned=[])

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "auto_decompose": False,
                "max_spawn": raw_max_spawn,
            }
        },
    )
    monkeypatch.setattr(watchers, "_acquire_singleton_lock", lambda _path: (None, "unavailable"))
    monkeypatch.setattr(watchers.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(kb, "kanban_db_path", lambda _slug=None: Path(tmp_path / "kanban.db"))
    monkeypatch.setattr(kb, "list_boards", lambda **_kwargs: [{"slug": "default"}])
    monkeypatch.setattr(kb, "connect", lambda **_kwargs: _Conn())
    monkeypatch.setattr(kb, "dispatch_once", dispatch_once)
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda _conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda _conn: False)

    await GatewayKanbanWatchersMixin._kanban_dispatcher_watcher(runner)

    assert captured["max_spawn"] == expected
