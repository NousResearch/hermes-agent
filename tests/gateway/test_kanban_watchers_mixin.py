"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect

from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _dispatcher_health_is_stuck,
)
from hermes_cli.kanban_db import DispatchResult

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


def test_dispatcher_health_does_not_call_capacity_backpressure_stuck():
    capped = DispatchResult(skipped_global_capped=True)

    assert _dispatcher_health_is_stuck(
        [("default", capped)], ready_boards={"default"}
    ) is False


def test_dispatcher_health_still_flags_unexplained_zero_spawn():
    unexplained = DispatchResult()

    assert _dispatcher_health_is_stuck(
        [("default", unexplained)], ready_boards={"default"}
    ) is True


def test_dispatcher_health_does_not_hide_unexplained_board_behind_capped_board():
    capped = DispatchResult(skipped_global_capped=True)
    unexplained = DispatchResult()

    assert _dispatcher_health_is_stuck(
        [("capped", capped), ("broken", unexplained)],
        ready_boards={"capped", "broken"},
    ) is True


def test_dispatcher_health_does_not_hide_unexplained_board_behind_active_board():
    active = DispatchResult(spawned=[("t1", "default", "/tmp/work")])
    unexplained = DispatchResult()

    assert _dispatcher_health_is_stuck(
        [("active", active), ("broken", unexplained)],
        ready_boards={"active", "broken"},
    ) is True
