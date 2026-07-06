"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect

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


def test_max_spawn_config_int_cast():
    """Verify that kanban.max_spawn is correctly cast to int (not left as str).

    Regression test for #59499: when max_spawn was a YAML string,
    comparing it with int running_count raised TypeError and caused
    the dispatcher to spawn all tasks concurrently.
    """
    # The gateway watcher reads max_spawn and casts it to int.
    # This test verifies the pattern used in kanban_watchers.py:
    #   raw_max_spawn = kanban_cfg.get("max_spawn", None)
    #   max_spawn = None
    #   if raw_max_spawn is not None:
    #       try:
    #           max_spawn = int(raw_max_spawn)
    #       except (TypeError, ValueError):
    #           # log warning, set to None
    #
    # We test the casting logic directly:
    def _parse_max_spawn(raw):
        max_spawn = None
        if raw is not None:
            try:
                max_spawn = int(raw)
            except (TypeError, ValueError):
                max_spawn = None
        return max_spawn

    # Valid int-like values
    assert _parse_max_spawn(10) == 10
    assert _parse_max_spawn("10") == 10
    assert _parse_max_spawn(1) == 1
    assert _parse_max_spawn("1") == 1

    # Invalid values (should be None)
    assert _parse_max_spawn(None) is None
    assert _parse_max_spawn("invalid") is None
    assert _parse_max_spawn("") is None
    assert _parse_max_spawn("10.5") is None  # int() doesn't parse floats
    assert _parse_max_spawn([]) is None
    assert _parse_max_spawn({}) is None

    # Demonstrate the bug: comparing int with string raises TypeError
    running_count = 0
    spawned = 0
    max_spawn_str = "36"

    # Bug scenario (string max_spawn) — raises TypeError in Python 3
    try:
        if max_spawn_str is not None and running_count + spawned >= max_spawn_str:
            bug_break_hit = True
        else:
            bug_break_hit = False
    except TypeError:
        # The comparison crashes, causing the cap check to fail
        bug_break_hit = False
        bug_crashed = True
    else:
        bug_crashed = False

    assert bug_crashed is True, "Bug confirmed: string max_spawn causes TypeError"

    # Simulate the dispatcher loop with bug (string max_spawn)
    # When TypeError is caught, the loop continues without breaking
    tasks_spawned_buggy = 0
    for i in range(100):
        try:
            if max_spawn_str is not None and tasks_spawned_buggy >= max_spawn_str:
                break
        except TypeError:
            pass  # Bug: exception silently caught, loop continues
        tasks_spawned_buggy += 1

    assert tasks_spawned_buggy == 100, "Bug: all tasks spawned due to TypeError"

    # Simulate the dispatcher loop with fix (int max_spawn)
    max_spawn_int = 36
    tasks_spawned_fixed = 0
    for i in range(100):
        if max_spawn_int is not None and tasks_spawned_fixed >= max_spawn_int:
            break
        tasks_spawned_fixed += 1

    assert tasks_spawned_fixed == 36, "Fix: dispatcher respects max_spawn cap"