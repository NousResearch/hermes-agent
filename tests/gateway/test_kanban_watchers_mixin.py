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


# ── _rate_limited_log — controlled-time rate-limit tests ────────────────────


def test_rate_limit_initial_warning(caplog):
    """First failure logs at WARNING level."""
    import logging

    from gateway.kanban_watchers import _rate_limited_log

    warn_state = [0]
    caplog.set_level(logging.DEBUG, logger="gateway.run")
    _rate_limited_log(warn_state, 500, "test skipped: %s", "no client")
    assert warn_state[0] == 500  # timestamp updated
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "test skipped: no client" in warnings[0].message


def test_rate_limit_debug_suppression(caplog):
    """Subsequent failures within 5 minutes stay at DEBUG."""
    import logging

    from gateway.kanban_watchers import _rate_limited_log

    warn_state = [100]  # last warn at t=100
    caplog.set_level(logging.DEBUG, logger="gateway.run")
    _rate_limited_log(warn_state, 200, "test skipped: %s", "no client")
    assert warn_state[0] == 100  # timestamp NOT updated (still suppressed)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 0
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debugs) == 1
    assert "test skipped: no client" in debugs[0].message


def test_rate_limit_warning_after_five_minutes(caplog):
    """After 300 seconds, the WARNING boundary resets."""
    import logging

    from gateway.kanban_watchers import _rate_limited_log

    warn_state = [100]  # last warn at t=100
    caplog.set_level(logging.DEBUG, logger="gateway.run")
    _rate_limited_log(warn_state, 400, "test skipped: %s", "no client")
    assert warn_state[0] == 400  # timestamp updated
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "test skipped: no client" in warnings[0].message


def test_rate_limit_exact_boundary(caplog):
    """Exactly 300 seconds since last warn triggers WARNING."""
    import logging

    from gateway.kanban_watchers import _rate_limited_log

    warn_state = [100]
    caplog.set_level(logging.DEBUG, logger="gateway.run")
    _rate_limited_log(warn_state, 400, "test skipped: %s", "no client")  # 300s >= 300
    assert warn_state[0] == 400  # updated
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_rate_limit_just_under_boundary(caplog):
    """299 seconds since last warn stays at DEBUG."""
    import logging

    from gateway.kanban_watchers import _rate_limited_log

    warn_state = [100]
    caplog.set_level(logging.DEBUG, logger="gateway.run")
    _rate_limited_log(warn_state, 399, "test skipped: %s", "no client")  # 299s < 300
    assert warn_state[0] == 100  # NOT updated
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debugs) == 1
