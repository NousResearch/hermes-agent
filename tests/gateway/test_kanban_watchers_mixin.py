"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect

from gateway import kanban_watchers as watchers
from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _is_quarantine_fence_error,
    _should_skip_process_local_quarantine,
    _should_skip_quarantined_board,
)
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_safety as safety

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


def test_same_fingerprint_quarantine_does_not_expire_with_time(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    db_path.write_bytes(b"broken")
    safety.quarantine_board(db_path, reason="malformed", source="watcher-test")

    monkeypatch.setattr("gateway.kanban_watchers.time.monotonic", lambda: 10**12)

    assert _should_skip_quarantined_board(db_path)


def test_process_local_fallback_only_recovers_after_fingerprint_change():
    disabled: dict[str, object] = {"tour-platform": {"path": "/tmp/db", "db": "old"}}

    assert _should_skip_process_local_quarantine(
        disabled, "tour-platform", {"path": "/tmp/db", "db": "old"}
    )
    assert disabled

    assert not _should_skip_process_local_quarantine(
        disabled, "tour-platform", {"path": "/tmp/db", "db": "new"}
    )
    assert disabled == {}


def test_process_local_fence_recovers_after_board_generation_bump(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path):
        pass
    board_state_fingerprint = getattr(watchers, "_board_state_fingerprint")
    initial_fingerprint = board_state_fingerprint(db_path)
    disabled = {"tour-platform": initial_fingerprint}

    assert _should_skip_process_local_quarantine(
        disabled, "tour-platform", initial_fingerprint
    )

    safety.bump_board_generation(db_path)
    current_fingerprint = board_state_fingerprint(db_path)

    assert not _should_skip_process_local_quarantine(
        disabled, "tour-platform", current_fingerprint
    )
    assert disabled == {}


def test_gateway_only_classifies_marker_persistence_failure_for_local_fallback(tmp_path):
    marker = {"reason": "malformed"}
    assert not _is_quarantine_fence_error(
        safety.BoardQuarantinedError(tmp_path, marker)
    )
    quarantine_persistence_error = getattr(safety, "QuarantinePersistenceError")
    assert _is_quarantine_fence_error(
        quarantine_persistence_error("marker fsync failed")
    )
    assert not _is_quarantine_fence_error(safety.MaintenanceLockError("busy"))


def test_persistent_marker_prevents_redundant_process_local_fence(tmp_path):
    db_path = tmp_path / "kanban.db"
    db_path.write_bytes(b"broken")
    safety.quarantine_board(db_path, reason="malformed", source="watcher-test")

    should_record = getattr(watchers, "_should_record_process_local_quarantine")
    assert not should_record(
        db_path, is_corruption_error=True
    )


def test_marker_persistence_failure_requires_process_local_fence(tmp_path):
    db_path = tmp_path / "kanban.db"
    db_path.write_bytes(b"broken")

    should_record = getattr(watchers, "_should_record_process_local_quarantine")
    assert should_record(
        db_path, is_corruption_error=True
    )


def test_watcher_recovers_only_after_clear_or_board_generation_change(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path):
        pass
    safety.quarantine_board(
        db_path, reason="maintenance", source="watcher-test"
    )
    assert _should_skip_quarantined_board(db_path)

    current_fingerprint = safety.db_fingerprint(db_path)
    safety.clear_quarantine(db_path, expected_fingerprint=current_fingerprint)
    assert not _should_skip_quarantined_board(db_path)

    safety.quarantine_board(db_path, reason="malformed", source="watcher-test")
    safety.bump_board_generation(db_path)
    assert not _should_skip_quarantined_board(db_path)
