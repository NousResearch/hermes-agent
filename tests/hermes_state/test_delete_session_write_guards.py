"""Invariant tests for entry-side deletion refusal on active write guards (#123583)."""

import os
import pytest
from hermes_state import SessionDB
from hermes_state_errors import SessionActiveWriteGuardError


def test_delete_session_refuses_when_write_guard_active(tmp_path):
    """Invariant 1: delete_session(..., exclude_active_write_guards=True) refuses inside
    the transaction while an active turn lease or compression lock protects the row."""
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("sess-lease", source="test")
    db.create_session("sess-cmp", source="test")

    turn_holder = f"pid={os.getpid()}:turn=1"
    cmp_holder = f"pid={os.getpid()}:cmp=1"

    assert db.try_acquire_session_turn_lease("sess-lease", turn_holder, ttl_seconds=300.0) is True
    assert db.try_acquire_compression_lock("sess-cmp", cmp_holder, ttl_seconds=300.0) is True

    # 1. Active turn lease -> raises SessionActiveWriteGuardError and row survives
    with pytest.raises(SessionActiveWriteGuardError):
        db.delete_session("sess-lease", exclude_active_write_guards=True)
    assert db.get_session("sess-lease") is not None

    # 2. Active compression lock -> raises SessionActiveWriteGuardError and row survives
    with pytest.raises(SessionActiveWriteGuardError):
        db.delete_session("sess-cmp", exclude_active_write_guards=True)
    assert db.get_session("sess-cmp") is not None

    # 3. Released guards -> deletion succeeds
    db.release_session_turn_lease("sess-lease", turn_holder)
    db.release_compression_lock("sess-cmp", cmp_holder)

    assert db.delete_session("sess-lease", exclude_active_write_guards=True) is True
    assert db.get_session("sess-lease") is None

    assert db.delete_session("sess-cmp", exclude_active_write_guards=True) is True
    assert db.get_session("sess-cmp") is None

    # 4. An idle compression-ended row is a closed parent, not a live write: delete must succeed
    #    (no CompressionSessionClosedError leaking out of the guard check).
    db.create_session("sess-compressed", source="test")
    db.end_session("sess-compressed", "compression")
    assert db.delete_session("sess-compressed", exclude_active_write_guards=True) is True
    assert db.get_session("sess-compressed") is None
    db.close()


def test_delete_sessions_bulk_skips_active_write_guards(tmp_path):
    """Invariant 2: delete_sessions(..., exclude_active_write_guards=True) atomically
    skips rows with active guards and removes only the idle ones."""
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("bulk-active", source="test")
    db.create_session("bulk-idle", source="test")

    turn_holder = f"pid={os.getpid()}:turn=bulk"
    assert db.try_acquire_session_turn_lease("bulk-active", turn_holder, ttl_seconds=300.0) is True

    skipped: list[str] = []
    deleted_count = db.delete_sessions(
        ["bulk-active", "bulk-idle"], exclude_active_write_guards=True, skipped_ids=skipped)
    assert deleted_count == 1
    assert skipped == ["bulk-active"]  # reported to the caller, not silently dropped

    # Protected row survived; idle row was deleted
    assert db.get_session("bulk-active") is not None
    assert db.get_session("bulk-idle") is None

    db.release_session_turn_lease("bulk-active", turn_holder)

    # Delegate children cascade with their root, so a guarded child protects the root too:
    # single delete refuses, bulk delete skips the root and never cascades the guarded child away.
    db.create_session("deleg-root", source="test")
    db.create_session(
        "deleg-child", source="test", parent_session_id="deleg-root",
        model_config={"_delegate_from": "deleg-root"},
    )
    child_holder = f"pid={os.getpid()}:turn=child"
    assert db.try_acquire_session_turn_lease("deleg-child", child_holder, ttl_seconds=300.0) is True
    with pytest.raises(SessionActiveWriteGuardError):
        db.delete_session("deleg-root", exclude_active_write_guards=True)
    skipped = []
    assert db.delete_sessions(["deleg-root"], exclude_active_write_guards=True, skipped_ids=skipped) == 0
    assert skipped == ["deleg-root"]
    assert db.get_session("deleg-root") is not None and db.get_session("deleg-child") is not None
    db.release_session_turn_lease("deleg-child", child_holder)
    db.close()


def test_delete_session_if_empty_refuses_when_write_guard_active(tmp_path):
    """delete_session_if_empty shares the guarded-delete refusal: the emptiness predicate
    reads committed state, so a row whose first turn is leased but not yet flushed would
    be deleted mid-turn (#123583)."""
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("empty-live", source="test")
    turn_holder = f"pid={os.getpid()}:turn=1"
    assert db.try_acquire_session_turn_lease("empty-live", turn_holder, ttl_seconds=300.0) is True

    with pytest.raises(SessionActiveWriteGuardError):
        db.delete_session_if_empty("empty-live")
    assert db.get_session("empty-live") is not None

    db.release_session_turn_lease("empty-live", turn_holder)
    assert db.delete_session_if_empty("empty-live") is True
    assert db.get_session("empty-live") is None
    db.close()
