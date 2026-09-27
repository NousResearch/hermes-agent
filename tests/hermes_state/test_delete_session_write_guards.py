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
    db.close()
