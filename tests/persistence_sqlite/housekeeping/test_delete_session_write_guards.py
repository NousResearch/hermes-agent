"""Invariant tests for entry-side deletion refusal on active write guards on a SQLite backend store (#123583): ``delete_session``, ``delete_sessions`` and ``delete_session_if_empty`` against the coordination repository's turn lease and compression lock."""

import os

import pytest

from hermes_state_errors import SessionActiveWriteGuardError
from persistence.sqlite.conversation._store.sessions import (
    create_session, delete_session, delete_session_if_empty, delete_sessions, end_session, get_session,
)
from tests.contracts._scopes import MEDIA_ROOT


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def test_delete_session_refuses_when_write_guard_active(db, store):
    """Invariant 1: delete_session(..., exclude_active_write_guards=True) refuses inside
    the transaction while an active turn lease or compression lock protects the row."""
    create_session(db, "sess-lease", source="test")
    create_session(db, "sess-cmp", source="test")

    turn_holder = f"pid={os.getpid()}:turn=1"
    cmp_holder = f"pid={os.getpid()}:cmp=1"

    assert store.coordination.try_acquire_turn_lease(
        session_id="sess-lease", holder=turn_holder, ttl_seconds=300.0) is True
    assert store.coordination.try_acquire_compression_lock(
        session_id="sess-cmp", holder=cmp_holder, ttl_seconds=300.0) is True

    # 1. Active turn lease -> raises SessionActiveWriteGuardError and row survives
    with pytest.raises(SessionActiveWriteGuardError):
        delete_session(db, "sess-lease", exclude_active_write_guards=True, media_root=MEDIA_ROOT)
    assert get_session(db, "sess-lease") is not None

    # 2. Active compression lock -> raises SessionActiveWriteGuardError and row survives
    with pytest.raises(SessionActiveWriteGuardError):
        delete_session(db, "sess-cmp", exclude_active_write_guards=True, media_root=MEDIA_ROOT)
    assert get_session(db, "sess-cmp") is not None

    # 3. Released guards -> deletion succeeds
    store.coordination.release_turn_lease(session_id="sess-lease", holder=turn_holder)
    store.coordination.release_compression_lock(session_id="sess-cmp", holder=cmp_holder)

    assert "sess-lease" in delete_session(db, "sess-lease", exclude_active_write_guards=True, media_root=MEDIA_ROOT)
    assert get_session(db, "sess-lease") is None

    assert "sess-cmp" in delete_session(db, "sess-cmp", exclude_active_write_guards=True, media_root=MEDIA_ROOT)
    assert get_session(db, "sess-cmp") is None

    # 4. An idle compression-ended row is a closed parent, not a live write: delete must succeed
    #    (no CompressionSessionClosedError leaking out of the guard check).
    create_session(db, "sess-compressed", source="test")
    end_session(db, "sess-compressed", "compression")
    assert "sess-compressed" in delete_session(
        db, "sess-compressed", exclude_active_write_guards=True, media_root=MEDIA_ROOT)
    assert get_session(db, "sess-compressed") is None


def test_delete_sessions_bulk_skips_active_write_guards(db, store):
    """Invariant 2: delete_sessions(..., exclude_active_write_guards=True) atomically
    skips rows with active guards and removes only the idle ones."""
    create_session(db, "bulk-active", source="test")
    create_session(db, "bulk-idle", source="test")

    turn_holder = f"pid={os.getpid()}:turn=bulk"
    assert store.coordination.try_acquire_turn_lease(
        session_id="bulk-active", holder=turn_holder, ttl_seconds=300.0) is True

    skipped: list[str] = []
    deleted_count = delete_sessions(
        db, ["bulk-active", "bulk-idle"], exclude_active_write_guards=True, skipped_ids=skipped,
        media_root=MEDIA_ROOT)[0]
    assert deleted_count == 1
    assert skipped == ["bulk-active"]  # reported to the caller, not silently dropped

    # Protected row survived; idle row was deleted
    assert get_session(db, "bulk-active") is not None
    assert get_session(db, "bulk-idle") is None

    store.coordination.release_turn_lease(session_id="bulk-active", holder=turn_holder)

    # Delegate children cascade with their root, so a guarded child protects the root too:
    # single delete refuses, bulk delete skips the root and never cascades the guarded child away.
    create_session(db, "deleg-root", source="test")
    create_session(
        db, "deleg-child", source="test", parent_session_id="deleg-root",
        model_config={"_delegate_from": "deleg-root"},
    )
    child_holder = f"pid={os.getpid()}:turn=child"
    assert store.coordination.try_acquire_turn_lease(
        session_id="deleg-child", holder=child_holder, ttl_seconds=300.0) is True
    with pytest.raises(SessionActiveWriteGuardError):
        delete_session(db, "deleg-root", exclude_active_write_guards=True, media_root=MEDIA_ROOT)
    skipped = []
    assert delete_sessions(
        db, ["deleg-root"], exclude_active_write_guards=True, skipped_ids=skipped, media_root=MEDIA_ROOT)[0] == 0
    assert skipped == ["deleg-root"]
    assert get_session(db, "deleg-root") is not None and get_session(db, "deleg-child") is not None
    store.coordination.release_turn_lease(session_id="deleg-child", holder=child_holder)


def test_delete_session_if_empty_refuses_when_write_guard_active(db, store):
    """delete_session_if_empty shares the guarded-delete refusal: the emptiness predicate
    reads committed state, so a row whose first turn is leased but not yet flushed would
    be deleted mid-turn (#123583)."""
    create_session(db, "empty-live", source="test")
    turn_holder = f"pid={os.getpid()}:turn=1"
    assert store.coordination.try_acquire_turn_lease(
        session_id="empty-live", holder=turn_holder, ttl_seconds=300.0) is True

    assert delete_session_if_empty(db, "empty-live", media_root=MEDIA_ROOT) == ()
    assert get_session(db, "empty-live") is not None

    store.coordination.release_turn_lease(session_id="empty-live", holder=turn_holder)
    assert "empty-live" in delete_session_if_empty(db, "empty-live", media_root=MEDIA_ROOT)
    assert get_session(db, "empty-live") is None
