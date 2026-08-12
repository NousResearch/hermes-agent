"""Cross-process session turn lease behavior (#84234)."""

from __future__ import annotations

import os
import threading
import time

from hermes_state import SessionDB


def test_turn_lease_serializes_separate_session_db_instances(tmp_path):
    """A second process-shaped DB handle waits for the current turn owner."""
    path = tmp_path / "state.db"
    first = SessionDB(path)
    second = SessionDB(path)
    first.create_session("shared", source="test")

    first_holder = f"pid={os.getpid()}:turn=first"
    second_holder = f"pid={os.getpid()}:turn=second"
    assert first.try_acquire_session_turn_lease(
        "shared", first_holder, ttl_seconds=5
    )

    released = threading.Event()

    def release_first():
        time.sleep(0.2)
        first.release_session_turn_lease("shared", first_holder)
        released.set()

    thread = threading.Thread(target=release_first, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        assert second.acquire_session_turn_lease(
            "shared",
            second_holder,
            ttl_seconds=5,
            wait_seconds=2,
            poll_interval_seconds=0.02,
        )
    finally:
        thread.join(timeout=2)

    assert released.is_set()
    assert time.monotonic() - started >= 0.15
    second.release_session_turn_lease("shared", second_holder)


def test_turn_lease_is_scoped_to_conversation_root(tmp_path):
    """Compression descendants share one durable serialization domain."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.end_session("root", "compression")
    db.create_session("child", source="test", parent_session_id="root")

    root_holder = f"pid={os.getpid()}:turn=root"
    child_holder = f"pid={os.getpid()}:turn=child"
    assert db.try_acquire_session_turn_lease(
        "root", root_holder, ttl_seconds=5
    )
    assert not db.try_acquire_session_turn_lease(
        "child", child_holder, ttl_seconds=5
    )
    db.release_session_turn_lease("child", root_holder)


def test_turn_lease_does_not_serialize_delegate_child_with_parent(tmp_path):
    """Only compression continuation segments share a conversation lease."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="test")
    db.create_session(
        "delegate",
        source="delegate",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )

    parent_holder = f"pid={os.getpid()}:turn=parent"
    delegate_holder = f"pid={os.getpid()}:turn=delegate"
    assert db.try_acquire_session_turn_lease(
        "parent", parent_holder, ttl_seconds=5
    )
    assert db.try_acquire_session_turn_lease(
        "delegate", delegate_holder, ttl_seconds=5
    )


def test_turn_lease_refresh_and_release_are_owner_fenced(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")

    current_holder = f"pid={os.getpid()}:turn=current"
    stale_holder = f"pid={os.getpid()}:turn=stale"
    next_holder = f"pid={os.getpid()}:turn=next"
    assert db.try_acquire_session_turn_lease(
        "shared", current_holder, ttl_seconds=5
    )
    assert not db.refresh_session_turn_lease(
        "shared", stale_holder, ttl_seconds=5
    )
    db.release_session_turn_lease("shared", stale_holder)
    assert not db.try_acquire_session_turn_lease(
        "shared", next_holder, ttl_seconds=5
    )

    assert db.refresh_session_turn_lease(
        "shared", current_holder, ttl_seconds=5
    )
    db.release_session_turn_lease("shared", current_holder)
    assert db.try_acquire_session_turn_lease(
        "shared", next_holder, ttl_seconds=5
    )


def test_expired_turn_lease_is_reclaimed(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    assert db.try_acquire_session_turn_lease(
        "shared", "legacy-holder", ttl_seconds=0.05
    )

    time.sleep(0.15)

    assert db.try_acquire_session_turn_lease(
        "shared", "pid=202:turn=reclaimer", ttl_seconds=5
    )