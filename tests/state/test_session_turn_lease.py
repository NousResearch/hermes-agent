"""Cross-process session turn lease behavior (#84234)."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_state import SessionDB, SessionTurnLeaseLostError


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


def test_turn_lease_walks_compression_child_that_inherited_fork_markers(tmp_path):
    """Inherited ``_delegate_from`` / ``_branched_from`` must not stop the walk.

    ``publish_compression_child`` copies ``model_config`` verbatim, so a
    delegate or branch continuation carries a marker pointing at some other
    session. Presence-only fork detection would key the child separately:
    the holder still owns the parent-key lease, but the first refresh after
    rotation looks up the child id and fail-closes with a hard interrupt.
    """
    db = SessionDB(tmp_path / "state.db")
    db.create_session("original-parent", source="test")
    db.create_session(
        "delegate",
        source="delegate",
        parent_session_id="original-parent",
        model_config={"_delegate_from": "original-parent"},
    )
    db.end_session("delegate", "compression")
    db.create_session(
        "delegate-continuation",
        source="delegate",
        parent_session_id="delegate",
        model_config={"_delegate_from": "original-parent"},
    )
    db.create_session(
        "branch",
        source="test",
        parent_session_id="original-parent",
        model_config={"_branched_from": "original-parent"},
    )
    db.end_session("branch", "compression")
    db.create_session(
        "branch-continuation",
        source="test",
        parent_session_id="branch",
        model_config={"_branched_from": "original-parent"},
    )

    assert db._session_turn_lease_key("delegate-continuation") == "delegate"
    assert db._session_turn_lease_key("branch-continuation") == "branch"

    delegate_holder = f"pid={os.getpid()}:turn=delegate"
    assert db.try_acquire_session_turn_lease(
        "delegate", delegate_holder, ttl_seconds=5
    )
    assert not db.try_acquire_session_turn_lease(
        "delegate-continuation",
        f"pid={os.getpid()}:turn=delegate-child",
        ttl_seconds=5,
    )
    assert db.refresh_session_turn_lease(
        "delegate-continuation", delegate_holder, ttl_seconds=5
    )

    branch_holder = f"pid={os.getpid()}:turn=branch"
    assert db.try_acquire_session_turn_lease(
        "branch", branch_holder, ttl_seconds=5
    )
    assert not db.try_acquire_session_turn_lease(
        "branch-continuation",
        f"pid={os.getpid()}:turn=branch-child",
        ttl_seconds=5,
    )
    assert db.refresh_session_turn_lease(
        "branch-continuation", branch_holder, ttl_seconds=5
    )

    original_holder = f"pid={os.getpid()}:turn=original"
    assert db.try_acquire_session_turn_lease(
        "original-parent", original_holder, ttl_seconds=5
    )
    db.release_session_turn_lease("delegate-continuation", delegate_holder)
    db.release_session_turn_lease("branch-continuation", branch_holder)
    db.release_session_turn_lease("original-parent", original_holder)


def test_turn_lease_write_txn_does_not_trust_fail_open_key_helper(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Acquire/refresh/release walk inside the write txn.

    The old helper swallowed get_session failures and returned the child id.
    P2 then proceeded to acquire; the write succeeded under that child key
    and the first working refresh walked to the parent and hard-interrupted.
    Poisoning the outer helper must not change the conversation key.
    """
    db = SessionDB(tmp_path / "state.db")
    db.create_session(
        "delegate",
        source="delegate",
        model_config={"_delegate_from": "original-parent"},
    )
    db.end_session("delegate", "compression")
    db.create_session(
        "delegate-continuation",
        source="delegate",
        parent_session_id="delegate",
        model_config={"_delegate_from": "original-parent"},
    )

    monkeypatch.setattr(db, "_session_turn_lease_key", lambda sid: sid)
    holder = f"pid={os.getpid()}:turn=delegate"
    assert db.try_acquire_session_turn_lease(
        "delegate", holder, ttl_seconds=5
    )
    assert not db.try_acquire_session_turn_lease(
        "delegate-continuation",
        f"pid={os.getpid()}:turn=child",
        ttl_seconds=5,
    )
    assert db.refresh_session_turn_lease(
        "delegate-continuation", holder, ttl_seconds=5
    )
    db.release_session_turn_lease("delegate-continuation", holder)
    assert db.try_acquire_session_turn_lease(
        "delegate", f"pid={os.getpid()}:turn=next", ttl_seconds=5
    )


def test_turn_lease_retries_locked_in_txn_key_walk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A locked lineage walk must retry, not INSERT under the child id."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session(
        "delegate",
        source="delegate",
        model_config={"_delegate_from": "original-parent"},
    )
    db.end_session("delegate", "compression")
    db.create_session(
        "delegate-continuation",
        source="delegate",
        parent_session_id="delegate",
        model_config={"_delegate_from": "original-parent"},
    )

    attempts = {"n": 0}
    original = db._session_turn_lease_key_on_conn

    def flaky_walk(conn, session_id):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return original(conn, session_id)

    monkeypatch.setattr(db, "_session_turn_lease_key_on_conn", flaky_walk)
    holder = f"pid={os.getpid()}:turn=delegate"
    assert db.try_acquire_session_turn_lease(
        "delegate-continuation", holder, ttl_seconds=5
    )
    assert attempts["n"] >= 2
    monkeypatch.setattr(db, "_session_turn_lease_key_on_conn", original)
    assert not db.try_acquire_session_turn_lease(
        "delegate", f"pid={os.getpid()}:turn=other", ttl_seconds=5
    )
    assert db.refresh_session_turn_lease("delegate", holder, ttl_seconds=5)
    db.release_session_turn_lease("delegate-continuation", holder)


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


def test_acquire_turn_lease_notifies_wait_callback(tmp_path):
    """Waiters get a progress callback while another holder owns the lease."""
    path = tmp_path / "state.db"
    first = SessionDB(path)
    second = SessionDB(path)
    first.create_session("shared", source="test")

    first_holder = f"pid={os.getpid()}:turn=first"
    second_holder = f"pid={os.getpid()}:turn=second"
    assert first.try_acquire_session_turn_lease(
        "shared", first_holder, ttl_seconds=5
    )

    notices = []

    def release_first():
        time.sleep(0.12)
        first.release_session_turn_lease("shared", first_holder)

    thread = threading.Thread(target=release_first, daemon=True)
    thread.start()
    try:
        assert second.acquire_session_turn_lease(
            "shared",
            second_holder,
            ttl_seconds=5,
            wait_seconds=2,
            poll_interval_seconds=0.02,
            on_wait=notices.append,
            wait_notice_interval_seconds=0.05,
        )
    finally:
        thread.join(timeout=2)

    assert notices
    assert notices[0] < 0.05
    second.release_session_turn_lease("shared", second_holder)


def test_acquire_turn_lease_honors_should_abort(tmp_path):
    """Waiters stop immediately when should_abort() returns True."""
    path = tmp_path / "state.db"
    first = SessionDB(path)
    second = SessionDB(path)
    first.create_session("shared", source="test")

    first_holder = f"pid={os.getpid()}:turn=first"
    second_holder = f"pid={os.getpid()}:turn=second"
    assert first.try_acquire_session_turn_lease(
        "shared", first_holder, ttl_seconds=60
    )

    abort_checks = {"count": 0}

    def should_abort():
        abort_checks["count"] += 1
        return True

    started = time.monotonic()
    assert not second.acquire_session_turn_lease(
        "shared",
        second_holder,
        wait_seconds=30,
        poll_interval_seconds=0.05,
        should_abort=should_abort,
    )
    assert time.monotonic() - started < 1.0
    assert abort_checks["count"] >= 1
    first.release_session_turn_lease("shared", first_holder)


def test_acquire_turn_lease_retries_sqlite_lock(tmp_path, monkeypatch):
    """Write-lock exhaustion is contended, not a hard abort of the wait."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    holder = f"pid={os.getpid()}:turn=waiter"
    attempts = {"n": 0}
    original = db.try_acquire_session_turn_lease

    def flaky_acquire(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError(
                "database is locked (another Hermes process held the "
                "state.db write lock for over 20s)"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "try_acquire_session_turn_lease", flaky_acquire)
    assert db.acquire_session_turn_lease(
        "shared",
        holder,
        wait_seconds=2,
        poll_interval_seconds=0.02,
        acquire_patience_s=0.05,
    )
    assert attempts["n"] >= 2
    db.release_session_turn_lease("shared", holder)


def test_acquire_turn_lease_reraises_non_lock_sqlite_error(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")

    def disk_full(*args, **kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(db, "try_acquire_session_turn_lease", disk_full)
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        db.acquire_session_turn_lease(
            "shared",
            f"pid={os.getpid()}:turn=waiter",
            wait_seconds=1,
            poll_interval_seconds=0.02,
        )


def test_non_expired_turn_lease_from_dead_pid_is_reclaimed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holder whose structured pid= no longer exists can be reclaimed early."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")

    dead_holder = "pid=424242:turn=dead:platform=test"
    assert db.try_acquire_session_turn_lease(
        "shared", dead_holder, ttl_seconds=300
    ) is True

    probed: list[int] = []

    def pid_exists(pid: int) -> bool:
        probed.append(pid)
        return False

    monkeypatch.setattr(
        hermes_state, "psutil", SimpleNamespace(pid_exists=pid_exists)
    )

    fresh_holder = "pid=525252:turn=fresh:platform=test"
    assert db.try_acquire_session_turn_lease(
        "shared", fresh_holder, ttl_seconds=300
    ) is True
    assert probed == [424242]


def test_turn_lease_fences_stale_transcript_flush_after_reclaim(tmp_path):
    """A lost holder cannot persist after B has taken the conversation.

    Refresh-loss interrupt is cooperative; the lease itself must reject the
    late append inside the same SQLite write transaction.
    """
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    stale_holder = f"pid={os.getpid()}:turn=stale"
    next_holder = f"pid={os.getpid()}:turn=next"

    assert db.try_acquire_session_turn_lease(
        "shared", stale_holder, ttl_seconds=5
    )
    assert db.append_messages_batch(
        "shared",
        [{"role": "user", "content": "stale-owned"}],
        turn_lease_holder=stale_holder,
    ) == 1

    db.release_session_turn_lease("shared", stale_holder)
    assert db.try_acquire_session_turn_lease(
        "shared", next_holder, ttl_seconds=5
    )

    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        db.append_messages_batch(
            "shared",
            [{"role": "assistant", "content": "late stale reply"}],
            turn_lease_holder=stale_holder,
        )
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        db.append_message(
            "shared",
            "assistant",
            "late stale single-row",
            turn_lease_holder=stale_holder,
        )

    assert db.append_messages_batch(
        "shared",
        [{"role": "assistant", "content": "next reply"}],
        turn_lease_holder=next_holder,
    ) == 1
    assert [m["content"] for m in db.get_messages("shared")] == [
        "stale-owned",
        "next reply",
    ]
    db.release_session_turn_lease("shared", next_holder)


def test_turn_lease_fences_flush_when_row_is_absent_or_expired(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    holder = f"pid={os.getpid()}:turn=owner"

    assert db.try_acquire_session_turn_lease("shared", holder, ttl_seconds=0.05)
    time.sleep(0.12)
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        db.append_messages_batch(
            "shared",
            [{"role": "assistant", "content": "after ttl"}],
            turn_lease_holder=holder,
        )

    db.release_session_turn_lease("shared", holder)
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        db.append_messages_batch(
            "shared",
            [{"role": "assistant", "content": "after release"}],
            turn_lease_holder=holder,
        )
    assert db.get_messages("shared") == []


def test_turn_lease_fence_walks_compression_child_to_root(tmp_path):
    """A parent-key holder still fences writes against the rotated tip."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.end_session("root", "compression")
    db.create_session("child", source="test", parent_session_id="root")

    root_holder = f"pid={os.getpid()}:turn=root"
    stale_holder = f"pid={os.getpid()}:turn=stale"
    assert db.try_acquire_session_turn_lease(
        "root", root_holder, ttl_seconds=5
    )
    assert db.append_messages_batch(
        "child",
        [{"role": "user", "content": "owner on tip"}],
        turn_lease_holder=root_holder,
    ) == 1
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        db.append_messages_batch(
            "child",
            [{"role": "assistant", "content": "impostor"}],
            turn_lease_holder=stale_holder,
        )
    db.release_session_turn_lease("child", root_holder)


def test_lost_turn_lease_flush_fails_fast_without_patience_retry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Sibling of test_a_lost_compression_lease_still_fails_fast.

    SessionTurnLeaseLostError is permanent fencing, not a live-busy signal.
    Retrying it would burn transcript write patience and still fail.
    """
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    stale_holder = f"pid={os.getpid()}:turn=stale"
    next_holder = f"pid={os.getpid()}:turn=next"
    assert db.try_acquire_session_turn_lease(
        "shared", stale_holder, ttl_seconds=5
    )
    db.release_session_turn_lease("shared", stale_holder)
    assert db.try_acquire_session_turn_lease(
        "shared", next_holder, ttl_seconds=5
    )

    sleeps = []
    original = db._sleep_before_write_retry

    def track_sleep(deadline, patience_s):
        sleeps.append(patience_s)
        return original(deadline, patience_s)

    monkeypatch.setattr(db, "_sleep_before_write_retry", track_sleep)
    monkeypatch.setattr(SessionDB, "_COMPRESSION_BUSY_WAIT_S", 5.0)

    started = time.monotonic()
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        db.append_messages_batch(
            "shared",
            [{"role": "assistant", "content": "late stale reply"}],
            turn_lease_holder=stale_holder,
        )
    assert time.monotonic() - started < 0.5
    assert sleeps == []
    assert db.get_messages("shared") == []
    db.release_session_turn_lease("shared", next_holder)


def test_turn_lease_fence_walks_continuation_that_inherited_fork_markers(tmp_path):
    """Owner flush on a rotated tip must use the parent-key lease.

    Presence-only ``_delegate_from`` / ``_branched_from`` detection would
    treat the continuation as its own conversation. The presented parent
    holder would then miss the row and fail-close a still-valid owner.
    """
    db = SessionDB(tmp_path / "state.db")
    db.create_session("original-parent", source="test")
    db.create_session(
        "delegate",
        source="delegate",
        parent_session_id="original-parent",
        model_config={"_delegate_from": "original-parent"},
    )
    db.end_session("delegate", "compression")
    db.create_session(
        "delegate-continuation",
        source="delegate",
        parent_session_id="delegate",
        model_config={"_delegate_from": "original-parent"},
    )
    db.create_session(
        "branch",
        source="test",
        parent_session_id="original-parent",
        model_config={"_branched_from": "original-parent"},
    )
    db.end_session("branch", "compression")
    db.create_session(
        "branch-continuation",
        source="test",
        parent_session_id="branch",
        model_config={"_branched_from": "original-parent"},
    )

    delegate_holder = f"pid={os.getpid()}:turn=delegate"
    assert db.try_acquire_session_turn_lease(
        "delegate", delegate_holder, ttl_seconds=5
    )
    assert db.append_messages_batch(
        "delegate-continuation",
        [{"role": "user", "content": "owner on inherited tip"}],
        turn_lease_holder=delegate_holder,
    ) == 1
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        db.append_messages_batch(
            "delegate-continuation",
            [{"role": "assistant", "content": "impostor"}],
            turn_lease_holder=f"pid={os.getpid()}:turn=impostor",
        )

    branch_holder = f"pid={os.getpid()}:turn=branch"
    assert db.try_acquire_session_turn_lease(
        "branch", branch_holder, ttl_seconds=5
    )
    assert db.append_messages_batch(
        "branch-continuation",
        [{"role": "user", "content": "branch owner on inherited tip"}],
        turn_lease_holder=branch_holder,
    ) == 1
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        db.append_messages_batch(
            "branch-continuation",
            [{"role": "assistant", "content": "branch impostor"}],
            turn_lease_holder=f"pid={os.getpid()}:turn=branch-impostor",
        )

    assert [m["content"] for m in db.get_messages("delegate-continuation")] == [
        "owner on inherited tip"
    ]
    assert [m["content"] for m in db.get_messages("branch-continuation")] == [
        "branch owner on inherited tip"
    ]
    db.release_session_turn_lease("delegate-continuation", delegate_holder)
    db.release_session_turn_lease("branch-continuation", branch_holder)
