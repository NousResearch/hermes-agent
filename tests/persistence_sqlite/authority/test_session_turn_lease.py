"""Cross-process session turn lease behavior (#84234)."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from persistence import DiskFull
from persistence.authority.coordination import acquire_turn_lease_waiting
from persistence.sqlite._holder import StoreHolder
from hermes_state_errors import SessionTurnLeaseLostError
from persistence.sqlite.conversation._store.compression import _session_turn_lease_key_on_conn
from persistence.sqlite.conversation._store.messages import append_message, append_messages_batch, get_messages
from persistence.sqlite.conversation._store.sessions import create_session, end_session
from tests.persistence_sqlite.core._lent_store import lent_store


def _handle(path):
    """A process-shaped handle on *path*: its own holder and the coordination repository served through it."""
    holder = StoreHolder(path)
    return holder, lent_store(holder).coordination


def _key(holder, session_id):
    with holder._read_ctx() as conn:
        return _session_turn_lease_key_on_conn(holder, conn, session_id)


def _fail_first_writes(monkeypatch, holder, error, count=1):
    """Make the holder's next *count* write transactions raise *error* (all of them for ``count=None``); returns
    the attempt counter."""
    attempts = {"n": 0}
    write = holder._execute_write

    def flaky(*args, **kwargs):
        attempts["n"] += 1
        if count is None or attempts["n"] <= count:
            raise error
        return write(*args, **kwargs)

    monkeypatch.setattr(holder, "_execute_write", flaky)
    return attempts


def test_turn_lease_serializes_separate_holder_instances(tmp_path):
    """A second process-shaped DB handle waits for the current turn owner."""
    path = tmp_path / "state.db"
    first_db, first = _handle(path)
    second_db, second = _handle(path)
    create_session(first_db, "shared", "test")

    first_holder = f"pid={os.getpid()}:turn=first"
    second_holder = f"pid={os.getpid()}:turn=second"
    assert first.try_acquire_turn_lease(
        session_id="shared", holder=first_holder, ttl_seconds=5
    )

    released = threading.Event()

    def release_first():
        time.sleep(0.2)  # health: allow HX013 -- the first holder keeps the lease while the second caller waits on it; its release after the pause is the contention under test
        first.release_turn_lease(session_id="shared", holder=first_holder)
        released.set()

    thread = threading.Thread(target=release_first, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        assert acquire_turn_lease_waiting(
            second,
            session_id="shared",
            holder=second_holder,
            ttl_seconds=5,
            wait_seconds=2,
            poll_interval_seconds=0.02,
        )
    finally:
        thread.join(timeout=2)

    assert released.is_set()
    assert time.monotonic() - started >= 0.15
    second.release_turn_lease(session_id="shared", holder=second_holder)
    first_db.close()
    second_db.close()


def test_turn_lease_is_scoped_to_conversation_root(store):
    """Compression descendants share one durable serialization domain."""
    db, lease = store.holder(), store.coordination
    create_session(db, "root", "test")
    end_session(db, "root", "compression")
    create_session(db, "child", "test", parent_session_id="root")

    root_holder = f"pid={os.getpid()}:turn=root"
    child_holder = f"pid={os.getpid()}:turn=child"
    assert lease.try_acquire_turn_lease(
        session_id="root", holder=root_holder, ttl_seconds=5
    )
    assert not lease.try_acquire_turn_lease(
        session_id="child", holder=child_holder, ttl_seconds=5
    )
    lease.release_turn_lease(session_id="child", holder=root_holder)


def test_turn_lease_does_not_serialize_delegate_child_with_parent(store):
    """Only compression continuation segments share a conversation lease."""
    db, lease = store.holder(), store.coordination
    create_session(db, "parent", "test")
    create_session(
        db,
        "delegate",
        "delegate",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )

    parent_holder = f"pid={os.getpid()}:turn=parent"
    delegate_holder = f"pid={os.getpid()}:turn=delegate"
    assert lease.try_acquire_turn_lease(
        session_id="parent", holder=parent_holder, ttl_seconds=5
    )
    assert lease.try_acquire_turn_lease(
        session_id="delegate", holder=delegate_holder, ttl_seconds=5
    )


def _inherited_fork_markers(db):
    """A delegate and a branch of ``original-parent``, each compressed into a continuation that kept the
    fork marker."""
    create_session(db, "original-parent", "test")
    create_session(
        db,
        "delegate",
        "delegate",
        parent_session_id="original-parent",
        model_config={"_delegate_from": "original-parent"},
    )
    end_session(db, "delegate", "compression")
    create_session(
        db,
        "delegate-continuation",
        "delegate",
        parent_session_id="delegate",
        model_config={"_delegate_from": "original-parent"},
    )
    create_session(
        db,
        "branch",
        "test",
        parent_session_id="original-parent",
        model_config={"_branched_from": "original-parent"},
    )
    end_session(db, "branch", "compression")
    create_session(
        db,
        "branch-continuation",
        "test",
        parent_session_id="branch",
        model_config={"_branched_from": "original-parent"},
    )


def test_turn_lease_walks_compression_child_that_inherited_fork_markers(store):
    """Inherited ``_delegate_from`` / ``_branched_from`` must not stop the walk.

    ``publish_compression_child`` copies ``model_config`` verbatim, so a
    delegate or branch continuation carries a marker pointing at some other
    session. Presence-only fork detection would key the child separately:
    the holder still owns the parent-key lease, but the first refresh after
    rotation looks up the child id and fail-closes with a hard interrupt.
    """
    db, lease = store.holder(), store.coordination
    _inherited_fork_markers(db)

    assert _key(db, "delegate-continuation") == "delegate"
    assert _key(db, "branch-continuation") == "branch"

    delegate_holder = f"pid={os.getpid()}:turn=delegate"
    assert lease.try_acquire_turn_lease(
        session_id="delegate", holder=delegate_holder, ttl_seconds=5
    )
    assert not lease.try_acquire_turn_lease(
        session_id="delegate-continuation",
        holder=f"pid={os.getpid()}:turn=delegate-child",
        ttl_seconds=5,
    )
    assert lease.refresh_turn_lease(
        session_id="delegate-continuation", holder=delegate_holder, ttl_seconds=5
    )

    branch_holder = f"pid={os.getpid()}:turn=branch"
    assert lease.try_acquire_turn_lease(
        session_id="branch", holder=branch_holder, ttl_seconds=5
    )
    assert not lease.try_acquire_turn_lease(
        session_id="branch-continuation",
        holder=f"pid={os.getpid()}:turn=branch-child",
        ttl_seconds=5,
    )
    assert lease.refresh_turn_lease(
        session_id="branch-continuation", holder=branch_holder, ttl_seconds=5
    )

    original_holder = f"pid={os.getpid()}:turn=original"
    assert lease.try_acquire_turn_lease(
        session_id="original-parent", holder=original_holder, ttl_seconds=5
    )
    lease.release_turn_lease(session_id="delegate-continuation", holder=delegate_holder)
    lease.release_turn_lease(session_id="branch-continuation", holder=branch_holder)
    lease.release_turn_lease(session_id="original-parent", holder=original_holder)


def test_turn_lease_write_txn_does_not_trust_fail_open_key_helper(store):
    """Acquire/refresh/release walk inside the write txn.

    The old helper swallowed get_session failures and returned the child id.
    P2 then proceeded to acquire; the write succeeded under that child key
    and the first working refresh walked to the parent and hard-interrupted.
    The repository has no key helper outside its write transactions: each verb
    walks the lineage on the transaction's own connection.
    """
    db, lease = store.holder(), store.coordination
    create_session(
        db,
        "delegate",
        "delegate",
        model_config={"_delegate_from": "original-parent"},
    )
    end_session(db, "delegate", "compression")
    create_session(
        db,
        "delegate-continuation",
        "delegate",
        parent_session_id="delegate",
        model_config={"_delegate_from": "original-parent"},
    )

    holder = f"pid={os.getpid()}:turn=delegate"
    assert lease.try_acquire_turn_lease(
        session_id="delegate", holder=holder, ttl_seconds=5
    )
    assert not lease.try_acquire_turn_lease(
        session_id="delegate-continuation",
        holder=f"pid={os.getpid()}:turn=child",
        ttl_seconds=5,
    )
    assert lease.refresh_turn_lease(
        session_id="delegate-continuation", holder=holder, ttl_seconds=5
    )
    lease.release_turn_lease(session_id="delegate-continuation", holder=holder)
    assert lease.try_acquire_turn_lease(
        session_id="delegate", holder=f"pid={os.getpid()}:turn=next", ttl_seconds=5
    )


def test_turn_lease_retries_locked_in_txn_key_walk(store, monkeypatch: pytest.MonkeyPatch):
    """A locked lineage walk must retry, not INSERT under the child id."""
    from persistence.sqlite.authority import coordination
    db, lease = store.holder(), store.coordination
    create_session(
        db,
        "delegate",
        "delegate",
        model_config={"_delegate_from": "original-parent"},
    )
    end_session(db, "delegate", "compression")
    create_session(
        db,
        "delegate-continuation",
        "delegate",
        parent_session_id="delegate",
        model_config={"_delegate_from": "original-parent"},
    )

    attempts = {"n": 0}
    original = coordination._session_turn_lease_key_on_conn

    def flaky_walk(holder_, conn, session_id):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return original(holder_, conn, session_id)

    monkeypatch.setattr(coordination, "_session_turn_lease_key_on_conn", flaky_walk)
    holder = f"pid={os.getpid()}:turn=delegate"
    assert lease.try_acquire_turn_lease(
        session_id="delegate-continuation", holder=holder, ttl_seconds=5
    )
    assert attempts["n"] >= 2
    monkeypatch.setattr(coordination, "_session_turn_lease_key_on_conn", original)
    assert not lease.try_acquire_turn_lease(
        session_id="delegate", holder=f"pid={os.getpid()}:turn=other", ttl_seconds=5
    )
    assert lease.refresh_turn_lease(session_id="delegate", holder=holder, ttl_seconds=5)
    lease.release_turn_lease(session_id="delegate-continuation", holder=holder)


def test_turn_lease_refresh_and_release_are_owner_fenced(store):
    db, lease = store.holder(), store.coordination
    create_session(db, "shared", "test")

    current_holder = f"pid={os.getpid()}:turn=current"
    stale_holder = f"pid={os.getpid()}:turn=stale"
    next_holder = f"pid={os.getpid()}:turn=next"
    assert lease.try_acquire_turn_lease(
        session_id="shared", holder=current_holder, ttl_seconds=5
    )
    assert not lease.refresh_turn_lease(
        session_id="shared", holder=stale_holder, ttl_seconds=5
    )
    lease.release_turn_lease(session_id="shared", holder=stale_holder)
    assert not lease.try_acquire_turn_lease(
        session_id="shared", holder=next_holder, ttl_seconds=5
    )

    assert lease.refresh_turn_lease(
        session_id="shared", holder=current_holder, ttl_seconds=5
    )
    lease.release_turn_lease(session_id="shared", holder=current_holder)
    assert lease.try_acquire_turn_lease(
        session_id="shared", holder=next_holder, ttl_seconds=5
    )


def test_expired_turn_lease_is_reclaimed(store):
    db, lease = store.holder(), store.coordination
    create_session(db, "shared", "test")
    assert lease.try_acquire_turn_lease(
        session_id="shared", holder="legacy-holder", ttl_seconds=0.05
    )

    time.sleep(0.15)  # health: allow HX013 -- the short lease must expire on the wall clock before the reclaim

    assert lease.try_acquire_turn_lease(
        session_id="shared", holder="pid=202:turn=reclaimer", ttl_seconds=5
    )


def test_acquire_turn_lease_notifies_wait_callback(tmp_path):
    """Waiters get a progress callback while another holder owns the lease."""
    path = tmp_path / "state.db"
    first_db, first = _handle(path)
    second_db, second = _handle(path)
    create_session(first_db, "shared", "test")

    first_holder = f"pid={os.getpid()}:turn=first"
    second_holder = f"pid={os.getpid()}:turn=second"
    assert first.try_acquire_turn_lease(
        session_id="shared", holder=first_holder, ttl_seconds=5
    )

    notices = []

    def release_first():
        time.sleep(0.12)  # health: allow HX013 -- the first holder keeps the lease while the second caller waits on it; its release after the pause is the contention under test
        first.release_turn_lease(session_id="shared", holder=first_holder)

    thread = threading.Thread(target=release_first, daemon=True)
    thread.start()
    try:
        assert acquire_turn_lease_waiting(
            second,
            session_id="shared",
            holder=second_holder,
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
    second.release_turn_lease(session_id="shared", holder=second_holder)
    first_db.close()
    second_db.close()


def test_acquire_turn_lease_honors_should_abort(tmp_path):
    """Waiters stop immediately when should_abort() returns True."""
    path = tmp_path / "state.db"
    first_db, first = _handle(path)
    second_db, second = _handle(path)
    create_session(first_db, "shared", "test")

    first_holder = f"pid={os.getpid()}:turn=first"
    second_holder = f"pid={os.getpid()}:turn=second"
    assert first.try_acquire_turn_lease(
        session_id="shared", holder=first_holder, ttl_seconds=60
    )

    abort_checks = {"count": 0}

    def should_abort():
        abort_checks["count"] += 1
        return True

    started = time.monotonic()
    assert not acquire_turn_lease_waiting(
        second,
        session_id="shared",
        holder=second_holder,
        wait_seconds=30,
        poll_interval_seconds=0.05,
        should_abort=should_abort,
    )
    assert time.monotonic() - started < 1.0
    assert abort_checks["count"] >= 1
    first.release_turn_lease(session_id="shared", holder=first_holder)
    first_db.close()
    second_db.close()


def test_acquire_turn_lease_retries_sqlite_lock(store, monkeypatch):
    """Write-lock exhaustion is retried, not a hard abort of the wait, and is not a lease wait:
    it reports on_contended instead of on_wait and does not sit out a poll interval on top of
    the patience it already spent."""
    db, lease = store.holder(), store.coordination
    create_session(db, "shared", "test")
    holder = f"pid={os.getpid()}:turn=waiter"
    attempts = _fail_first_writes(monkeypatch, db, sqlite3.OperationalError(
        "database is locked (another Hermes process held the "
        "state.db write lock for over 20s)"
    ))
    notices, contended = [], []
    started = time.monotonic()
    assert acquire_turn_lease_waiting(
        lease,
        session_id="shared",
        holder=holder,
        wait_seconds=10,
        poll_interval_seconds=5,
        on_wait=notices.append,
        on_contended=lambda: contended.append(True),
        acquire_patience_s=0.05,
    )
    assert time.monotonic() - started < 2.0
    assert (notices, contended) == ([], [True])
    assert attempts["n"] >= 2
    lease.release_turn_lease(session_id="shared", holder=holder)


def test_acquire_turn_lease_locked_for_the_whole_wait_times_out(store, monkeypatch):
    db, lease = store.holder(), store.coordination
    create_session(db, "shared", "test")
    _fail_first_writes(monkeypatch, db, sqlite3.OperationalError("database is locked"), count=None)
    notices = []
    assert not acquire_turn_lease_waiting(
        lease, session_id="shared", holder=f"pid={os.getpid()}:turn=waiter", wait_seconds=0.3,
        poll_interval_seconds=5, on_wait=notices.append,
    )
    assert notices == []


def test_acquire_turn_lease_reraises_non_lock_sqlite_error(store, monkeypatch):
    db, lease = store.holder(), store.coordination
    create_session(db, "shared", "test")
    _fail_first_writes(monkeypatch, db, sqlite3.OperationalError("database or disk is full"), count=None)
    with pytest.raises(DiskFull) as raised:
        acquire_turn_lease_waiting(
            lease,
            session_id="shared",
            holder=f"pid={os.getpid()}:turn=waiter",
            wait_seconds=1,
            poll_interval_seconds=0.02,
        )
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
    assert "disk is full" in str(raised.value.__cause__)


def test_non_expired_turn_lease_from_dead_pid_is_reclaimed(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holder whose structured pid= no longer exists can be reclaimed early.

    Stamped with this process's own PID namespace: same-namespace kernel proof
    is still proof (a foreign or unstamped holder would defer to TTL instead)."""
    from persistence.sqlite.conversation._store import core
    db, lease = store.holder(), store.coordination
    create_session(db, "shared", "test")

    from hermes_state_pidns import holder_namespace_token
    dead_holder = f"pid=424242{holder_namespace_token()}:turn=dead:platform=test"
    assert lease.try_acquire_turn_lease(
        session_id="shared", holder=dead_holder, ttl_seconds=300
    ) is True

    probed: list[int] = []

    def pid_exists(pid: int) -> bool:
        probed.append(pid)
        return False

    monkeypatch.setattr(
        core, "psutil", SimpleNamespace(pid_exists=pid_exists)
    )

    fresh_holder = "pid=525252:turn=fresh:platform=test"
    assert lease.try_acquire_turn_lease(
        session_id="shared", holder=fresh_holder, ttl_seconds=300
    ) is True
    assert probed == [424242]


def test_turn_lease_fences_stale_transcript_flush_after_reclaim(store):
    """A lost holder cannot persist after B has taken the conversation.

    Refresh-loss interrupt is cooperative; the lease itself must reject the
    late append inside the same SQLite write transaction.
    """
    db, lease = store.holder(), store.coordination
    create_session(db, "shared", "test")
    stale_holder = f"pid={os.getpid()}:turn=stale"
    next_holder = f"pid={os.getpid()}:turn=next"

    assert lease.try_acquire_turn_lease(
        session_id="shared", holder=stale_holder, ttl_seconds=5
    )
    assert append_messages_batch(
        db,
        "shared",
        [{"role": "user", "content": "stale-owned"}],
        turn_lease_holder=stale_holder,
    ) == 1

    lease.release_turn_lease(session_id="shared", holder=stale_holder)
    assert lease.try_acquire_turn_lease(
        session_id="shared", holder=next_holder, ttl_seconds=5
    )

    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        append_messages_batch(
            db,
            "shared",
            [{"role": "assistant", "content": "late stale reply"}],
            turn_lease_holder=stale_holder,
        )
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        append_message(
            db,
            "shared",
            "assistant",
            "late stale single-row",
            turn_lease_holder=stale_holder,
        )

    assert append_messages_batch(
        db,
        "shared",
        [{"role": "assistant", "content": "next reply"}],
        turn_lease_holder=next_holder,
    ) == 1
    assert [m["content"] for m in get_messages(db, "shared")] == [
        "stale-owned",
        "next reply",
    ]
    lease.release_turn_lease(session_id="shared", holder=next_holder)


def test_turn_lease_revives_expired_row_still_owned_by_writer(store):
    db, lease = store.holder(), store.coordination
    create_session(db, "shared", "test")
    holder = f"pid={os.getpid()}:turn=owner"

    assert lease.try_acquire_turn_lease(session_id="shared", holder=holder, ttl_seconds=0.05)
    time.sleep(0.12)  # health: allow HX013 -- the short lease must expire on the wall clock before the writer's append
    assert append_messages_batch(
        db,
        "shared",
        [{"role": "assistant", "content": "after ttl"}],
        turn_lease_holder=holder,
        turn_lease_ttl_seconds=0.2,
    ) == 1
    assert not lease.try_acquire_turn_lease(
        session_id="shared", holder=f"pid={os.getpid()}:turn=contender", ttl_seconds=5
    )


def test_turn_lease_acquisition_sweeps_long_expired_rows_of_other_conversations(store):
    """A holder that died without releasing leaves its row behind; the next acquisition anywhere
    removes rows expired past the grace, and keeps recent expiries their live owner may renew."""
    db, lease = store.holder(), store.coordination
    for sid in ("abandoned", "recent", "within-grace", "fresh"):
        create_session(db, sid, "test")
    raw = sqlite3.connect(store.store_path)
    with raw:
        raw.executemany(
            "INSERT INTO session_turn_leases (conversation_id, holder, acquired_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            [("abandoned", "pid=1:turn=crashed", time.time() - 86400 - 120, time.time() - 86400 - 60),
             ("recent", f"pid={os.getpid()}:turn=starved", time.time() - 60, time.time() - 10),
             ("within-grace", "pid=1:turn=suspended", time.time() - 86400, time.time() - 86400 + 60)])

    assert lease.try_acquire_turn_lease(session_id="fresh", holder=f"pid={os.getpid()}:turn=next", ttl_seconds=5)

    rows = dict(raw.execute("SELECT conversation_id, holder FROM session_turn_leases").fetchall())
    raw.close()
    assert rows == {"recent": f"pid={os.getpid()}:turn=starved",
                    "within-grace": "pid=1:turn=suspended",
                    "fresh": f"pid={os.getpid()}:turn=next"}
    assert append_messages_batch(
        db, "recent", [{"role": "assistant", "content": "after a starved refresher"}],
        turn_lease_holder=f"pid={os.getpid()}:turn=starved", turn_lease_ttl_seconds=5,
    ) == 1


def test_turn_lease_fences_flush_when_row_is_absent(store):
    db = store.holder()
    create_session(db, "shared", "test")
    holder = f"pid={os.getpid()}:turn=owner"

    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        append_messages_batch(
            db,
            "shared",
            [{"role": "assistant", "content": "after release"}],
            turn_lease_holder=holder,
        )
    assert get_messages(db, "shared") == []


def test_turn_lease_fence_walks_compression_child_to_root(store):
    """A parent-key holder still fences writes against the rotated tip."""
    db, lease = store.holder(), store.coordination
    create_session(db, "root", "test")
    end_session(db, "root", "compression")
    create_session(db, "child", "test", parent_session_id="root")

    root_holder = f"pid={os.getpid()}:turn=root"
    stale_holder = f"pid={os.getpid()}:turn=stale"
    assert lease.try_acquire_turn_lease(
        session_id="root", holder=root_holder, ttl_seconds=5
    )
    assert append_messages_batch(
        db,
        "child",
        [{"role": "user", "content": "owner on tip"}],
        turn_lease_holder=root_holder,
    ) == 1
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        append_messages_batch(
            db,
            "child",
            [{"role": "assistant", "content": "impostor"}],
            turn_lease_holder=stale_holder,
        )
    lease.release_turn_lease(session_id="child", holder=root_holder)


def test_lost_turn_lease_flush_fails_fast_without_patience_retry(
    store, monkeypatch: pytest.MonkeyPatch
):
    """Sibling of test_a_lost_compression_lease_still_fails_fast.

    SessionTurnLeaseLostError is permanent fencing, not a live-busy signal.
    Retrying it would burn transcript write patience and still fail.
    """
    db, lease = store.holder(), store.coordination
    create_session(db, "shared", "test")
    stale_holder = f"pid={os.getpid()}:turn=stale"
    next_holder = f"pid={os.getpid()}:turn=next"
    assert lease.try_acquire_turn_lease(
        session_id="shared", holder=stale_holder, ttl_seconds=5
    )
    lease.release_turn_lease(session_id="shared", holder=stale_holder)
    assert lease.try_acquire_turn_lease(
        session_id="shared", holder=next_holder, ttl_seconds=5
    )

    sleeps = []
    original = db._sleep_before_write_retry

    def track_sleep(deadline, patience_s):
        sleeps.append(patience_s)
        return original(deadline, patience_s)

    monkeypatch.setattr(db, "_sleep_before_write_retry", track_sleep)
    monkeypatch.setattr(StoreHolder, "_COMPRESSION_BUSY_WAIT_S", 5.0)

    started = time.monotonic()
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        append_messages_batch(
            db,
            "shared",
            [{"role": "assistant", "content": "late stale reply"}],
            turn_lease_holder=stale_holder,
        )
    assert time.monotonic() - started < 0.5
    assert sleeps == []
    assert get_messages(db, "shared") == []
    lease.release_turn_lease(session_id="shared", holder=next_holder)


def test_turn_lease_fence_walks_continuation_that_inherited_fork_markers(store):
    """Owner flush on a rotated tip must use the parent-key lease.

    Presence-only ``_delegate_from`` / ``_branched_from`` detection would
    treat the continuation as its own conversation. The presented parent
    holder would then miss the row and fail-close a still-valid owner.
    """
    db, lease = store.holder(), store.coordination
    _inherited_fork_markers(db)

    delegate_holder = f"pid={os.getpid()}:turn=delegate"
    assert lease.try_acquire_turn_lease(
        session_id="delegate", holder=delegate_holder, ttl_seconds=5
    )
    assert append_messages_batch(
        db,
        "delegate-continuation",
        [{"role": "user", "content": "owner on inherited tip"}],
        turn_lease_holder=delegate_holder,
    ) == 1
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        append_messages_batch(
            db,
            "delegate-continuation",
            [{"role": "assistant", "content": "impostor"}],
            turn_lease_holder=f"pid={os.getpid()}:turn=impostor",
        )

    branch_holder = f"pid={os.getpid()}:turn=branch"
    assert lease.try_acquire_turn_lease(
        session_id="branch", holder=branch_holder, ttl_seconds=5
    )
    assert append_messages_batch(
        db,
        "branch-continuation",
        [{"role": "user", "content": "branch owner on inherited tip"}],
        turn_lease_holder=branch_holder,
    ) == 1
    with pytest.raises(SessionTurnLeaseLostError, match="turn lease lost"):
        append_messages_batch(
            db,
            "branch-continuation",
            [{"role": "assistant", "content": "branch impostor"}],
            turn_lease_holder=f"pid={os.getpid()}:turn=branch-impostor",
        )

    assert [m["content"] for m in get_messages(db, "delegate-continuation")] == [
        "owner on inherited tip"
    ]
    assert [m["content"] for m in get_messages(db, "branch-continuation")] == [
        "branch owner on inherited tip"
    ]
    lease.release_turn_lease(session_id="delegate-continuation", holder=delegate_holder)
    lease.release_turn_lease(session_id="branch-continuation", holder=branch_holder)
