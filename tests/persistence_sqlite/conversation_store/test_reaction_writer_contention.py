"""Collecting unseen reactions must not put an empty history scan under the writer."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from threading import Barrier, Event

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import append_message, set_message_reaction, take_unseen_reactions
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture
def sessions(tmp_path):
    db = StoreHolder(db_path=tmp_path / "state.db")
    try:
        first = create_session(db, "reaction-test", "test")
        other = create_session(db, "other-session", "test")
        for _ in range(11):
            append_message(db, first, "assistant", "an earlier answer")
        target = append_message(db, first, "assistant", "the latest answer")
        yield db, first, other, target
    finally:
        db.close()


@pytest.mark.parametrize("with_seen_reaction", [False, True])
def test_no_unseen_reactions_do_not_enter_write_transaction(sessions, monkeypatch, with_seen_reaction):
    db, first, _, target = sessions
    if with_seen_reaction:
        set_message_reaction(db, first, target, "👍")
        assert len(take_unseen_reactions(db, first)) == 1
    writes = []
    write = db._execute_write

    def record_write(fn, **kwargs):
        writes.append(fn)
        return write(fn, **kwargs)

    monkeypatch.setattr(db, "_execute_write", record_write)
    assert take_unseen_reactions(db, first) == []
    assert writes == []


def test_empty_reaction_scan_can_overlap_another_session_commit(sessions, monkeypatch):
    db, first, other, _ = sessions
    scanning, release = Event(), Event()

    def pause_scan(sql):
        if sql.startswith("SELECT id,") and "FROM messages" in sql and "json_extract" in sql:
            scanning.set()
            release.wait(8)

    # Observe the real scan whichever connection it chooses, not a mock storage method.
    db._conn.set_trace_callback(pause_scan)
    read_ctx = db._read_ctx

    @contextmanager
    def observe_read(**kwargs):
        with read_ctx(**kwargs) as conn:
            if conn is not None and conn is not db._conn:
                conn.set_trace_callback(pause_scan)
                try:
                    yield conn
                finally:
                    conn.set_trace_callback(None)
            else:
                yield conn

    monkeypatch.setattr(db, "_read_ctx", observe_read)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reader = pool.submit(take_unseen_reactions, db, first)
            try:
                assert scanning.wait(4), "did not reach the reaction scan"
                writer = pool.submit(append_message, db, other, "user", "a concurrent write")
                row_id = writer.result(timeout=3)
                with closing(sqlite3.connect(f"file:{db.db_path}?mode=ro", uri=True)) as observer:
                    assert observer.execute("SELECT content FROM messages WHERE id = ?", (row_id,)).fetchone() == (
                        "a concurrent write",
                    )
                assert not reader.done(), "the scan must still be held at the barrier"
            finally:
                release.set()
            assert reader.result(timeout=4) == []
    finally:
        db._conn.set_trace_callback(None)


def test_cleared_reaction_is_rechecked_at_the_claim(sessions, monkeypatch):
    db, first, _, target = sessions
    set_message_reaction(db, first, target, "👍")
    entering_write, release = Event(), Event()
    write = db._execute_write

    def wait_to_claim(fn, **kwargs):
        entering_write.set()
        assert release.wait(8)
        return write(fn, **kwargs)

    monkeypatch.setattr(db, "_execute_write", wait_to_claim)
    peer = StoreHolder(db_path=db.db_path)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            reader = pool.submit(take_unseen_reactions, db, first)
            try:
                assert entering_write.wait(4)
                assert set_message_reaction(peer, first, target, None) == []
            finally:
                release.set()
            assert reader.result(timeout=4) == []
    finally:
        peer.close()


def test_missing_reader_uses_existing_atomic_claim(sessions, monkeypatch):
    db, first, _, target = sessions
    set_message_reaction(db, first, target, "👍")
    monkeypatch.setattr(db, "_checkout_read_conn", lambda: None)
    assert [r["row_id"] for r in take_unseen_reactions(db, first)] == [target]
    assert take_unseen_reactions(db, first) == []


def test_concurrent_collectors_return_the_reaction_only_once(sessions, monkeypatch):
    db, first, _, target = sessions
    set_message_reaction(db, first, target, "👍")
    barrier = Barrier(2, timeout=5)
    write = db._execute_write

    def collect_together(fn, **kwargs):
        barrier.wait()
        return write(fn, **kwargs)

    monkeypatch.setattr(db, "_execute_write", collect_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = [pool.submit(take_unseen_reactions, db, first) for _ in range(2)]
        assert [item["row_id"] for claim in claims for item in claim.result(timeout=8)] == [target]


def test_old_transcript_snapshot_does_not_hide_newly_committed_reaction(sessions):
    db, first, _, target = sessions
    peer = StoreHolder(db_path=db.db_path)
    try:
        with db.transcript_read_snapshot() as snapshot:
            assert snapshot.pinned
            db._read_one("SELECT id FROM messages WHERE id = ?", (target,))
            set_message_reaction(peer, first, target, "👍")
            assert [r["row_id"] for r in take_unseen_reactions(db, first)] == [target]
        assert take_unseen_reactions(db, first) == []
    finally:
        peer.close()
