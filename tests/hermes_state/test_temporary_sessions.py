"""Temporary chats in SessionDB: never listed or searched, deleted only when flagged, swept when orphaned."""

import os
import socket

import pytest

from hermes_state import SessionDB
from hermes_state_temporary import TEMPORARY_ORPHAN_MAX_IDLE_SECONDS, temporary_owner_record


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _chat(db, sid, text, *, temporary=False, owner=None):
    db.create_session(sid, source="desktop")
    db.append_messages_batch(sid, [{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}])
    if temporary:
        assert db.mark_session_temporary(sid, owner=owner) is True


def _dead_owner():
    # Same host + namespace as this process, a pid that cannot be live.
    return {**temporary_owner_record(), "pid": 2**22 + 12345, "create_time": None}


def test_temporary_chat_is_hidden_from_every_listing_and_search(db):
    _chat(db, "saved", "zebra crossing notes")
    _chat(db, "temp", "zebra secret plans", temporary=True)

    assert db.is_session_temporary("temp") and not db.is_session_temporary("saved")
    assert db.get_session("temp")["hidden"] == 1
    for kwargs in ({}, {"include_hidden": True}, {"include_archived": True, "include_hidden": True}):
        assert "temp" not in {s["id"] for s in db.list_sessions_rich(**kwargs)}, kwargs
    assert "saved" in {s["id"] for s in db.list_sessions_rich()}
    assert {hit["session_id"] for hit in db.search_messages("zebra")} == {"saved"}
    assert "temp" not in {s["id"] for s in db.list_recent_sessions_bounded(limit=10)}
    # `hermes -c` / `--resume latest` pick from search_sessions: never a temporary chat.
    assert [s["id"] for s in db.search_sessions(source="desktop")] == ["saved"]


def test_delete_temporary_session_never_deletes_a_saved_chat(db):
    _chat(db, "saved", "keep me")
    _chat(db, "temp", "drop me", temporary=True)

    assert db.delete_temporary_session("saved") is False
    assert db.get_session("saved") is not None
    assert db.delete_temporary_session("temp") is True
    assert db.get_session("temp") is None
    assert db.get_messages("temp") == []


def test_delete_temporary_session_takes_the_compression_chain(db):
    _chat(db, "root", "first half", temporary=True)
    db.end_session("root", "compression")
    db.create_session("tip", source="desktop", parent_session_id="root",
                      model_config={"temporary": temporary_owner_record()})
    db.append_messages_batch("tip", [{"role": "user", "content": "second half"}])

    assert db.delete_temporary_session("tip") is True
    assert db.get_session("root") is None and db.get_session("tip") is None


def test_sweep_deletes_only_orphans_of_dead_owners(db):
    _chat(db, "mine", "live chat", temporary=True)  # owned by this (live) process
    _chat(db, "orphan", "crashed chat", temporary=True, owner=_dead_owner())
    _chat(db, "kept", "crashed but in use", temporary=True, owner=_dead_owner())
    _chat(db, "saved", "normal chat")

    assert db.sweep_orphaned_temporary_sessions(keep_ids={"kept"}) == ["orphan"]
    assert db.get_session("orphan") is None
    for sid in ("mine", "kept", "saved"):
        assert db.get_session(sid) is not None, sid


def test_sweep_falls_back_to_idle_age_when_owner_is_unverifiable(db):
    foreign = {**temporary_owner_record(), "host": f"not-{socket.gethostname()}"}
    _chat(db, "foreign", "other machine", temporary=True, owner=foreign)

    assert db.sweep_orphaned_temporary_sessions() == []
    later = db.get_session("foreign")["started_at"] + TEMPORARY_ORPHAN_MAX_IDLE_SECONDS + 3600
    assert db.sweep_orphaned_temporary_sessions(now=later) == ["foreign"]


def test_marking_records_this_process_as_owner(db):
    _chat(db, "temp", "hi", temporary=True)
    owner = db.get_session_temporary_owner("temp")
    assert owner["pid"] == os.getpid() and owner["host"] == socket.gethostname()


def _bytes_on_disk(path):
    return b"".join(p.read_bytes() for p in (path, path.with_name(path.name + "-wal")) if p.exists())


@pytest.mark.parametrize("merge_first", [False, True])
@pytest.mark.parametrize("end", ["delete", "sweep"])
def test_ended_temporary_chat_leaves_no_bytes_in_the_database_files(tmp_path, end, merge_first):
    # A deleted row normally survives in the WAL, in freed pages and in FTS5 segments until a VACUUM;
    # a temporary chat must be unrecoverable from state.db once it ends, like a chat never written.
    path = tmp_path / "state.db"
    canary = "zqxcanary" + "7b7a1ec7b22f"
    db = SessionDB(path)
    try:
        _chat(db, "saved", "a saved chat that stays")
        owner = _dead_owner() if end == "sweep" else None
        _chat(db, "temp", f"secret {canary} " + "padding " * 900, temporary=True, owner=owner)
        db.append_messages_batch("temp", [{"role": "assistant", "content": f"echo {canary.upper()}"}])
        assert canary.encode() in _bytes_on_disk(path).lower()
        if merge_first:
            # Regression: a routine FTS merge between the chat and its deletion freed the old
            # segment page (still holding the chat's terms) to the freelist without zeroing it.
            for i in range(30):
                db.append_messages_batch("saved", [{"role": "user", "content": f"more saved words {i}"}])
            def _optimize(conn):
                for name in ("messages_fts", "messages_fts_trigram"):
                    conn.execute(f"INSERT INTO {name}({name}) VALUES ('optimize')")
            db._execute_write(_optimize)

        if end == "delete":
            assert db.delete_temporary_session("temp") is True
        else:
            assert db.sweep_orphaned_temporary_sessions() == ["temp"]
        on_disk = _bytes_on_disk(path).lower()
        assert canary.encode() not in on_disk
        assert b"a saved chat that stays" in on_disk
    finally:
        db.close()
    assert canary.encode() not in _bytes_on_disk(path).lower()
