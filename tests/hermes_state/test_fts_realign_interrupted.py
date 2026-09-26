"""The ``messages_fts`` realign survives being cut off in its rebuild.

Realigning a v1/v2 index onto ``messages_fts_src`` drops the old index, creates the aligned
one and fills it with FTS5 ``'rebuild'``. On the autocommit writer the DDL used to commit on
its own, so a process killed during the rebuild (minutes on a multi-GB store) left the aligned
shape over an empty index: the shape probe that schedules the realign no longer fired, and
search missed every older message from then on.

Behaviour contracts on the index a reopen serves, not on how the migration is written: every
row carrying the token is found by the index (MATCH count == LIKE count), and the layout marker
is current.
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_SQL, FTS_STORAGE_VERSION

NEEDLE = "zephyrquill"
NEEDLE_ROWS = 120
# Progress callbacks (one per VM step) the rebuild runs before it is cut off: it has started, not finished.
_STEPS_BEFORE_ABORT = 50


def _store_with_needles(db_path):
    db = SessionDB(db_path=db_path)
    if not db._fts_enabled:
        db.close()
        pytest.skip("SQLite FTS5 unavailable")
    db.create_session("needles", source="cli")
    for i in range(NEEDLE_ROWS):
        db.append_message("needles", role="user", content=f"{NEEDLE} user row {i}")
    return db


def _layout2_store_with_needles(db_path):
    """A layout-2 store: ``messages_fts`` reads raw ``messages`` over a filled index, and the
    trigram is already current, so the open stamps the new layout BEFORE it realigns. The marker
    then cannot tell a realign that died from one that finished; only the index can."""
    db = _store_with_needles(db_path)
    try:
        db._conn.execute("DROP TABLE messages_fts")
        db._conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, tool_name, tool_calls, content='messages', content_rowid='id')"
        )
        db._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        db.set_meta("fts_storage_version", "2")
        assert _needle_counts(db) == (NEEDLE_ROWS, NEEDLE_ROWS)
    finally:
        db.close()


def _needle_counts(db):
    """(rows the index finds, rows that hold the token)."""
    indexed = db._conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", (NEEDLE,)
    ).fetchone()[0]
    stored = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE content LIKE ?", (f"%{NEEDLE}%",)
    ).fetchone()[0]
    return indexed, stored


def _connect_cutting_off_the_realign_rebuild(real_connect):
    """``sqlite3.connect`` whose connections abort the base index's 'rebuild' once it is under
    way (SQLITE_INTERRUPT from the progress handler): an in-process stand-in for the kill."""

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        budget = []

        def trace(sql):
            if not budget and "INTO messages_fts(messages_fts) VALUES('rebuild')" in sql:
                budget.append(_STEPS_BEFORE_ABORT)

        def progress():
            if not budget or budget[0] <= 0:
                return 0
            budget[0] -= 1
            return budget[0] == 0

        conn.set_trace_callback(trace)
        conn.set_progress_handler(progress, 1)
        return conn

    return connect


def test_a_realign_cut_off_in_its_rebuild_completes_on_the_next_open(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _layout2_store_with_needles(db_path)
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", _connect_cutting_off_the_realign_rebuild(sqlite3.connect))
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            SessionDB(db_path=db_path)

    db = SessionDB(db_path=db_path)
    try:
        assert _needle_counts(db) == (NEEDLE_ROWS, NEEDLE_ROWS)
        assert db.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
    finally:
        db.close()


@pytest.mark.parametrize("cut_off", ["realign", "optimize-storage-demote"])
def test_an_aligned_index_a_cut_off_realign_left_unfilled_is_rebuilt_on_open(tmp_path, cut_off):
    """A store an earlier build already left behind: the aligned shape over an index holding
    only what was written since the cut-off. After a realign the layout marker is current and
    the open must rebuild. After an optimize-storage demote (trash tables, no markers) the store
    is optimize-storage's to resume, so the open must not stamp it current over the trash."""
    db_path = tmp_path / "state.db"
    db = _store_with_needles(db_path)
    try:
        # What the autocommit realign had committed when its rebuild died.
        for trigger in ("messages_fts_insert", "messages_fts_delete", "messages_fts_update"):
            db._conn.execute(f"DROP TRIGGER {trigger}")
        db._conn.execute("DROP TABLE messages_fts")
        db._conn.executescript(FTS_SQL)
        if cut_off == "realign":
            db.set_meta("fts_storage_version", str(FTS_STORAGE_VERSION))
        else:
            db._conn.execute("CREATE TABLE fts_v22_trash_messages_fts_data(id INTEGER PRIMARY KEY, block BLOB)")
            db._conn.execute("DELETE FROM state_meta WHERE key = 'fts_storage_version'")
        # One message written since makes the index non-empty, so emptiness no longer shows the loss.
        db.create_session("after-the-cut-off", source="cli")
        db.append_message("after-the-cut-off", role="user", content=f"{NEEDLE} written after the cut-off")
        assert _needle_counts(db) == (1, NEEDLE_ROWS + 1)
    finally:
        db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        indexed, stored = _needle_counts(reopened)
        stamped = reopened.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        trash_left = reopened._has_fts_trash(reopened._conn)
    finally:
        reopened.close()
    assert not (stamped and trash_left)
    if cut_off == "realign":
        assert (indexed, stamped) == (stored, True)
