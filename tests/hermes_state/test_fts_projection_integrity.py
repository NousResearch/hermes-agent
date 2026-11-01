"""Regression for the ``messages_fts`` external-content drift (issue #114169).

The base word index used to read its external content from the raw ``messages``
table while the triggers indexed only a bounded prefix of every long tool row
and the boundary itself moved with a ``state_meta`` high-water marker. FTS5's
strict ``rank=1`` integrity check re-reads the content source and compares it
with the stored token stream, so that construction could not stay consistent: a
long tool row was enough to make the check report ``fts5: checksum mismatch``,
and the delete/update commands sent full content for a truncated row, leaving
index tokens behind.

These are behaviour contracts on the projection, not on its shape: the strict
probe must survive arbitrary churn. Both an index left reading raw ``messages``
and a v3 index whose triggers were downgraded by an old opener must realign on
open. Replacing triggers alone cannot remove postings already left by a wrong
DELETE; the existing admitted rebuild must run before subsequent mutations.
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_STORAGE_VERSION, FTS_TOOL_CONTENT_PREFIX_CHARS

LONG_TOOL_ROW = "prefix " + ("padding " * ((FTS_TOOL_CONTENT_PREFIX_CHARS // 8) + 64)) + " tailtoken"


def _strict_integrity_probe(db) -> None:
    """FTS5's strict check: re-reads the content source and compares it with the
    stored token stream. Raises ``sqlite3.DatabaseError`` when they disagree."""
    db._conn.execute(
        "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
    )


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    if not session_db._fts_enabled:
        session_db.close()
        pytest.skip("SQLite FTS5 unavailable")
    session_db.create_session("session", source="cli")
    try:
        yield session_db
    finally:
        session_db.close()


def test_strict_integrity_probe_survives_tool_row_churn(db):
    """Every write path must leave the index readable by FTS5's own checker.

    Pre-fix, the long-row assertion raises ``DatabaseError: fts5: checksum
    mismatch for table "messages_fts"``: the row was indexed as a prefix while
    ``content='messages'`` kept the full body, so the checker disagreed with the
    stored tokens. The UPDATE and DELETE legs are the triggers whose content
    command used to be re-evaluated against a moving mark.
    """
    _strict_integrity_probe(db)

    long_id = db.append_message(
        "session", role="tool", content=LONG_TOOL_ROW, tool_name="terminal"
    )
    _strict_integrity_probe(db)

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?", ("short body", long_id)
        )
    )
    _strict_integrity_probe(db)

    db.append_message("session", role="user", content="short user row")
    _strict_integrity_probe(db)

    db._execute_write(
        lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (long_id,))
    )
    _strict_integrity_probe(db)


def test_index_reading_raw_messages_realigns_once_on_open(tmp_path):
    """A store whose index still reads raw ``messages`` (the shipped shape for
    versions before the aligned projection) realigns on the next open, records
    the new storage version, and is consistent from then on."""
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    if not first._fts_enabled:
        first.close()
        pytest.skip("SQLite FTS5 unavailable")
    first.create_session("session", source="cli")
    row_id = first.append_message(
        "session", role="tool", content=LONG_TOOL_ROW, tool_name="terminal"
    )

    # Rewind to the pre-fix on-disk shape: the base index reads raw `messages`,
    # and its tokens were written from the bounded projection the triggers used
    # for tool rows. That mixture IS the drift: the checker re-reads the full
    # body and disagrees with the stored prefix.
    first._conn.execute("DROP TABLE messages_fts")
    first._conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5("
        "content, tool_name, tool_calls, content='messages', content_rowid='id')"
    )
    first._conn.execute(
        "INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) VALUES(?, ?, ?, ?)",
        (row_id, LONG_TOOL_ROW[:FTS_TOOL_CONTENT_PREFIX_CHARS], "terminal", None),
    )
    first._conn.execute(
        "INSERT INTO state_meta(key, value) VALUES('fts_storage_version', '2') "
        "ON CONFLICT(key) DO UPDATE SET value = '2'"
    )
    first._conn.execute(
        "INSERT OR REPLACE INTO state_meta(key, value) VALUES('fts_tool_full_content_high_water', ?)",
        (str(row_id),),
    )
    with pytest.raises(sqlite3.DatabaseError):
        _strict_integrity_probe(first)
    first.close()

    migrated = SessionDB(db_path=path)
    try:
        assert migrated.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        assert migrated.get_meta("fts_tool_full_content_high_water") is None
        _strict_integrity_probe(migrated)
        index_sql = migrated._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'messages_fts'"
        ).fetchone()[0]
        assert "messages_fts_src" in index_sql
        # The realigned index answers the prefix; the tail comes from the
        # explicit tool path, which reads stored content rather than the index.
        assert migrated.search_messages("prefix", role_filter=["tool"])
        assert [
            row["id"]
            for row in migrated.search_messages("tailtoken", role_filter=["tool"])
        ] == [row_id]
    finally:
        migrated.close()


@pytest.mark.parametrize("operation", ["delete_session", "prune_sessions", "replace_messages", "archive_and_compact"])
@pytest.mark.parametrize("damaged", [False, True])
def test_v3_with_legacy_triggers_realigns_before_mutation(tmp_path, operation, damaged, monkeypatch):
    """A v3 view plus old high-water triggers survives neither reopen nor cleanup.

    The truncation cuts a word in half. A legacy DELETE supplies the full word
    instead, leaving the indexed partial word behind even though rank 0 and
    document counts still agree. Exercise actual native cleanup paths, not just
    a trigger replacement or row-count assertion.
    """
    from hermes_state_common import _FTS_NEW_INDEXED_CONTENT_SQL, _FTS_OLD_INDEXED_CONTENT_SQL

    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    n = FTS_TOOL_CONTENT_PREFIX_CHARS
    text = "feedback " + " " * (n - len("feedback ") - 5) + "ghostboundary tailtoken"
    assert text[:n].endswith("ghost")
    for session in ("session", "doomed", "anchor"):
        first.create_session(session, source="cli")
    first.append_message("session", role="tool", content=text, tool_name="terminal")
    doomed = first.append_message("doomed", role="tool", content=text, tool_name="terminal")
    # Keep enough tokens for the wrong deletion to commit without an FTS totals underflow.
    first.append_message("anchor", role="assistant", content="feedback anchor reserve unrelated tokens remain alive")
    _strict_integrity_probe(first)

    # Reproduce the historical moving-marker projection, preserving each trigger's
    # real insertion/update/delete predicate and transaction boundaries.
    triggers = first._conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
        "AND name IN ('messages_fts_insert','messages_fts_update','messages_fts_delete')"
    ).fetchall()
    for name, sql in triggers:
        for alias, projection in (("new", _FTS_NEW_INDEXED_CONTENT_SQL), ("old", _FTS_OLD_INDEXED_CONTENT_SQL)):
            legacy = (
                f"CASE WHEN {alias}.role = 'tool' AND {alias}.id > COALESCE("
                "(SELECT CAST(value AS INTEGER) FROM state_meta "
                "WHERE key='fts_tool_full_content_high_water'), -1) "
                f"THEN substr(COALESCE({alias}.content, ''), 1, {n}) ELSE {alias}.content END"
            )
            sql = sql.replace(projection, legacy)
        first._conn.execute(f"DROP TRIGGER {name}")
        first._conn.execute(sql)
    first.set_meta("fts_tool_full_content_high_water", "883027")
    first._conn.execute("CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main,messages_fts,instance)")
    if damaged:
        assert first.delete_session("doomed")
        assert first._conn.execute("SELECT doc FROM temp.vocab WHERE term='ghost' AND doc=?", (doomed,)).fetchone()
        first._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('integrity-check')")
        assert first._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == first._conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
        with pytest.raises(sqlite3.DatabaseError):
            _strict_integrity_probe(first)
    first.close()

    migrated = SessionDB(db_path=path)
    try:
        _strict_integrity_probe(migrated)
        assert migrated.get_meta("fts_tool_full_content_high_water") is None
        assert migrated.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        migrated._conn.execute("CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main,messages_fts,instance)")
        assert not migrated._conn.execute("SELECT doc FROM temp.vocab WHERE doc NOT IN (SELECT id FROM messages) LIMIT 1").fetchone()
        migrated.append_message("session", role="tool", content=text, tool_name="terminal")
        migrated._execute_write(lambda c: c.execute("UPDATE messages SET content=? WHERE id=?", (text + " changed", doomed)))
        if operation == "prune_sessions":
            migrated.end_session("session", "test")
            assert migrated.prune_sessions(older_than_days=None) == 1
        elif operation in ("replace_messages", "archive_and_compact"):
            getattr(migrated, operation)("session", [{"role": "assistant", "content": "feedback summary"}])
        else:
            assert migrated.delete_session("session")
        _strict_integrity_probe(migrated)
        assert not migrated._conn.execute("SELECT doc FROM temp.vocab WHERE doc NOT IN (SELECT id FROM messages) LIMIT 1").fetchone()
        for query in ("f*", "fe*", "fee*", "feedback*"):
            sql, params = migrated._fts_match_sql(
                "messages_fts", query, "ORDER BY rank", limit=100, offset=0,
                include_inactive=False, source_filter=None, exclude_sources=None,
                role_filter=None, after_ts=None, before_ts=None,
            )
            assert migrated._conn.execute(sql, params).fetchall()
    finally:
        migrated.close()

    # A settled schema must not rebuild on every new reader.
    from hermes_state_schema import SessionSchemaMixin
    def unexpected_rebuild(*args, **kwargs):
        pytest.fail("settled v3 database rebuilt again")
    monkeypatch.setattr(SessionSchemaMixin, "_run_admitted_startup_rebuild", unexpected_rebuild)
    reopened = SessionDB(db_path=path)
    try:
        _strict_integrity_probe(reopened)
    finally:
        reopened.close()
