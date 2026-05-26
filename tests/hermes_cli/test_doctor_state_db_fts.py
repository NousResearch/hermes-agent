"""Doctor checks for state.db FTS5 index health."""

import sqlite3

from hermes_state import SessionDB
from hermes_cli.doctor import (
    _check_state_db_fts_integrity,
    _rebuild_state_db_fts_indexes,
)


def _seed_state_db(path):
    db = SessionDB(path)
    db.create_session("s1", source="cli")
    first_id = db.append_message("s1", role="user", content="the quick brown fox")
    db.append_message("s1", role="assistant", content="the lazy dog")
    db.close()
    return first_id


def test_state_db_fts_integrity_detects_missing_index_rows(tmp_path):
    db_path = tmp_path / "state.db"
    first_id = _seed_state_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        ok, detail, repairable = _check_state_db_fts_integrity(conn)
        assert ok is True
        assert repairable is False

        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (first_id,))
        conn.commit()

        ok, detail, repairable = _check_state_db_fts_integrity(conn)
        assert ok is False
        assert repairable is True
        assert "messages_fts" in detail
    finally:
        conn.close()


def test_rebuild_state_db_fts_indexes_restores_search(tmp_path):
    db_path = tmp_path / "state.db"
    first_id = _seed_state_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (first_id,))
        conn.execute("DELETE FROM messages_fts_trigram WHERE rowid = ?", (first_id,))
        conn.commit()
    finally:
        conn.close()

    _rebuild_state_db_fts_indexes(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        ok, detail, repairable = _check_state_db_fts_integrity(conn)
        assert ok is True, detail
        assert repairable is False
    finally:
        conn.close()

    db = SessionDB(db_path)
    try:
        hits = db.search_messages("quick", limit=5)
        assert any(hit["session_id"] == "s1" for hit in hits)
    finally:
        db.close()
