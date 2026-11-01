"""A short FTS projection is not a finished one (follow-up to #72716).

The open-time check only asks whether the base index holds any row, so a store whose trigram index was truncated
(or whose base index lost most of ``messages``) used to be stamped complete, and ``optimize-storage`` reported
nothing to do. These contracts pin that optimize-storage offers and finishes that repair, that the repair cannot
leave redacted text searchable, and that a layout this release does not recognise is still left alone.
"""

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_STORAGE_VERSION
from hermes_state_fts import fts_projection_short


def _count(db: SessionDB, sql: str, *params) -> int:
    return db._conn.execute(sql, params).fetchone()[0]


def _indexed(db: SessionDB, table: str, row_id: int) -> bool:
    return db._conn.execute(f"SELECT 1 FROM {table}_docsize WHERE id = ?", (row_id,)).fetchone() is not None


def _match_rowids(db: SessionDB, token: str) -> list[int]:
    return [r[0] for r in db._conn.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?", (token,))]


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    if not session_db._trigram_available:
        session_db.close()
        pytest.skip("trigram tokenizer unavailable")
    yield session_db
    session_db.close()


def _truncate_trigram_to(db: SessionDB, keep_id: int) -> None:
    """Empty the trigram index, then index only *keep_id*: the post-switch shape where historical rows are gone."""
    db._conn.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('delete-all')")
    db._conn.execute(
        "INSERT INTO messages_fts_trigram(rowid, content, tool_name) "
        "SELECT id, content, tool_name FROM messages WHERE id = ?", (keep_id,))
    db._conn.commit()


def _trigram_shortfall(db: SessionDB, token: str) -> int:
    """13 indexed rows, then one trigram entry dropped so only that projection is short; returns the newest id."""
    db.create_session(session_id="s1", source="cli")
    for i in range(12):
        db.append_message("s1", "user", f"filler payload {i}")
    db.append_message("s1", "user", f"{token} redaction target")
    target_id = _count(db, "SELECT MAX(id) FROM messages")
    dropped = db._conn.execute(
        "SELECT id, content, tool_name FROM messages WHERE id = (SELECT MIN(id) FROM messages)").fetchone()
    db._conn.execute(
        "INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name) "
        "VALUES ('delete', ?, ?, ?)", tuple(dropped))
    db._conn.commit()
    assert _count(db, "SELECT COUNT(*) FROM messages_fts_docsize") == 13
    assert _count(db, "SELECT COUNT(*) FROM messages_fts_trigram_docsize") == 12
    return target_id


def test_optimize_heals_a_truncated_trigram_index_stamped_complete(db):
    """Stamped v23, complete base index, a trigram index of one row and no rebuild markers: optimize-storage used
    to print "already on the compact layout" and leave historical substring coverage missing."""
    db.create_session(session_id="s1", source="cli")
    for i, role in enumerate(("user", "assistant", "tool") * 4):
        db.append_message("s1", role=role, content=f"trigram-history-{role}-{i}")
    n_msg = _count(db, "SELECT COUNT(*) FROM messages")
    n_src = _count(db, "SELECT COUNT(*) FROM messages_fts_trigram_src")
    keep_id = _count(db, "SELECT MIN(id) FROM messages WHERE role <> 'tool'")
    later_id = _count(db, "SELECT MAX(id) FROM messages WHERE role <> 'tool'")
    _truncate_trigram_to(db, keep_id)
    assert db._fts_external_index_empty_with_messages(db._conn) is False
    assert fts_projection_short(db._conn) is True
    assert db.get_meta("fts_rebuild_high_water") is None

    assert db.fts_optimize_available() is True
    assert db.optimize_fts_storage(vacuum=False)["ok"] is True

    assert _count(db, "SELECT COUNT(*) FROM messages_fts_docsize") == n_msg
    assert _count(db, "SELECT COUNT(*) FROM messages_fts_trigram_docsize") == n_src
    assert _indexed(db, "messages_fts_trigram", keep_id) and _indexed(db, "messages_fts_trigram", later_id)
    assert db.get_meta("fts_rebuild_high_water") is None
    assert db.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
    assert db.fts_optimize_available() is False


def test_repairing_a_short_projection_does_not_keep_redacted_tokens(db):
    """Seeding rebuild markers over a populated base index made the UPDATE trigger skip those ids, so a row
    redacted inside (progress, high_water] stayed searchable after optimize reported success."""
    token = "secrettokenbefore"
    target_id = _trigram_shortfall(db, token)
    assert target_id in _match_rowids(db, token)

    db._repair_optimize_bookkeeping()
    assert int(db.get_meta("fts_rebuild_high_water")) >= target_id
    db._conn.execute("UPDATE messages SET content = '' WHERE id = ?", (target_id,))
    db._conn.commit()

    assert db.optimize_fts_storage(vacuum=False)["ok"] is True
    assert target_id not in _match_rowids(db, token)
    assert db.search_messages(token) == []


def test_a_delete_during_the_repair_leaves_no_index_row_behind(db):
    token = "secrettokenbefore"
    target_id = _trigram_shortfall(db, token)
    db._repair_optimize_bookkeeping()
    db._conn.execute("DELETE FROM messages WHERE id = ?", (target_id,))
    db._conn.commit()

    assert db.optimize_fts_storage(vacuum=False)["ok"] is True
    assert not _indexed(db, "messages_fts", target_id)
    assert not _indexed(db, "messages_fts_trigram", target_id)
    assert _count(db, "SELECT COUNT(*) FROM messages_fts_docsize") == _count(db, "SELECT COUNT(*) FROM messages")
    assert _count(db, "SELECT COUNT(*) FROM messages_fts_trigram_docsize") == _count(
        db, "SELECT COUNT(*) FROM messages_fts_trigram_src")


def test_a_short_index_on_a_layout_this_release_does_not_recognise_is_left_alone(db):
    """Downgrade: a newer release's trigram definition is not this release's to empty and rebuild, short or not."""
    db.create_session(session_id="s1", source="cli")
    for i in range(6):
        db.append_message("s1", "user", f"newer layout row {i}")
    sql = db._conn.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_fts_trigram'").fetchone()[0]
    db._conn.execute("DROP TABLE messages_fts_trigram")
    db._conn.execute(sql.replace("tokenize='trigram'", "tokenize='trigram case_sensitive 1'"))
    keep_id = _count(db, "SELECT MIN(id) FROM messages")
    _truncate_trigram_to(db, keep_id)

    assert fts_projection_short(db._conn) is False
    assert db.fts_optimize_available() is False
    db.optimize_fts_storage(vacuum=False)
    assert _count(db, "SELECT COUNT(*) FROM messages_fts_trigram_docsize") == 1
