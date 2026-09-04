"""Delegate-child (subagent) transcripts stay out of the trigram FTS index (v30).

Mirrors ``test_fts_trigram_cron_exclusion.py``: children are canonical rows
in ``messages`` and stay searchable through the standard ``messages_fts``
word index; only the trigram (CJK substring) shadow index skips them.
"""

from __future__ import annotations

import pytest

from hermes_state import SCHEMA_VERSION, SessionDB
from hermes_state_common import FTS_TRIGRAM_EXCLUDED_SOURCES, fts_trigram_session_sql


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    if not session_db._trigram_available:
        session_db.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    yield session_db
    session_db.close()


def _trigram_rowids(db: SessionDB) -> set[int]:
    return {
        row[0]
        for row in db._conn.execute("SELECT id FROM messages_fts_trigram_docsize").fetchall()
    }


def _fts_rowids(db: SessionDB) -> set[int]:
    return {
        row[0] for row in db._conn.execute("SELECT id FROM messages_fts_docsize").fetchall()
    }


def _seed(db: SessionDB) -> dict[str, int]:
    db.create_session("root", source="cli")
    # delegate_tool children: source='subagent' via platform, plus the
    # _delegate_from creation marker.
    db.create_session(
        "kid", source="subagent", parent_session_id="root",
        model_config={"_delegate_from": "root"},
    )
    # A child spawned under a gateway turn inherits the gateway's source but
    # still carries the marker.
    db.create_session(
        "gw-kid", source="telegram", parent_session_id="root",
        model_config={"_delegate_from": "root"},
    )
    # Compression continuation: parent_session_id but NO marker -> indexed.
    db.create_session("cont", source="cli", parent_session_id="root")
    return {
        "root": db.append_message("root", role="user", content="交付状态正常 root-word"),
        "kid": db.append_message("kid", role="assistant", content="子任务状态正常 kid-word"),
        "gw-kid": db.append_message("gw-kid", role="assistant", content="网关子任务 gwkid-word"),
        "cont": db.append_message("cont", role="assistant", content="继续会话内容 cont-word"),
    }


def test_subagent_rows_skip_trigram_but_stay_in_standard_fts(db: SessionDB):
    ids = _seed(db)
    assert _trigram_rowids(db) == {ids["root"], ids["cont"]}
    assert _fts_rowids(db) >= set(ids.values())


def test_subagent_rows_remain_word_searchable(db: SessionDB):
    _seed(db)
    assert [r["session_id"] for r in db.search_messages("kid-word")] == ["kid"]
    assert [r["session_id"] for r in db.search_messages("gwkid-word")] == ["gw-kid"]
    # Explicit CJK search scoped to the excluded source falls back to LIKE.
    assert [
        r["session_id"]
        for r in db.search_messages("子任务状态", source_filter=["subagent"])
    ] == ["kid"]
    # Top-level CJK substring search unaffected.
    assert [r["session_id"] for r in db.search_messages("交付状态")] == ["root"]


def test_update_and_delete_of_unindexed_child_row_keep_trigram_consistent(db: SessionDB):
    ids = _seed(db)
    db._conn.execute(
        "UPDATE messages SET content = ? WHERE id = ?", ("改写后的内容", ids["kid"])
    )
    db._conn.execute("DELETE FROM messages WHERE id = ?", (ids["kid"],))
    db._conn.execute(
        "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('integrity-check')"
    )
    assert _trigram_rowids(db) == {ids["root"], ids["cont"]}


def test_deferred_rebuild_does_not_reintroduce_children(db: SessionDB):
    ids = _seed(db)
    with db._lock:
        db._reset_fts_index_to_empty(db._conn)
        db._seed_fts_rebuild_markers(db._conn, force=True)
        db._conn.commit()
    while db.fts_rebuild_step():
        pass
    assert _trigram_rowids(db) == {ids["root"], ids["cont"]}
    assert _fts_rowids(db) >= set(ids.values())


def test_full_rebuild_honours_exclusion(db: SessionDB):
    ids = _seed(db)
    db.rebuild_fts()
    assert _trigram_rowids(db) == {ids["root"], ids["cont"]}


def test_v29_install_purges_child_rows_on_upgrade(tmp_path):
    db_path = tmp_path / "state.db"
    old = SessionDB(db_path=db_path)
    if not old._trigram_available:
        old.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    # Recreate the v29 (cron-only) view/trigger boundary.
    old._conn.executescript(
        """
        DROP TRIGGER messages_fts_trigram_insert;
        DROP TRIGGER messages_fts_trigram_delete;
        DROP TRIGGER messages_fts_trigram_update;
        DROP VIEW messages_fts_trigram_src;
        CREATE VIEW messages_fts_trigram_src AS
            SELECT m.id, m.role, m.content, m.tool_name
            FROM messages AS m JOIN sessions AS s ON s.id = m.session_id
            WHERE m.role <> 'tool' AND s.source <> 'cron';
        CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages
        WHEN new.role <> 'tool'
           AND EXISTS (SELECT 1 FROM sessions WHERE id = new.session_id AND source <> 'cron')
        BEGIN
            INSERT INTO messages_fts_trigram(rowid, content, tool_name)
            VALUES (new.id, new.content, new.tool_name);
        END;
        """
    )
    ids = _seed(old)
    assert _trigram_rowids(old) == set(ids.values())
    old._conn.execute("UPDATE schema_version SET version = 29")
    old._conn.commit()
    old.close()

    migrated = SessionDB(db_path=db_path)
    try:
        assert _trigram_rowids(migrated) == {ids["root"], ids["cont"]}
        assert migrated._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0] == SCHEMA_VERSION
        migrated._conn.execute(
            "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('integrity-check')"
        )
    finally:
        migrated.close()


def test_predicate_constants_agree():
    assert "subagent" in FTS_TRIGRAM_EXCLUDED_SOURCES
    assert "cron" in FTS_TRIGRAM_EXCLUDED_SOURCES
    sql = fts_trigram_session_sql("s")
    assert sql.startswith("s.source NOT IN (") and "s.model_config" in sql
