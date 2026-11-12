"""A state.db that an earlier release gave the cjk_unicode61 index opens without it: the index, its triggers
and its markers are dropped, so writers that never load the tokenizer (the sqlite3 CLI) can write again."""

import argparse
import sqlite3

import pytest

import hermes_state
from hermes_cli import sessions_cmd
from hermes_state import SessionDB

_CJK_TRIGGERS = ("messages_fts_cjk_insert", "messages_fts_cjk_delete", "messages_fts_cjk_update")


def _outside_insert(db_path) -> None:
    with sqlite3.connect(db_path) as raw:
        raw.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s1', 'user', '北京', 0)")


@pytest.fixture
def cjk_db(tmp_path):
    """The on-disk shape of a built index, as seen by a process that cannot load the tokenizer."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", source="cli", model="m")
    db.append_message("s1", "user", "东京都")
    db.close()
    raw = sqlite3.connect(db_path)
    raw.isolation_level = None
    raw.executescript(
        "CREATE VIEW messages_fts_cjk_src AS SELECT id, role, content, tool_name, tool_calls FROM messages"
        " WHERE role <> 'tool';"
        "CREATE VIRTUAL TABLE messages_fts_cjk USING fts5(content, tool_name, tool_calls,"
        " content='messages_fts_cjk_src', content_rowid='id', tokenize='unicode61');"
        "INSERT INTO state_meta (key, value) VALUES ('fts_cjk_rebuild_high_water', '1'),"
        " ('fts_cjk_rebuild_progress', '0'), ('fts_cjk_stale', '1');"
    )
    for name, event, row in zip(_CJK_TRIGGERS, ("INSERT", "DELETE", "UPDATE"), ("new", "old", "new")):
        raw.execute(f"CREATE TRIGGER {name} AFTER {event} ON messages BEGIN"
                    f" INSERT INTO messages_fts_cjk(rowid, content) VALUES ({row}.id, {row}.content); END")
    raw.execute("PRAGMA writable_schema=ON")
    raw.execute("UPDATE sqlite_master SET sql = replace(sql, \"'unicode61'\", \"'cjk_unicode61'\")"
                " WHERE name = 'messages_fts_cjk'")
    version = raw.execute("PRAGMA schema_version").fetchone()[0]
    raw.execute(f"PRAGMA schema_version={version + 1}")
    raw.execute("PRAGMA writable_schema=OFF")
    raw.close()
    with pytest.raises(sqlite3.OperationalError, match="no such tokenizer: cjk_unicode61"):
        _outside_insert(db_path)
    return db_path


def _cjk_residue(db_path) -> list:
    with sqlite3.connect(db_path) as raw:
        objects = raw.execute("SELECT name FROM sqlite_master WHERE name LIKE 'messages\\_fts\\_cjk%' ESCAPE '\\'")
        markers = raw.execute("SELECT key FROM state_meta WHERE key LIKE 'fts\\_cjk\\_%' ESCAPE '\\'")
        return [row[0] for row in (*objects.fetchall(), *markers.fetchall())]


def test_open_drops_the_cjk_index_and_restores_outside_writers(cjk_db):
    SessionDB(db_path=cjk_db).close()

    assert _cjk_residue(cjk_db) == []
    _outside_insert(cjk_db)
    with sqlite3.connect(cjk_db) as raw:
        assert raw.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_a_second_open_finds_nothing_to_drop(cjk_db, caplog):
    SessionDB(db_path=cjk_db).close()
    caplog.clear()

    db = SessionDB(db_path=cjk_db)
    try:
        db.append_message("s1", "user", "서울")
        assert db.search_messages("东京都")
    finally:
        db.close()
    assert _cjk_residue(cjk_db) == []
    assert not [r for r in caplog.records if "messages_fts_cjk" in r.getMessage()]


def _store_snapshot(db_path) -> tuple:
    with sqlite3.connect(db_path) as raw:
        schema = raw.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
        return schema, raw.execute("SELECT key, value FROM state_meta ORDER BY key").fetchall()


def test_repair_check_only_leaves_the_store_unchanged(cjk_db, monkeypatch):
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", cjk_db)
    before = _store_snapshot(cjk_db)

    assert not sessions_cmd._cmd_repair(argparse.Namespace(check_only=True, no_backup=False))

    assert _store_snapshot(cjk_db) == before
    SessionDB(db_path=cjk_db).close()
    assert _cjk_residue(cjk_db) == []


def test_open_finishes_an_interrupted_removal_once(cjk_db, caplog):
    with sqlite3.connect(cjk_db) as raw:
        for trigger in _CJK_TRIGGERS:
            raw.execute(f"DROP TRIGGER {trigger}")
        raw.execute("PRAGMA writable_schema=ON")
        raw.execute("DELETE FROM sqlite_master WHERE name = 'messages_fts_cjk'")
        raw.execute("PRAGMA writable_schema=RESET")
    assert _cjk_residue(cjk_db)

    SessionDB(db_path=cjk_db).close()
    assert _cjk_residue(cjk_db) == []
    caplog.clear()
    SessionDB(db_path=cjk_db).close()
    assert not [r for r in caplog.records if "messages_fts_cjk" in r.getMessage()]


def test_open_stages_the_index_for_chunked_teardown(cjk_db):
    db = SessionDB(db_path=cjk_db)
    try:
        staged = [r[0] for r in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'fts\\_v22\\_trash\\_messages\\_fts\\_cjk\\_%' ESCAPE '\\'")]
        assert "fts_v22_trash_messages_fts_cjk_data" in staged
        assert db.fts_optimize_available()
        assert db.optimize_fts_storage(vacuum=False)["ok"]
        assert not db._has_fts_trash(db._conn)
    finally:
        db.close()
