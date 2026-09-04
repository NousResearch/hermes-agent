"""Cron-source exclusion from the external-content trigram FTS index."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_TRIGRAM_SQL, SCHEMA_VERSION


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
        for row in db._conn.execute(
            "SELECT id FROM messages_fts_trigram_docsize ORDER BY id"
        ).fetchall()
    }


def _install_pre_v27_trigram(db: SessionDB, *, with_tool_calls: bool = False) -> None:
    """Recreate the pre-cron-exclusion external-content trigram boundary.

    ``with_tool_calls=True`` reproduces the FTS_STORAGE_VERSION 1 vtable
    (``tool_calls`` projected) that installs upgraded before #88217 carry;
    the default is the v2 column set with only the view/trigger predicates
    behind, which is what the in-place v29 migration handles.
    """
    cols = "content, tool_name" + (", tool_calls" if with_tool_calls else "")
    vals = "new.content, new.tool_name" + (", new.tool_calls" if with_tool_calls else "")
    db._conn.executescript(
        f"""
        DROP TRIGGER messages_fts_trigram_insert;
        DROP TRIGGER messages_fts_trigram_delete;
        DROP TRIGGER messages_fts_trigram_update;
        DROP TABLE messages_fts_trigram;
        DROP VIEW messages_fts_trigram_src;
        CREATE VIEW messages_fts_trigram_src AS
            SELECT id, role, content, tool_name, tool_calls
            FROM messages WHERE role <> 'tool';
        CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
            {cols},
            content='messages_fts_trigram_src',
            content_rowid='id',
            tokenize='trigram'
        );
        CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages
        WHEN new.role <> 'tool'
        BEGIN
            INSERT INTO messages_fts_trigram(rowid, {cols})
            VALUES (new.id, {vals});
        END;
        """
    )


def test_fresh_trigram_indexes_conversations_but_not_cron(db: SessionDB):
    db.create_session("cli", source="cli")
    db.create_session("cron", source="cron")
    cli_id = db.append_message("cli", role="user", content="交付状态正常")
    cron_id = db.append_message("cron", role="user", content="定时任务状态正常")

    assert _trigram_rowids(db) == {cli_id}
    assert cron_id not in _trigram_rowids(db)
    assert db._conn.execute(
        "SELECT id FROM messages_fts_docsize WHERE id = ?", (cron_id,)
    ).fetchone() is not None


def test_cron_remains_searchable_via_standard_fts_and_explicit_cjk_fallback(
    db: SessionDB,
):
    db.create_session("cron", source="cron")
    db.append_message(
        "cron", role="assistant", content="quarterly archive 大别山项目 complete"
    )

    assert [row["session_id"] for row in db.search_messages("quarterly")] == [
        "cron"
    ]
    assert [
        row["session_id"]
        for row in db.search_messages("大别山项目", source_filter=["cron"])
    ] == ["cron"]


def test_deferred_rebuild_does_not_reintroduce_cron(db: SessionDB):
    db.create_session("cli", source="cli")
    db.create_session("cron", source="cron")
    cli_id = db.append_message("cli", role="assistant", content="交互会话内容")
    db.append_message("cron", role="assistant", content="定时会话内容")

    with db._lock:
        db._reset_fts_index_to_empty(db._conn)
        db._seed_fts_rebuild_markers(db._conn, force=True)
        db._conn.commit()
    while db.fts_rebuild_step():
        pass

    assert _trigram_rowids(db) == {cli_id}


def test_existing_external_layout_rebuilds_trigram_on_upgrade(tmp_path):
    db_path = tmp_path / "state.db"
    old = SessionDB(db_path=db_path)
    if not old._trigram_available:
        old.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    # The virtual table keeps referring to the view by name, so this recreates
    # the exact old external-content boundary without reading source text.
    _install_pre_v27_trigram(old)
    old.create_session("cli", source="cli")
    old.create_session("cron", source="cron")
    cli_id = old.append_message("cli", role="user", content="交互迁移内容")
    cron_id = old.append_message("cron", role="user", content="定时迁移内容")
    assert _trigram_rowids(old) == {cli_id, cron_id}
    old._conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
    old._conn.commit()
    old.close()

    migrated = SessionDB(db_path=db_path)
    try:
        assert _trigram_rowids(migrated) == {cli_id}
        view_sql = migrated._conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'view' AND name = 'messages_fts_trigram_src'"
        ).fetchone()[0]
        assert "sessions" in view_sql
        assert "cron" in view_sql
        migrated._conn.execute(
            "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('integrity-check')"
        )
    finally:
        migrated.close()


def test_install_already_at_v28_still_gets_the_cron_exclusion_migration(tmp_path):
    """The migration gate must fire for installs that were on main's v28.

    The original PR gated on ``current_version < 27``; main had meanwhile
    reached SCHEMA_VERSION 28 via column-reconciliation bumps, so a v28
    database would have skipped the rebuild and kept cron rows in the trigram
    index forever. Pin the gate against the version main actually shipped.
    """
    db_path = tmp_path / "state.db"
    old = SessionDB(db_path=db_path)
    if not old._trigram_available:
        old.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    _install_pre_v27_trigram(old)
    old.create_session("cli", source="cli")
    old.create_session("cron", source="cron")
    cli_id = old.append_message("cli", role="user", content="交互迁移内容")
    cron_id = old.append_message("cron", role="user", content="定时迁移内容")
    assert _trigram_rowids(old) == {cli_id, cron_id}
    old._conn.execute("UPDATE schema_version SET version = 28")
    old._conn.commit()
    old.close()

    migrated = SessionDB(db_path=db_path)
    try:
        assert _trigram_rowids(migrated) == {cli_id}, (
            "a v28 database kept cron rows in the trigram index: the migration gate did not fire"
        )
    finally:
        migrated.close()


def test_v1_tool_calls_layout_is_left_for_optimize_storage(tmp_path):
    """A FTS_STORAGE_VERSION 1 trigram vtable (``tool_calls`` projected) must
    survive the v29 startup migration untouched and be finished by the opt-in
    ``optimize_fts_storage`` path — not half-migrated into a view/vtable
    column mismatch (which used to fail the rebuild with
    ``no such column: T.tool_calls``)."""
    db_path = tmp_path / "state.db"
    old = SessionDB(db_path=db_path)
    if not old._trigram_available:
        old.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    _install_pre_v27_trigram(old, with_tool_calls=True)
    old.create_session("cli", source="cli")
    old.create_session("cron", source="cron")
    cli_id = old.append_message("cli", role="user", content="交互迁移内容")
    cron_id = old.append_message("cron", role="user", content="定时迁移内容")
    assert _trigram_rowids(old) == {cli_id, cron_id}
    old._conn.execute("UPDATE schema_version SET version = 28")
    old._conn.commit()
    old.close()

    migrated = SessionDB(db_path=db_path)  # must not raise
    try:
        # Startup left the v1 layout alone (cron row still there) …
        assert _trigram_rowids(migrated) == {cli_id, cron_id}
        assert migrated.fts_optimize_available() is True
        # … and the opt-in path completes the transition: v2 columns,
        # cron-filtered view, cron row purged.
        migrated.optimize_fts_storage()
        cols = [r[1] for r in migrated._conn.execute("PRAGMA table_info(messages_fts_trigram)")]
        assert "tool_calls" not in cols
        assert _trigram_rowids(migrated) == {cli_id}
    finally:
        migrated.close()


def test_partial_upgrade_view_does_not_skip_historical_rebuild(tmp_path):
    db_path = tmp_path / "state.db"
    old = SessionDB(db_path=db_path)
    if not old._trigram_available:
        old.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    _install_pre_v27_trigram(old)
    old.create_session("cron", source="cron")
    cron_id = old.append_message("cron", role="assistant", content="迁移中断内容")
    assert _trigram_rowids(old) == {cron_id}

    # Simulate a crash after new DDL landed but before the rebuild/schema stamp.
    for name in (
        "messages_fts_trigram_insert",
        "messages_fts_trigram_delete",
        "messages_fts_trigram_update",
    ):
        old._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    old._conn.execute("DROP VIEW messages_fts_trigram_src")
    old._conn.executescript(FTS_TRIGRAM_SQL)
    old._conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
    old._conn.commit()
    old.close()

    migrated = SessionDB(db_path=db_path)
    try:
        assert _trigram_rowids(migrated) == set()
    finally:
        migrated.close()


def test_delete_of_unindexed_cron_row_keeps_trigram_consistent(db: SessionDB):
    db.create_session("cron", source="cron")
    cron_id = db.append_message("cron", role="user", content="不会进入索引")
    assert cron_id not in _trigram_rowids(db)

    db._conn.execute("DELETE FROM messages WHERE id = ?", (cron_id,))
    db._conn.execute(
        "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('integrity-check')"
    )
