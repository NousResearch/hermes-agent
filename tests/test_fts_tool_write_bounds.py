import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import (
    FTS_TOOL_CONTENT_PREFIX_CHARS,
    FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY,
    LEGACY_FTS_SQL,
    _FTS_TRIGGERS,
)


def _long_message(prefix: str, tail: str) -> str:
    padding = "padding " * (FTS_TOOL_CONTENT_PREFIX_CHARS // len("padding ") + 8)
    return f"{prefix} {padding} {tail}"


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


def test_new_tool_rows_bound_fts_content_but_explicit_tool_search_is_complete(db):
    tool_id = db.append_message(
        "session",
        role="tool",
        content=_long_message("indexed-prefix-token", "tool-tail-token"),
        tool_name="terminal",
    )
    user_id = db.append_message(
        "session",
        role="user",
        content=_long_message("user-prefix-token", "user-tail-token"),
    )

    assert [row["id"] for row in db.search_messages("indexed-prefix-token")] == [
        tool_id
    ]
    assert db.search_messages("tool-tail-token") == []
    assert [
        row["id"]
        for row in db.search_messages("tool-tail-token", role_filter=["tool"])
    ] == [tool_id]
    assert [row["id"] for row in db.search_messages("user-tail-token")] == [
        user_id
    ]


def test_trigger_migration_preserves_historical_tool_tokens_without_rebuild(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    if not first._fts_enabled:
        first.close()
        pytest.skip("SQLite FTS5 unavailable")
    first.create_session("session", source="cli")

    # Model the pre-migration trigger contract: every id through this artificial
    # boundary receives full-content indexing.
    first.set_meta(FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY, str(2**62))
    old_id = first.append_message(
        "session",
        role="tool",
        content=_long_message("old-prefix-token", "old-tail-token"),
    )
    assert [row["id"] for row in first.search_messages("old-tail-token")] == [old_id]
    first._conn.execute(
        "DELETE FROM state_meta WHERE key = ?",
        (FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY,),
    )
    first.close()

    migrated = SessionDB(db_path=path)
    try:
        assert int(migrated.get_meta(FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY)) == old_id
        assert [
            row["id"] for row in migrated.search_messages("old-tail-token")
        ] == [old_id]

        new_id = migrated.append_message(
            "session",
            role="tool",
            content=_long_message("new-prefix-token", "new-tail-token"),
        )
        assert migrated.search_messages("new-tail-token") == []
        assert [
            row["id"]
            for row in migrated.search_messages(
                "new-tail-token", role_filter=["tool"]
            )
        ] == [new_id]

        # Historical rows still use their full old token stream for the FTS5
        # external-content delete command; redaction must remove the tail token.
        migrated._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET content = '' WHERE id = ?", (old_id,)
            )
        )
        assert migrated.search_messages("old-tail-token") == []

        # New bounded rows use the same prefix for delete as insert. A mismatch
        # corrupts external-content FTS and makes this delete or later write fail.
        migrated._execute_write(
            lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (new_id,))
        )
        migrated.append_message("session", role="assistant", content="fts-still-healthy")
        assert migrated.search_messages("fts-still-healthy")
    finally:
        migrated.close()


def test_full_rebuild_moves_boundary_before_future_tool_writes(db):
    before_id = db.append_message(
        "session",
        role="tool",
        content=_long_message("before-prefix-token", "before-tail-token"),
    )
    assert db.search_messages("before-tail-token") == []

    assert db.rebuild_fts() >= 1
    assert int(db.get_meta(FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY)) == before_id
    assert [row["id"] for row in db.search_messages("before-tail-token")] == [
        before_id
    ]

    after_id = db.append_message(
        "session",
        role="tool",
        content=_long_message("after-prefix-token", "after-tail-token"),
    )
    assert db.search_messages("after-tail-token") == []
    assert [
        row["id"]
        for row in db.search_messages("after-tail-token", role_filter=["tool"])
    ] == [after_id]


def test_role_changes_switch_between_bounded_and_full_indexing(db):
    message_id = db.append_message(
        "session",
        role="tool",
        content=_long_message("role-prefix-token", "role-tail-token"),
    )
    assert db.search_messages("role-tail-token") == []

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET role = 'assistant' WHERE id = ?", (message_id,)
        )
    )
    assert [row["id"] for row in db.search_messages("role-tail-token")] == [
        message_id
    ]

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET role = 'tool' WHERE id = ?", (message_id,)
        )
    )
    assert db.search_messages("role-tail-token") == []


def test_legacy_inline_fts_also_bounds_new_tool_rows(tmp_path):
    path = tmp_path / "legacy.db"
    initial = SessionDB(db_path=path)
    initial.create_session("session", source="cli")
    for trigger in _FTS_TRIGGERS:
        initial._conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    initial._conn.execute("DROP TABLE IF EXISTS messages_fts_trigram")
    initial._conn.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
    initial._conn.execute("DROP TABLE IF EXISTS messages_fts")
    initial._conn.executescript(LEGACY_FTS_SQL)
    initial._conn.execute(
        "DELETE FROM state_meta WHERE key IN (?, 'fts_storage_version')",
        (FTS_TOOL_FULL_CONTENT_HIGH_WATER_KEY,),
    )
    initial.close()

    legacy = SessionDB(db_path=path)
    try:
        assert legacy._db_has_legacy_inline_fts(legacy._conn.cursor()) is True
        message_id = legacy.append_message(
            "session",
            role="tool",
            content=_long_message("legacy-prefix-token", "legacy-tail-token"),
        )
        assert legacy.search_messages("legacy-tail-token") == []
        assert [
            row["id"]
            for row in legacy.search_messages(
                "legacy-tail-token", role_filter=["tool"]
            )
        ] == [message_id]
    finally:
        legacy.close()
