import sqlite3

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema import _db_has_legacy_inline_fts
from persistence.sqlite._schema_ddl import (
    FTS_TOOL_CONTENT_PREFIX_CHARS,
    LEGACY_FTS_SQL,
    _FTS_TRIGGERS,
)
from persistence.sqlite.admin._maintenance import _FtsStorageOptimizer
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.search import rebuild_fts, search_messages
from persistence.sqlite.conversation._store.sessions import create_session


def _long_message(prefix: str, tail: str) -> str:
    padding = "padding " * (FTS_TOOL_CONTENT_PREFIX_CHARS // len("padding ") + 8)
    return f"{prefix} {padding} {tail}"


@pytest.fixture
def db(tmp_path):
    session_db = StoreHolder(db_path=tmp_path / "state.db")
    if not session_db._fts_enabled:
        session_db.close()
        pytest.skip("SQLite FTS5 unavailable")
    create_session(session_db, "session", source="cli")
    try:
        yield session_db
    finally:
        session_db.close()


def test_new_tool_rows_bound_fts_content_but_explicit_tool_search_is_complete(db):
    tool_id = append_message(db,
        "session",
        role="tool",
        content=_long_message("indexed-prefix-token", "tool-tail-token"),
        tool_name="terminal",
    )
    user_id = append_message(db,
        "session",
        role="user",
        content=_long_message("user-prefix-token", "user-tail-token"),
    )

    assert [row["id"] for row in search_messages(db, "indexed-prefix-token")] == [
        tool_id
    ]
    assert search_messages(db, "tool-tail-token") == []
    assert [
        row["id"]
        for row in search_messages(db, "tool-tail-token", role_filter=["tool"])
    ] == [tool_id]
    assert [row["id"] for row in search_messages(db, "user-tail-token")] == [
        user_id
    ]


def test_reopen_keeps_bounded_rows_consistent_without_a_marker(tmp_path):
    """The bounded projection is a fixed per-row function: reopening a store does
    not need a marker, and the redaction/delete paths that used to re-evaluate
    one keep the index consistent for both historical and new tool rows."""
    path = tmp_path / "state.db"
    first = StoreHolder(db_path=path)
    if not first._fts_enabled:
        first.close()
        pytest.skip("SQLite FTS5 unavailable")
    create_session(first, "session", source="cli")
    historical_id = append_message(first,
        "session",
        role="tool",
        content=_long_message("historical-prefix-token", "historical-tail-token"),
    )
    assert search_messages(first, "historical-tail-token") == []
    first.close()

    reopened = StoreHolder(db_path=path)
    try:
        assert reopened.get_meta("fts_storage_version") is not None
        new_id = append_message(reopened,
            "session",
            role="tool",
            content=_long_message("new-prefix-token", "new-tail-token"),
        )
        assert search_messages(reopened, "new-tail-token") == []
        assert [
            row["id"]
            for row in search_messages(reopened,
                "new-tail-token", role_filter=["tool"]
            )
        ] == [new_id]

        # Redaction must remove the tail token from the index as well.
        reopened._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET content = '' WHERE id = ?", (historical_id,)
            )
        )
        assert search_messages(reopened, "historical-tail-token") == []

        # Bounded rows use the same projection for insert and delete; a mismatch
        # corrupts external-content FTS and makes this delete or later write fail.
        reopened._execute_write(
            lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (new_id,))
        )
        append_message(reopened, "session", role="assistant", content="fts-still-healthy")
        assert search_messages(reopened, "fts-still-healthy")
        reopened._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
    finally:
        reopened.close()


def test_full_rebuild_keeps_every_tool_row_bounded(db):
    """A rebuild fills the index from the same stable projection as the triggers,
    so it cannot introduce full-content tokens for tool rows again."""
    before_id = append_message(db,
        "session",
        role="tool",
        content=_long_message("before-prefix-token", "before-tail-token"),
    )
    assert search_messages(db, "before-tail-token") == []

    assert rebuild_fts(db) >= 1
    assert search_messages(db, "before-tail-token") == []

    after_id = append_message(db,
        "session",
        role="tool",
        content=_long_message("after-prefix-token", "after-tail-token"),
    )
    assert search_messages(db, "after-tail-token") == []
    assert [
        row["id"]
        for row in search_messages(db, "after-tail-token", role_filter=["tool"])
    ] == [after_id]

    db._conn.execute(
        "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
    )

    # The deferred chunked backfill + boundary sweep must feed the index through the
    # same truncated tool projection; an untruncated write path leaves the
    # external-content index disagreeing with the triggers and fails the strict probe.
    with db._lock:
        _FtsStorageOptimizer(db)._reset_fts_index_to_empty(db._conn)
        _FtsStorageOptimizer(db)._seed_fts_rebuild_markers(db._conn, force=True)
        db._conn.commit()
    while _FtsStorageOptimizer(db).fts_rebuild_step():
        pass
    assert db.get_meta("fts_rebuild_high_water") is None
    assert search_messages(db, "after-tail-token") == []
    assert {
        row["id"] for row in search_messages(db, "prefix-token", role_filter=["tool"])
    } == {before_id, after_id}
    db._conn.execute(
        "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
    )


def test_role_changes_switch_between_bounded_and_full_indexing(db):
    message_id = append_message(db,
        "session",
        role="tool",
        content=_long_message("role-prefix-token", "role-tail-token"),
    )
    assert search_messages(db, "role-tail-token") == []

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET role = 'assistant' WHERE id = ?", (message_id,)
        )
    )
    assert [row["id"] for row in search_messages(db, "role-tail-token")] == [
        message_id
    ]

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET role = 'tool' WHERE id = ?", (message_id,)
        )
    )
    assert search_messages(db, "role-tail-token") == []


def test_legacy_inline_fts_also_bounds_new_tool_rows(tmp_path):
    path = tmp_path / "legacy.db"
    initial = StoreHolder(db_path=path)
    create_session(initial, "session", source="cli")
    for trigger in _FTS_TRIGGERS:
        initial._conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    initial._conn.execute("DROP TABLE IF EXISTS messages_fts_trigram")
    initial._conn.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
    initial._conn.execute("DROP TABLE IF EXISTS messages_fts")
    initial._conn.executescript(LEGACY_FTS_SQL)
    initial._conn.execute(
        "DELETE FROM state_meta WHERE key IN (?, 'fts_storage_version')",
        ("fts_tool_full_content_high_water",),
    )
    initial.close()

    legacy = StoreHolder(db_path=path)
    try:
        assert _db_has_legacy_inline_fts(legacy._conn.cursor()) is True
        message_id = append_message(legacy,
            "session",
            role="tool",
            content=_long_message("legacy-prefix-token", "legacy-tail-token"),
        )
        assert search_messages(legacy, "legacy-tail-token") == []
        assert [
            row["id"]
            for row in search_messages(legacy,
                "legacy-tail-token", role_filter=["tool"]
            )
        ] == [message_id]
    finally:
        legacy.close()
