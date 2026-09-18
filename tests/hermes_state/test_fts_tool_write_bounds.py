import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import (
    FTS_TOOL_CONTENT_PREFIX_CHARS,
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


def test_reopen_keeps_bounded_rows_consistent_without_a_marker(tmp_path):
    """The bounded projection is a fixed per-row function: reopening a store does
    not need a marker, and the redaction/delete paths that used to re-evaluate
    one keep the index consistent for both historical and new tool rows."""
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    if not first._fts_enabled:
        first.close()
        pytest.skip("SQLite FTS5 unavailable")
    first.create_session("session", source="cli")
    historical_id = first.append_message(
        "session",
        role="tool",
        content=_long_message("historical-prefix-token", "historical-tail-token"),
    )
    assert first.search_messages("historical-tail-token") == []
    first.close()

    reopened = SessionDB(db_path=path)
    try:
        assert reopened.get_meta("fts_storage_version") is not None
        new_id = reopened.append_message(
            "session",
            role="tool",
            content=_long_message("new-prefix-token", "new-tail-token"),
        )
        assert reopened.search_messages("new-tail-token") == []
        assert [
            row["id"]
            for row in reopened.search_messages(
                "new-tail-token", role_filter=["tool"]
            )
        ] == [new_id]

        # Redaction must remove the tail token from the index as well.
        reopened._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET content = '' WHERE id = ?", (historical_id,)
            )
        )
        assert reopened.search_messages("historical-tail-token") == []

        # Bounded rows use the same projection for insert and delete; a mismatch
        # corrupts external-content FTS and makes this delete or later write fail.
        reopened._execute_write(
            lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (new_id,))
        )
        reopened.append_message("session", role="assistant", content="fts-still-healthy")
        assert reopened.search_messages("fts-still-healthy")
        reopened._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
    finally:
        reopened.close()


def test_full_rebuild_keeps_every_tool_row_bounded(db):
    """A rebuild fills the index from the same stable projection as the triggers,
    so it cannot introduce full-content tokens for tool rows again."""
    before_id = db.append_message(
        "session",
        role="tool",
        content=_long_message("before-prefix-token", "before-tail-token"),
    )
    assert db.search_messages("before-tail-token") == []

    assert db.rebuild_fts() >= 1
    assert db.search_messages("before-tail-token") == []

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

    db._conn.execute(
        "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
    )

    # The deferred chunked backfill + boundary sweep must feed the index through the
    # same truncated tool projection; an untruncated write path leaves the
    # external-content index disagreeing with the triggers and fails the strict probe.
    with db._lock:
        db._reset_fts_index_to_empty(db._conn)
        db._seed_fts_rebuild_markers(db._conn, force=True)
        db._conn.commit()
    while db.fts_rebuild_step():
        pass
    assert db.get_meta("fts_rebuild_high_water") is None
    assert db.search_messages("after-tail-token") == []
    assert {
        row["id"] for row in db.search_messages("prefix-token", role_filter=["tool"])
    } == {before_id, after_id}
    db._conn.execute(
        "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
    )


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
        ("fts_tool_full_content_high_water",),
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
