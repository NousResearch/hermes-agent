"""Tests for gateway session scope metadata.

Phase 1 of Discord thread isolation stores scope labels on session rows
without changing session_search behavior yet.
"""

import sqlite3

import gateway.session as session_module
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


def test_discord_thread_scope_key_uses_thread_id() -> None:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-456",
        chat_type="thread",
        thread_id="thread-456",
        parent_chat_id="parent-123",
        guild_id="guild-1",
    )

    scope = session_module.build_session_scope(
        source,
        session_key="agent:main:discord:thread:thread-456",
    )

    assert scope.scope_key == "discord:thread:thread-456"
    assert scope.scope_kind == "current_thread"


def test_discord_channel_scope_key_uses_channel_id_without_thread() -> None:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-123",
        chat_type="group",
        guild_id="guild-1",
    )

    scope = session_module.build_session_scope(
        source,
        session_key="agent:main:discord:group:channel-123",
    )

    assert scope.scope_key == "discord:channel:channel-123"
    assert scope.scope_kind == "channel"


def test_session_db_has_scope_columns(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")

    cols = {
        row[1]
        for row in db._conn.execute("PRAGMA table_info(sessions)").fetchall()
    }

    assert {
        "session_key",
        "chat_type",
        "chat_id",
        "thread_id",
        "parent_chat_id",
        "guild_id",
        "scope_key",
        "scope_kind",
    } <= cols


def test_session_db_reconciles_scope_columns_on_existing_db(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (10);
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT,
            pricing_version TEXT,
            title TEXT,
            api_call_count INTEGER DEFAULT 0,
            handoff_state TEXT,
            handoff_platform TEXT,
            handoff_error TEXT,
            FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
        );
        """
    )
    conn.close()

    db = SessionDB(db_path)

    cols = {
        row[1]
        for row in db._conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    assert {
        "session_key",
        "chat_type",
        "chat_id",
        "thread_id",
        "parent_chat_id",
        "guild_id",
        "scope_key",
        "scope_kind",
    } <= cols


def test_session_db_create_session_persists_scope_metadata(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")

    db.create_session(
        "session-1",
        source="discord",
        user_id="user-1",
        session_key="agent:main:discord:thread:thread-456",
        chat_type="thread",
        chat_id="thread-456",
        thread_id="thread-456",
        parent_chat_id="parent-123",
        guild_id="guild-1",
        scope_key="discord:thread:thread-456",
        scope_kind="current_thread",
    )

    row = db.get_session("session-1")
    assert row["session_key"] == "agent:main:discord:thread:thread-456"
    assert row["chat_type"] == "thread"
    assert row["chat_id"] == "thread-456"
    assert row["thread_id"] == "thread-456"
    assert row["parent_chat_id"] == "parent-123"
    assert row["guild_id"] == "guild-1"
    assert row["scope_key"] == "discord:thread:thread-456"
    assert row["scope_kind"] == "current_thread"


def test_session_db_create_session_inherits_scope_metadata_from_parent(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")

    db.create_session(
        "parent-session",
        source="discord",
        user_id="user-1",
        session_key="agent:main:discord:thread:thread-456",
        chat_type="thread",
        chat_id="thread-456",
        thread_id="thread-456",
        parent_chat_id="parent-123",
        guild_id="guild-1",
        scope_key="discord:thread:thread-456",
        scope_kind="current_thread",
    )
    db.create_session(
        "compression-child",
        source="discord",
        parent_session_id="parent-session",
    )

    row = db.get_session("compression-child")
    assert row["user_id"] == "user-1"
    assert row["session_key"] == "agent:main:discord:thread:thread-456"
    assert row["chat_type"] == "thread"
    assert row["chat_id"] == "thread-456"
    assert row["thread_id"] == "thread-456"
    assert row["parent_chat_id"] == "parent-123"
    assert row["guild_id"] == "guild-1"
    assert row["scope_key"] == "discord:thread:thread-456"
    assert row["scope_kind"] == "current_thread"


def test_session_db_create_session_inherits_scope_metadata_from_nearest_scoped_ancestor(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")

    db.create_session(
        "scoped-root",
        source="discord",
        user_id="user-1",
        session_key="agent:main:discord:thread:thread-456",
        chat_type="thread",
        chat_id="thread-456",
        thread_id="thread-456",
        parent_chat_id="parent-123",
        guild_id="guild-1",
        scope_key="discord:thread:thread-456",
        scope_kind="current_thread",
    )
    # Simulate a legacy compression child that was created before scope
    # inheritance existed, leaving every scope column NULL.
    db._conn.execute(
        """
        INSERT INTO sessions (id, source, parent_session_id, started_at)
        VALUES (?, ?, ?, ?)
        """,
        ("legacy-null-child", "discord", "scoped-root", 1.0),
    )
    db._conn.commit()

    db.create_session(
        "new-compression-child",
        source="discord",
        parent_session_id="legacy-null-child",
    )

    row = db.get_session("new-compression-child")
    assert row["parent_session_id"] == "legacy-null-child"
    assert row["user_id"] == "user-1"
    assert row["session_key"] == "agent:main:discord:thread:thread-456"
    assert row["chat_type"] == "thread"
    assert row["chat_id"] == "thread-456"
    assert row["thread_id"] == "thread-456"
    assert row["parent_chat_id"] == "parent-123"
    assert row["guild_id"] == "guild-1"
    assert row["scope_key"] == "discord:thread:thread-456"
    assert row["scope_kind"] == "current_thread"


def test_session_db_create_session_preserves_explicit_child_scope_metadata(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")

    db.create_session(
        "parent-session",
        source="discord",
        user_id="parent-user",
        session_key="agent:main:discord:thread:parent-thread",
        chat_type="thread",
        chat_id="parent-thread",
        thread_id="parent-thread",
        parent_chat_id="parent-channel",
        guild_id="parent-guild",
        scope_key="discord:thread:parent-thread",
        scope_kind="current_thread",
    )
    db.create_session(
        "child-session",
        source="discord",
        parent_session_id="parent-session",
        user_id="child-user",
        session_key="agent:main:discord:dm:child-dm",
        chat_type="dm",
        chat_id="child-dm",
        thread_id="child-thread",
        parent_chat_id="child-parent-channel",
        guild_id="child-guild",
        scope_key="discord:dm:child-dm",
        scope_kind="dm",
    )

    row = db.get_session("child-session")
    assert row["user_id"] == "child-user"
    assert row["session_key"] == "agent:main:discord:dm:child-dm"
    assert row["chat_type"] == "dm"
    assert row["chat_id"] == "child-dm"
    assert row["thread_id"] == "child-thread"
    assert row["parent_chat_id"] == "child-parent-channel"
    assert row["guild_id"] == "child-guild"
    assert row["scope_key"] == "discord:dm:child-dm"
    assert row["scope_kind"] == "dm"


def _make_thread_store(tmp_path) -> tuple[SessionStore, SessionDB, SessionSource]:
    db = SessionDB(tmp_path / "state.db")
    store = SessionStore(
        tmp_path / "sessions",
        GatewayConfig(sessions_dir=tmp_path / "sessions"),
    )
    store._db = db
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-456",
        chat_name="dh Studio / #비서실 / 쓰레드 격리",
        chat_type="thread",
        user_id="user-1",
        thread_id="thread-456",
        parent_chat_id="parent-123",
        guild_id="guild-1",
    )
    return store, db, source


def _assert_thread_scope_row(row, session_key: str) -> None:
    assert row["session_key"] == session_key
    assert row["chat_type"] == "thread"
    assert row["chat_id"] == "thread-456"
    assert row["thread_id"] == "thread-456"
    assert row["parent_chat_id"] == "parent-123"
    assert row["guild_id"] == "guild-1"
    assert row["scope_key"] == "discord:thread:thread-456"
    assert row["scope_kind"] == "current_thread"


def test_session_store_persists_scope_metadata_for_discord_thread(tmp_path) -> None:
    store, db, source = _make_thread_store(tmp_path)

    entry = store.get_or_create_session(source)

    row = db.get_session(entry.session_id)
    _assert_thread_scope_row(row, entry.session_key)


def test_session_store_persists_scope_metadata_after_reset(tmp_path) -> None:
    store, db, source = _make_thread_store(tmp_path)
    entry = store.get_or_create_session(source)

    reset_entry = store.reset_session(entry.session_key)

    row = db.get_session(reset_entry.session_id)
    _assert_thread_scope_row(row, reset_entry.session_key)
