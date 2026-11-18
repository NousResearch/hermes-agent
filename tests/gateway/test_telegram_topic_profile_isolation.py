"""Issue #76423: the store's telegram topic tables namespace by profile."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


CHAT = "208214988"


def _session(db, sid, profile_name=None):
    seed_session(db, sid, "telegram", user_id=CHAT, profile_name=profile_name)


def test_legacy_rows_migrate_only_to_default(tmp_path: Path):
    """v1 shape (no CASCADE, old user index) → v3: rows land in 'default' only."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        f"""
        CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO state_meta(key, value) VALUES ('telegram_dm_topic_schema_version', '1');
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT, user_id TEXT, model TEXT,
            model_config TEXT, system_prompt TEXT, parent_session_id TEXT,
            started_at REAL, ended_at REAL, end_reason TEXT,
            message_count INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0
        );
        INSERT INTO sessions(id, source, user_id, started_at)
            VALUES ('legacy-sess', 'telegram', '{CHAT}', 1.0);
        CREATE TABLE telegram_dm_topic_mode (
            chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            activated_at REAL NOT NULL, updated_at REAL NOT NULL,
            has_topics_enabled INTEGER, allows_users_to_create_topics INTEGER,
            capability_checked_at REAL, intro_message_id TEXT, pinned_message_id TEXT
        );
        INSERT INTO telegram_dm_topic_mode(chat_id, user_id, enabled, activated_at, updated_at)
            VALUES ('{CHAT}', '{CHAT}', 1, 1.0, 1.0);
        CREATE TABLE telegram_dm_topic_bindings (
            chat_id TEXT NOT NULL, thread_id TEXT NOT NULL, user_id TEXT NOT NULL,
            session_key TEXT NOT NULL,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            managed_mode TEXT NOT NULL DEFAULT 'auto',
            linked_at REAL NOT NULL, updated_at REAL NOT NULL,
            PRIMARY KEY (chat_id, thread_id)
        );
        CREATE INDEX idx_telegram_dm_topic_bindings_user
            ON telegram_dm_topic_bindings(user_id, chat_id);
        INSERT INTO telegram_dm_topic_bindings
            VALUES ('{CHAT}', '99', '{CHAT}', 'k', 'legacy-sess', 'auto', 1.0, 1.0);
        """
    )
    conn.close()

    db = storage_handle(tmp_path)
    db.routing.prepare_topic_storage()
    assert db.state_meta.get("telegram_dm_topic_schema_version") == "3"
    assert db.routing.is_topic_mode_enabled(
        chat_id=CHAT, user_id=CHAT, profile_name="default",
    )
    assert not db.routing.is_topic_mode_enabled(
        chat_id=CHAT, user_id=CHAT, profile_name="coder",
    )
    assert db.routing.get_topic_binding(
        chat_id=CHAT, thread_id="99", profile_name="default",
    ).session_id == "legacy-sess"
    assert db.routing.get_topic_binding(
        chat_id=CHAT, thread_id="99", profile_name="coder",
    ) is None
    with raw_holder(db)._read_ctx() as conn:
        fk = conn.execute("PRAGMA foreign_key_list('telegram_dm_topic_bindings')").fetchall()
    assert any(row[2] == "sessions" and row[6] == "CASCADE" for row in fk)


def test_mode_and_bindings_isolated_across_profiles(tmp_path: Path):
    db = storage_handle(tmp_path)
    _session(db, "sess-a", "alpha")
    _session(db, "sess-b", "beta")

    db.routing.enable_topic_mode(chat_id=CHAT, user_id=CHAT, profile_name="alpha")
    db.routing.enable_topic_mode(chat_id=CHAT, user_id=CHAT, profile_name="beta")
    db.routing.disable_topic_mode(chat_id=CHAT, profile_name="alpha")
    assert not db.routing.is_topic_mode_enabled(chat_id=CHAT, user_id=CHAT, profile_name="alpha")
    assert db.routing.is_topic_mode_enabled(chat_id=CHAT, user_id=CHAT, profile_name="beta")

    db.routing.bind_topic(
        chat_id=CHAT, thread_id="77", user_id=CHAT,
        session_key="ka", session_id="sess-a", profile_name="alpha",
    )
    db.routing.bind_topic(
        chat_id=CHAT, thread_id="77", user_id=CHAT,
        session_key="kb", session_id="sess-b", profile_name="beta",
    )
    assert db.routing.get_topic_binding(
        chat_id=CHAT, thread_id="77", profile_name="alpha",
    ).session_id == "sess-a"
    assert db.routing.get_topic_binding(
        chat_id=CHAT, thread_id="77", profile_name="beta",
    ).session_id == "sess-b"

    assert db.routing.delete_topic_binding(
        chat_id=CHAT, thread_id="77", profile_name="alpha",
    ) == 1
    assert db.routing.get_topic_binding(
        chat_id=CHAT, thread_id="77", profile_name="alpha",
    ) is None
    assert db.routing.get_topic_binding(
        chat_id=CHAT, thread_id="77", profile_name="beta",
    ) is not None
    # Omitted kwarg == the single-profile "default" namespace, not a wildcard.
    assert not db.routing.is_topic_mode_enabled(chat_id=CHAT, user_id=CHAT)
    assert db.routing.get_topic_binding(chat_id=CHAT, thread_id="77") is None
