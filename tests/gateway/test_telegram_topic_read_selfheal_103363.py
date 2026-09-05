"""Issue #103363 — reading pre-v3 telegram topic tables self-heals to v3.

Upgrades that enabled DM topic mode before #76423 keep v2 tables (no
``profile_name`` column). Reads used to swallow ``no such column`` as an empty
result, so auto topic-rename silently died on existing installs; the write-only
migration was never reached. Reads now heal the table once and retry.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_state import SessionDB


CHAT = "208214988"


def _create_v2_state(db_path: Path) -> None:
    """The schema an install predating #76423 has on disk after upgrading."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        f"""
        CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO state_meta(key, value) VALUES ('telegram_dm_topic_schema_version', '2');
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
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            managed_mode TEXT NOT NULL DEFAULT 'auto',
            linked_at REAL NOT NULL, updated_at REAL NOT NULL,
            PRIMARY KEY (chat_id, thread_id)
        );
        INSERT INTO telegram_dm_topic_bindings
            VALUES ('{CHAT}', '99', '{CHAT}', 'k', 'legacy-sess', 'auto', 1.0, 1.0);
        """
    )
    conn.close()


def test_topic_reads_selfheal_pre_v3_tables(tmp_path: Path):
    db_path = tmp_path / "v2-upgrade.db"
    _create_v2_state(db_path)
    db = SessionDB(db_path=db_path)

    # Reads on the un-migrated v2 tables return the legacy rows instead of
    # silently reading as empty, and land them in the 'default' namespace.
    assert db.is_telegram_topic_mode_enabled(
        chat_id=CHAT, user_id=CHAT, profile_name="default",
    )
    binding = db.get_telegram_topic_binding(chat_id=CHAT, thread_id="99", profile_name="default")
    assert binding is not None
    assert binding["session_id"] == "legacy-sess"
    assert db.get_meta("telegram_dm_topic_schema_version") == "3"
    # Profile isolation still holds after the heal.
    assert not db.is_telegram_topic_mode_enabled(
        chat_id=CHAT, user_id=CHAT, profile_name="coder",
    )
    assert db.get_telegram_topic_binding(chat_id=CHAT, thread_id="99", profile_name="coder") is None
    db.close()


def test_list_bindings_selfheals_pre_v3_table(tmp_path: Path):
    db_path = tmp_path / "v2-upgrade.db"
    _create_v2_state(db_path)
    db = SessionDB(db_path=db_path)

    rows = db.list_telegram_topic_bindings_for_chat(chat_id=CHAT, profile_name="default")
    assert [row["session_id"] for row in rows] == ["legacy-sess"]
    assert db.get_meta("telegram_dm_topic_schema_version") == "3"
    db.close()


def test_absent_tables_still_read_empty_and_are_not_created(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "fresh.db")

    assert not db.is_telegram_topic_mode_enabled(chat_id=CHAT, user_id=CHAT)
    assert db.get_telegram_topic_binding(chat_id=CHAT, thread_id="99") is None
    assert db.list_telegram_topic_bindings_for_chat(chat_id=CHAT) == []
    # The deliberate "reads never create the tables" contract is preserved.
    tables = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "telegram_dm_topic_bindings" not in tables
    assert "telegram_dm_topic_mode" not in tables
    db.close()
