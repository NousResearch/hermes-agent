"""Regression for #103363: reading pre-v3 telegram topic tables self-heals to v3.

Upgrades that enabled DM topic mode before #76423 keep v2 tables (no
``profile_name`` column). Reads used to swallow ``no such column`` as an empty
result, so auto topic-rename silently died on existing installs; the write-only
migration was never reached. Reads now heal the table once and retry.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from persistence import ProfileScope
from persistence.sqlite import SQLiteBackend


CHAT = "208214988"


@contextlib.contextmanager
def _store(db_path: Path):
    """A writable SQLite store on *db_path*, closed on exit."""
    scope = ProfileScope(profile_dir=db_path.parent, name=db_path.parent.name)
    backend = SQLiteBackend({scope.hermes_home_key: db_path})
    try:
        yield backend.open(scope)
    finally:
        backend.close()


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
        -- v1 had no ON DELETE CASCADE: a pruned session leaves an orphan binding behind.
        INSERT INTO telegram_dm_topic_bindings
            VALUES ('{CHAT}', '7', '{CHAT}', 'k7', 'pruned-sess', 'auto', 1.0, 1.0);
        -- an older build's executescript rebuild crashed after CREATE: the scratch table is stranded.
        CREATE TABLE telegram_dm_topic_mode_new (profile_name TEXT, chat_id TEXT);
        """
    )
    conn.close()


def test_topic_reads_selfheal_pre_v3_tables(tmp_path: Path):
    db_path = tmp_path / "v2-upgrade.db"
    _create_v2_state(db_path)
    with _store(db_path) as store:
        routing = store.routing

        # Reads on the un-migrated v2 tables return the legacy rows instead of
        # silently reading as empty, and land them in the 'default' namespace.
        assert routing.is_topic_mode_enabled(
            chat_id=CHAT, user_id=CHAT, profile_name="default",
        )
        binding = routing.get_topic_binding(chat_id=CHAT, thread_id="99", profile_name="default")
        assert binding is not None
        assert binding.session_id == "legacy-sess"
        # The orphan row is dropped (a v2/v3 CASCADE would have removed it) and the stranded scratch table is gone.
        rows = routing.list_topic_bindings_for_chat(chat_id=CHAT, profile_name="default")
        assert [row.session_id for row in rows] == ["legacy-sess"]
        assert routing.get_topic_binding(chat_id=CHAT, thread_id="7", profile_name="default") is None
        assert store.holder().get_meta("telegram_dm_topic_schema_version") == "3"
        # Profile isolation still holds after the heal.
        assert not routing.is_topic_mode_enabled(
            chat_id=CHAT, user_id=CHAT, profile_name="coder",
        )
        assert routing.get_topic_binding(chat_id=CHAT, thread_id="99", profile_name="coder") is None


def test_absent_tables_still_read_empty_and_are_not_created(tmp_path: Path):
    with _store(tmp_path / "fresh.db") as store:
        routing = store.routing

        assert not routing.is_topic_mode_enabled(chat_id=CHAT, user_id=CHAT)
        assert routing.get_topic_binding(chat_id=CHAT, thread_id="99") is None
        assert list(routing.list_topic_bindings_for_chat(chat_id=CHAT)) == []
        # The deliberate "reads never create the tables" contract is preserved: the migration is the only
        # creator and always stamps the schema version in the same transaction.
        assert store.holder().get_meta("telegram_dm_topic_schema_version") is None


def test_heal_rebuild_rolls_back_atomically(tmp_path: Path):
    """The v2→v3 rebuild must stay inside _execute_write's transaction: a mid-rebuild
    failure leaves the intact v2 table (and its version stamp) for the next read to heal,
    never a half-built v3 table the retry would mistake for a finished one (#42004)."""
    db_path = tmp_path / "v2-upgrade.db"
    _create_v2_state(db_path)
    with _store(db_path) as store:
        holder = store.holder()

        def deny_topic_rebuild_copy(action, arg1, arg2, db_name, trigger):
            # Fail the copy into the fresh table, after the live table was renamed away: a
            # non-transactional rebuild leaves an empty v3 table behind and the retry skips it.
            if action == sqlite3.SQLITE_INSERT and arg1 == "telegram_dm_topic_mode":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        holder._conn.set_authorizer(deny_topic_rebuild_copy)
        try:
            # The guarded read degrades to "off" instead of raising...
            assert not store.routing.is_topic_mode_enabled(
                chat_id=CHAT, user_id=CHAT, profile_name="default",
            )
        finally:
            holder._conn.set_authorizer(None)

        # ...and the interrupted rebuild rolled back completely: the renamed-away original is back
        # (otherwise the retry below finds no legacy table, builds an empty v3 one and reads as off).
        assert holder.get_meta("telegram_dm_topic_schema_version") == "2"
        # Idempotent under retry (the _execute_write contract): the next clean
        # read heals and keeps the legacy rows.
        assert store.routing.is_topic_mode_enabled(
            chat_id=CHAT, user_id=CHAT, profile_name="default",
        )
