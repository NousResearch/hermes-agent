"""Legacy store shapes for the upgrade oracle, built by main's own SessionDB, then regressed with raw SQL to the
shape each migration step consumes.
"""

import json
import os
import re
import sqlite3
from pathlib import Path

from hermes_state import SessionDB
from hermes_state_common import (
    DEFERRED_INDEX_SQL, FTS_SQL, FTS_STORAGE_VERSION, LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL, SCHEMA_VERSION,
)
from hermes_state_schema import _SESSION_MODEL_USAGE_HEAL_DDL, SessionSchemaMixin

_TOOL_CALL = [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]
_PROMPT = "You are a careful assistant."
_FTS_TRIGGERS = ("messages_fts_insert", "messages_fts_delete", "messages_fts_update")
_TRIGRAM_TRIGGERS = ("messages_fts_trigram_insert", "messages_fts_trigram_delete", "messages_fts_trigram_update")
_CJK_TRIGGERS = ("messages_fts_cjk_insert", "messages_fts_cjk_delete", "messages_fts_cjk_update")

# The 5-column session_model_usage key installs carried before ``task``.
_USAGE_5_COLUMN_PK_SQL = """CREATE TABLE session_model_usage (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    billing_provider TEXT NOT NULL DEFAULT '',
    billing_base_url TEXT NOT NULL DEFAULT '',
    billing_mode TEXT NOT NULL DEFAULT '',
    api_call_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL DEFAULT 0,
    cost_status TEXT,
    cost_source TEXT,
    first_seen REAL,
    last_seen REAL,
    PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode)
)"""


def _raw(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path, isolation_level=None)


def _seed(path: Path) -> sqlite3.Connection:
    """A store current under main with one row of every kind the ladder rewrites, returned as a raw connection."""
    db = SessionDB(db_path=path)
    try:
        db.create_session("cli-1", "cli", model="seed-model", system_prompt=_PROMPT)
        db.append_message("cli-1", "user", content="deploy the payment service")
        db.append_message("cli-1", "assistant", content="running", tool_calls=_TOOL_CALL)
        db.append_message("cli-1", "tool", content="payment service deployed", tool_name="terminal", tool_call_id="c1")
        db.create_session("child-1", "cli", parent_session_id="cli-1", system_prompt=_PROMPT)
        db.append_message("child-1", "user", content="subagent payment audit")
        db.create_session("orphan-1", "cli")
        db.append_message("orphan-1", "tool", content="orphaned payment tool output", tool_name="terminal")
        db.create_session("cron-1", "cron", model="seed-model")
        db.append_message("cron-1", "user", content="定时迁移内容 payment report")
        db.create_session(
            "tg-1", "telegram", session_key="agent:main:telegram:dm:42", chat_id="42", chat_type="dm", user_id="42",
        )
        db.append_message("tg-1", "user", content="交互迁移内容 hello")
    finally:
        db.close()
    conn = _raw(path)
    conn.execute("UPDATE sessions SET input_tokens = 120, output_tokens = 30, api_call_count = 2 WHERE id = 'cli-1'")
    conn.execute(
        "INSERT INTO session_model_usage (session_id, model, task, api_call_count, input_tokens, output_tokens, "
        "first_seen, last_seen) VALUES ('cli-1', 'seed-model', '', 2, 120, 30, 1.0, 2.0)"
    )
    return conn


def _stamp(conn: sqlite3.Connection, version: int) -> None:
    conn.execute("UPDATE schema_version SET version = ?", (version,))
    conn.close()


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _drop(conn: sqlite3.Connection, kind: str, names) -> None:
    for name in names:
        conn.execute(f"DROP {kind} IF EXISTS {name}")


def _install_legacy_inline_fts(conn: sqlite3.Connection) -> None:
    """The pre-v23 inline layout, fts_storage_version marker absent, filled the way main refills it."""
    _drop(conn, "TRIGGER", _FTS_TRIGGERS + _TRIGRAM_TRIGGERS + _CJK_TRIGGERS)
    conn.executescript(
        "DROP TABLE IF EXISTS messages_fts_cjk; DROP TABLE IF EXISTS messages_fts_trigram;"
        "DROP VIEW IF EXISTS messages_fts_trigram_src; DROP TABLE messages_fts; DROP VIEW IF EXISTS messages_fts_src;"
        + LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL
    )
    SessionSchemaMixin._rebuild_fts_indexes(conn.cursor(), legacy=True)
    conn.execute("DELETE FROM state_meta WHERE key = 'fts_storage_version'")


def _install_raw_messages_base(conn: sqlite3.Connection) -> None:
    """The pre-v3 base index over raw ``messages``."""
    _drop(conn, "TRIGGER", _FTS_TRIGGERS)
    conn.executescript(
        """
        DROP TABLE messages_fts;
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content, tool_name, tool_calls, content='messages', content_rowid='id'
        );
        INSERT INTO messages_fts(messages_fts) VALUES('rebuild');
        """
    )


def _drop_deferred_objects(conn: sqlite3.Connection) -> None:
    """Every index and trigger _init_schema creates only after column reconciliation."""
    for kind, name in re.findall(r"CREATE (?:UNIQUE )?(INDEX|TRIGGER) IF NOT EXISTS (\w+)", DEFERRED_INDEX_SQL):
        conn.execute(f"DROP {kind} IF EXISTS {name}")
    _drop(conn, "INDEX", ("idx_messages_platform_msg_id", "idx_sessions_title_unique"))


def absent(path: Path) -> None:
    """No file: the first writable open takes the fresh-store branch."""


def current(path: Path) -> None:
    """A store at main's SCHEMA_VERSION with data: every gate false, only the per-open steps run."""
    _seed(path).close()


def v15(path: Path) -> None:
    """< 16: delegate and tool-bearing orphan rows without ``_delegate_from``."""
    conn = _seed(path)
    conn.execute(
        "UPDATE sessions SET model_config = json_remove(model_config, '$._delegate_from') WHERE model_config IS NOT NULL"
    )
    _stamp(conn, 15)


def v17(path: Path) -> None:
    """< 18: gateway metadata only in ``<home>/sessions/sessions.json``."""
    conn = _seed(path)
    conn.execute("UPDATE sessions SET display_name = NULL, origin_json = NULL, expiry_finalized = 0 WHERE id = 'tg-1'")
    sessions_dir = Path(os.environ["HERMES_HOME"]) / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sessions.json").write_text(json.dumps({
        "agent:main:telegram:dm:42": {
            "session_id": "tg-1", "session_key": "agent:main:telegram:dm:42", "display_name": "Alice",
            "chat_type": "dm", "origin": {"platform": "telegram", "chat_id": "42", "thread_id": None},
            "expiry_finalized": True,
        },
        "_meta": {"ignored": True},
    }))
    _stamp(conn, 17)


def v19(path: Path) -> None:
    """< 20: per-model usage only in the sessions aggregates."""
    conn = _seed(path)
    conn.execute("DELETE FROM session_model_usage")
    _stamp(conn, 19)


def v21(path: Path) -> None:
    """< 22: session_model_usage without ``task`` (the per-open PK heal reaches it first)."""
    conn = _seed(path)
    conn.execute("DROP TABLE session_model_usage")
    conn.execute(_USAGE_5_COLUMN_PK_SQL)
    conn.execute(
        "INSERT INTO session_model_usage (session_id, model, input_tokens, output_tokens) "
        "VALUES ('cli-1', 'seed-model', 120, 30)"
    )
    _stamp(conn, 21)


def v22(path: Path) -> None:
    """< 23: legacy inline FTS, which only earns the ``fts_optimize_available`` flag."""
    conn = _seed(path)
    _install_legacy_inline_fts(conn)
    _stamp(conn, 22)


def v24(path: Path) -> None:
    """< 25: system prompts inline on the session rows."""
    conn = _seed(path)
    conn.execute(
        "UPDATE sessions SET system_prompt = (SELECT prompt FROM system_prompts p WHERE p.hash = "
        "sessions.system_prompt_hash), system_prompt_hash = NULL WHERE system_prompt_hash IS NOT NULL"
    )
    conn.execute("DELETE FROM system_prompts")
    _stamp(conn, 24)


def v29(path: Path) -> None:
    """< 30: the pre-cron-exclusion trigram view with only its insert trigger, indexing cron rows."""
    conn = _seed(path)
    conn.executescript(
        """
        DROP TRIGGER messages_fts_trigram_insert;
        DROP TRIGGER messages_fts_trigram_delete;
        DROP TRIGGER messages_fts_trigram_update;
        DROP TABLE messages_fts_trigram;
        DROP VIEW messages_fts_trigram_src;
        CREATE VIEW messages_fts_trigram_src AS
            SELECT id, role, content, tool_name, tool_calls
            FROM messages WHERE role <> 'tool';
        CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
            content, tool_name,
            content='messages_fts_trigram_src',
            content_rowid='id',
            tokenize='trigram'
        );
        CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages
        WHEN new.role <> 'tool'
        BEGIN
            INSERT INTO messages_fts_trigram(rowid, content, tool_name)
            VALUES (new.id, new.content, new.tool_name);
        END;
        INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild');
        """
    )
    _stamp(conn, 29)


def v30(path: Path) -> None:
    """30, before the runtime ledger tables: SCHEMA_SQL recreates them and the ladder stamps 32 with no data
    step."""
    conn = _seed(path)
    _drop(conn, "TABLE", ("worker_receipts", "worker_executions", "session_admissions", "runtime_epoch"))
    _stamp(conn, 30)


def v31(path: Path) -> None:
    """31: v32 runs nothing, only the stamp moves."""
    _stamp(_seed(path), 31)


def usage_stale_pk(path: Path) -> None:
    """32 with the 5-column usage key and ``task`` appended outside it, so only the per-open heal can fix it."""
    conn = _seed(path)
    conn.execute("DROP TABLE session_model_usage")
    conn.execute(_USAGE_5_COLUMN_PK_SQL)
    conn.execute('ALTER TABLE session_model_usage ADD COLUMN "task" TEXT')
    conn.execute(
        "INSERT INTO session_model_usage (session_id, model, input_tokens, output_tokens) "
        "VALUES ('cli-1', 'seed-model', 120, 30)"
    )
    conn.close()


def usage_heal_interrupted(path: Path) -> None:
    """The PK heal's rebuild cut between RENAME and copy-back (the interleaving _rebuild_table's single
    transaction closes): rows stranded in session_model_usage_legacy_pk, live table empty."""
    conn = _seed(path)
    conn.execute("ALTER TABLE session_model_usage RENAME TO session_model_usage_legacy_pk")
    conn.execute(_SESSION_MODEL_USAGE_HEAL_DDL)
    conn.close()


def routing_legacy_pk(path: Path) -> None:
    """gateway_routing keyed by session_key alone, before ``scope``."""
    conn = _seed(path)
    conn.execute("DROP TABLE gateway_routing")
    conn.execute(
        "CREATE TABLE gateway_routing (session_key TEXT PRIMARY KEY, entry_json TEXT NOT NULL, updated_at REAL NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO gateway_routing VALUES (?, ?, ?)",
        [("agent:main:telegram:dm:42", '{"session_id": "tg-1"}', 2.0), ("agent:main:cli", '{"session_id": "cli-1"}', 1.0)],
    )
    conn.close()


def deferred_missing(path: Path) -> None:
    """Columns present, every deferred index and trigger missing."""
    conn = _seed(path)
    _drop_deferred_objects(conn)
    conn.close()


def columns_missing(path: Path) -> None:
    """Declared columns and tables a lagging store lacks, so reconciliation adds them before the deferred DDL;
    columns dropped through SQLite's table rewrite."""
    conn = _seed(path)
    _drop_deferred_objects(conn)
    for table, columns in (
        ("sessions", ("transport_profile", "tool_names", "last_read_at", "created_source", "pinned")),
        ("messages", ("display_order", "display_identity", "message_uid", "tool_call_uid", "absorbed_message_uids")),
    ):
        for column in columns:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    _drop(conn, "TABLE", ("async_delegations", "gateway_heartbeats", "conversation_generations"))
    conn.close()


def newer_v33(path: Path) -> None:
    """A newer release's store: schema_version 33, an unrecognised base FTS definition stamped
    FTS_STORAGE_VERSION + 1, a table and a column this code does not declare."""
    conn = _seed(path)
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_fts'").fetchone()[0]
    _drop(conn, "TRIGGER", _FTS_TRIGGERS)
    conn.execute("DROP TABLE messages_fts")
    conn.execute(sql.replace("content_rowid='id'", "content_rowid='id', tokenize='porter unicode61'"))
    conn.executescript(FTS_SQL)
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    _set_meta(conn, "fts_storage_version", str(FTS_STORAGE_VERSION + 1))
    conn.execute("CREATE TABLE future_feature (id INTEGER PRIMARY KEY, payload TEXT)")
    conn.execute("INSERT INTO future_feature (payload) VALUES ('from a newer release')")
    conn.execute("ALTER TABLE sessions ADD COLUMN future_col TEXT")
    conn.execute("UPDATE sessions SET future_col = 'newer' WHERE id = 'cli-1'")
    _stamp(conn, SCHEMA_VERSION + 1)


def lazy_tables(path: Path) -> None:
    """The feature-gated tables outside the ladder, present with rows: the Telegram topic tables,
    delivery_obligations and an async_delegations row (SCHEMA_SQL, written by tools/async_delegation.py)."""
    _seed(path).close()
    db = SessionDB(db_path=path)
    try:
        db.apply_telegram_topic_migration()
        db.enable_telegram_topic_mode(chat_id="42", user_id="42")
    finally:
        db.close()
    from gateway.delivery_ledger import _initialize_schema

    conn = _raw(path)
    _initialize_schema(conn)
    conn.execute(
        "INSERT INTO delivery_obligations (obligation_id, session_key, platform, chat_id, content, state, created_at, "
        "updated_at) VALUES ('ob-1', 'agent:main:telegram:dm:42', 'telegram', '42', 'pending reply', 'pending', 1.0, 1.0)"
    )
    conn.execute(
        "INSERT INTO telegram_dm_topic_bindings (profile_name, chat_id, thread_id, user_id, session_key, session_id, "
        "managed_mode, linked_at, updated_at) VALUES ('default', '42', '99', '42', 'agent:main:telegram:dm:42:99', "
        "'tg-1', 'auto', 1.0, 1.0)"
    )
    conn.execute(
        "INSERT INTO async_delegations (delegation_id, origin_session, state, dispatched_at, updated_at) "
        "VALUES ('del-1', 'cli-1', 'running', 1.0, 1.0)"
    )
    conn.close()


def telegram_topic_v2(path: Path) -> None:
    """Pre-v3 topic tables with an orphan binding and a stranded scratch table; open leaves them for the
    explicit migration."""
    conn = _seed(path)
    conn.executescript(
        """
        CREATE TABLE telegram_dm_topic_mode (
            chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            activated_at REAL NOT NULL, updated_at REAL NOT NULL,
            has_topics_enabled INTEGER, allows_users_to_create_topics INTEGER,
            capability_checked_at REAL, intro_message_id TEXT, pinned_message_id TEXT
        );
        INSERT INTO telegram_dm_topic_mode(chat_id, user_id, enabled, activated_at, updated_at)
            VALUES ('42', '42', 1, 1.0, 1.0);
        CREATE TABLE telegram_dm_topic_bindings (
            chat_id TEXT NOT NULL, thread_id TEXT NOT NULL, user_id TEXT NOT NULL,
            session_key TEXT NOT NULL,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            managed_mode TEXT NOT NULL DEFAULT 'auto',
            linked_at REAL NOT NULL, updated_at REAL NOT NULL,
            PRIMARY KEY (chat_id, thread_id)
        );
        INSERT INTO telegram_dm_topic_bindings VALUES ('42', '99', '42', 'k', 'tg-1', 'auto', 1.0, 1.0);
        INSERT INTO telegram_dm_topic_bindings VALUES ('42', '7', '42', 'k7', 'pruned-sess', 'auto', 1.0, 1.0);
        CREATE TABLE telegram_dm_topic_mode_new (profile_name TEXT, chat_id TEXT);
        """
    )
    _set_meta(conn, "telegram_dm_topic_schema_version", "2")
    conn.close()


def fts_stale(path: Path) -> None:
    """A deferred rebuild's breadcrumb: ``fts_stale`` set, every FTS trigger dropped, and a row written while
    detached, which an open recovers."""
    conn = _seed(path)
    _set_meta(conn, "fts_stale", "1")
    _drop(conn, "TRIGGER", _FTS_TRIGGERS + _TRIGRAM_TRIGGERS + _CJK_TRIGGERS)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, message_uid) "
        "VALUES ('cli-1', 'user', 'written while payment search was detached', 3.0, 'feedfacefeedfacefeedfacefeedface')"
    )
    conn.close()


def fts_v0_inline(path: Path) -> None:
    """Layout 0 at SCHEMA_VERSION: legacy inline FTS, marker absent; open keeps it."""
    conn = _seed(path)
    _install_legacy_inline_fts(conn)
    conn.close()


def fts_v1(path: Path) -> None:
    """Layout 1: base index over raw ``messages`` and the trigram projecting ``tool_calls``, as an open finds it before
    realigning it."""
    conn = _seed(path)
    conn.executescript(
        """
        DROP TRIGGER messages_fts_trigram_insert;
        DROP TRIGGER messages_fts_trigram_delete;
        DROP TRIGGER messages_fts_trigram_update;
        DROP TABLE messages_fts_trigram;
        DROP VIEW messages_fts_trigram_src;
        CREATE VIEW messages_fts_trigram_src AS
            SELECT id, role, content, tool_name, tool_calls FROM messages WHERE role <> 'tool';
        CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
            content, tool_name, tool_calls,
            content='messages_fts_trigram_src', content_rowid='id', tokenize='trigram'
        );
        INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild');
        """
    )
    _install_raw_messages_base(conn)
    _set_meta(conn, "fts_storage_version", "1")
    conn.close()


def fts_v2(path: Path) -> None:
    """Layout 2: base index over raw ``messages``, current trigram; open realigns and stamps 3."""
    conn = _seed(path)
    _install_raw_messages_base(conn)
    _set_meta(conn, "fts_storage_version", "2")
    conn.close()


def fts_orphan_shadow(path: Path) -> None:
    """A ``.recover`` image: the base vtable row gone, its shadow tables kept."""
    _seed(path).close()
    conn = _raw(path)
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute("DELETE FROM sqlite_master WHERE name = 'messages_fts' AND sql LIKE 'CREATE VIRTUAL TABLE%'")
    version = conn.execute("PRAGMA schema_version").fetchone()[0]
    conn.execute(f"PRAGMA schema_version={version + 1}")
    conn.execute("PRAGMA writable_schema=OFF")
    conn.close()


def fts_trigram_dropped(path: Path) -> None:
    """No trigram index, view or triggers (a store from a runtime without the trigram tokenizer, which only
    soft-fails that DDL); the next open creates and backfills it."""
    conn = _seed(path)
    _drop(conn, "TRIGGER", _TRIGRAM_TRIGGERS)
    conn.execute("DROP TABLE messages_fts_trigram")
    conn.execute("DROP VIEW messages_fts_trigram_src")
    conn.close()


def uid_backfill_partial(path: Path) -> None:
    """An interrupted message_uid backfill: rows past the cursor have no uid and the done marker is absent."""
    conn = _seed(path)
    conn.execute("UPDATE messages SET message_uid = NULL WHERE rowid > 2")
    conn.execute("DELETE FROM state_meta WHERE key = 'message_uid_backfill'")
    _set_meta(conn, "message_uid_backfill_id", "2")
    conn.close()


def optimize_interrupted(path: Path) -> None:
    """An optimize-storage backfill cut short: high-water markers set and the layout stamp unwritten, which the
    open-time stamp must leave alone."""
    conn = _seed(path)
    high = conn.execute("SELECT MAX(id) FROM messages").fetchone()[0]
    _set_meta(conn, "fts_rebuild_high_water", str(high))
    _set_meta(conn, "fts_rebuild_progress", "0")
    conn.execute("DELETE FROM state_meta WHERE key = 'fts_storage_version'")
    conn.close()


def jsonl_converted(path: Path) -> None:
    """A JSONL-converted store at 30: text ``id`` without a PRIMARY KEY, nullable columns, no other table."""
    conn = _raw(path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (30);
        CREATE TABLE messages (id TEXT, session_id TEXT, role TEXT, content TEXT, tool_call_id TEXT,
          tool_calls TEXT, tool_name TEXT, timestamp TEXT, token_count TEXT, finish_reason TEXT,
          reasoning TEXT, reasoning_content TEXT, reasoning_details TEXT, codex_reasoning_items TEXT,
          codex_message_items TEXT, platform_message_id TEXT, observed TEXT, active TEXT,
          compacted INTEGER NOT NULL DEFAULT 0, effect_disposition TEXT, api_content TEXT,
          display_kind TEXT, display_metadata TEXT, _compressed_summary INTEGER NOT NULL DEFAULT 0,
          display_identity BLOB, display_order INTEGER, message_uid TEXT,
          absorbed_message_uids TEXT, tool_call_uids TEXT, tool_call_uid TEXT);
        INSERT INTO messages (id, session_id, role, content, timestamp) VALUES
          ('11022', 's', 'user', 'hi', '1779587405.0'),
          ('11023', 's', 'assistant', 'hello payment', '1779587406.0'),
          ('5078', 's', 'user', 'older row whose text id sorts last', '1779587300.0');
        """
    )
    conn.close()


FIXTURES = {
    builder.__name__: builder for builder in (
        absent, current, v15, v17, v19, v21, v22, v24, v29, v30, v31, usage_stale_pk, usage_heal_interrupted,
        routing_legacy_pk, deferred_missing, columns_missing, newer_v33, lazy_tables, telegram_topic_v2, fts_stale,
        fts_v0_inline, fts_v1, fts_v2, fts_orphan_shadow, fts_trigram_dropped, uid_backfill_partial,
        optimize_interrupted, jsonl_converted,
    )
}
