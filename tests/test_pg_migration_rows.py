"""Migration fidelity through real SQLite I/O, independent of a live PG server.

Only target connection/schema setup and PostgreSQL locking/sequence functions
are substituted. Bindings, foreign keys, inserts, snapshots and verification use
actual databases. The live PostgreSQL smoke suite separately covers the driver.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest

import hermes_state_pg_import as pg_import
import hermes_state_pg_schema as pg_schema
import hermes_state_postgres as postgres
import migrate_state_to_postgres as migration
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL
from hermes_state_telegram import _TOPIC_TABLES


def _sqlite_target(monkeypatch, path, *, sequence=None):
    # Unlike table writes, PostgreSQL sequence changes survive rollback and are
    # shared by all connections. Keep their state outside the SQLite transaction.
    if sequence is None:
        sequence = {"last_value": None}

    class TargetConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "LOCK TABLE messages IN SHARE ROW EXCLUSIVE MODE":
                # SQLite has one database writer lock instead of PG table locks.
                # A no-op write acquires it without changing any imported row.
                return super().execute("UPDATE messages SET id=id WHERE 0")
            return super().execute(sql, parameters)

    def setval(_sequence, value, _called):
        sequence["last_value"] = value
        return value

    def connect(_dsn):
        conn = sqlite3.connect(path, factory=TargetConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.create_function("pg_get_serial_sequence", 2, lambda *_: "messages_id_seq")
        conn.create_function("pg_sequence_last_value", 1, lambda _: sequence["last_value"])
        conn.create_function("GREATEST", -1, max)
        conn.create_function("setval", 3, setval)
        return conn

    def initialize(conn, _version):
        conn.executescript(SCHEMA_SQL)

    def initialize_topics(conn):
        for name, _, ddl in _TOPIC_TABLES:
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{name}" ({ddl})')
        conn.commit()

    monkeypatch.setattr(postgres, "connect_postgres", connect)
    monkeypatch.setattr(pg_schema, "init_postgres_schema", initialize)
    monkeypatch.setattr(pg_schema, "init_postgres_topic_schema", initialize_topics)
    return connect


def _stored_rows(conn, table):
    return [dict(row) for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY 1')]


def test_migration_preserves_durable_rows_without_changing_the_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STATE_BACKEND", "sqlite")
    # URI metacharacters must be escaped when opening the read-only snapshot.
    source_path = tmp_path / "source #? state.db"
    target_path = tmp_path / "target.db"
    connect = _sqlite_target(monkeypatch, target_path)
    db = SessionDB(db_path=source_path)
    try:
        db.create_session("z-parent", "telegram", system_prompt="stable prompt prefix", session_key="peer",
                          model_config={"temperature": 0.3}, profile_name="work")
        db.create_session("a-child", "subagent", parent_session_id="z-parent",
                          model_config={"_delegate_from": "z-parent"})
        first = db.append_message("z-parent", "user", content=[{"type": "text", "text": "original request"}],
                                  display_kind="async_delegation_complete", display_metadata={"task_count": 2})
        hidden = db.append_message("z-parent", "assistant", content="rewound answer", _compressed_summary=True)
        db.append_message("a-child", "assistant", content="delegated work", api_content="wire content",
                          reasoning_details=[{"type": "reasoning", "text": "preserved"}])
        db.update_token_counts("z-parent", model="test-model", input_tokens=17, output_tokens=9,
                               billing_provider="provider", api_call_count=1, actual_cost_usd=0.001)
        db.save_gateway_routing_entry("peer", json.dumps({"session_id": "z-parent"}), scope="work")
        db.increment_hygiene_failure_streak("peer")
        db.enable_telegram_topic_mode(chat_id="42", user_id="42", profile_name="work")
        db.bind_telegram_topic(chat_id="42", thread_id="1", user_id="42", session_key="peer",
                               session_id="z-parent", profile_name="work")
        db.set_meta("goal:z-parent", "finish the task")
        db.set_meta("loop:z-parent", "repeat the check")
        db.set_meta("heartbeat:z-parent", "scheduled reminder")
    finally:
        db.close()

    with closing(sqlite3.connect(source_path)) as src:
        src.row_factory = sqlite3.Row
        src.execute("UPDATE messages SET active=0, compacted=1 WHERE id=?", (hidden,))
        raw = src.execute("SELECT content FROM messages WHERE id=?", (first,)).fetchone()[0]
        src.execute("UPDATE messages SET content=? WHERE id=?", (
            SessionDB._CONTENT_JSON_PREFIX_LEGACY + raw[len(SessionDB._CONTENT_JSON_PREFIX):], first,
        ))
        # Preserve display order; content conversion must invalidate its derived identity.
        src.execute("UPDATE messages SET display_identity=?, display_order=? WHERE id=?", (b"\x00identity", 7, first))
        src.execute("INSERT INTO conversation_generations VALUES ('telegram', 'peer', 7)")
        local_meta = {"db_file_generation", "store_instance_id", "store_created_at_utc", "last_vacuum",
                      "fts_storage_version", "fts_rebuild_progress"}
        src.executemany("INSERT OR REPLACE INTO state_meta VALUES (?, 'source-local')", ((key,) for key in local_meta))
        src.execute("INSERT INTO gateway_heartbeats VALUES ('old-backend', 123, 1, 2, 'work', 'host')")
        src.execute("INSERT INTO compression_locks VALUES ('z-parent', 'old-worker', 1, 99)")
        src.execute("INSERT INTO session_turn_leases VALUES ('z-parent', 'old-worker', 1, 99)")
        src.execute("INSERT INTO async_delegations (delegation_id, origin_session, state, dispatched_at, updated_at) "
                    "VALUES ('local-outbox', 'z-parent', 'pending', 1, 1)")
        src.commit()
        tables = ("system_prompts", "sessions", "messages", "session_model_usage", "gateway_routing",
                  "gateway_hygiene_state", "conversation_generations", *(name for name, _, _ in _TOPIC_TABLES))
        expected = {table: _stored_rows(src, table) for table in tables}
        expected_meta = {row["key"]: row["value"] for row in _stored_rows(src, "state_meta")
                         if row["key"] not in local_meta and not row["key"].startswith("fts_")}
        for row in expected["messages"]:
            row["content"] = SessionDB._encode_content(SessionDB._decode_content(row["content"]))
        next(row for row in expected["messages"] if row["id"] == first)["display_identity"] = None

    before_bytes = source_path.read_bytes()
    before_mtime = source_path.stat().st_mtime_ns
    # A second migration must preserve all rows and their keys without duplicates.
    for _ in range(2):
        summary = migration.migrate(source_path, "unused-test-dsn")
        assert summary["complete"], summary["field_check"]
        with closing(connect("unused-test-dsn")) as target:
            assert {table: _stored_rows(target, table) for table in tables} == expected
            assert dict(target.execute("SELECT key, value FROM state_meta")) == expected_meta
            for table in ("gateway_heartbeats", "compression_locks", "session_turn_leases", "async_delegations"):
                assert target.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0
        assert source_path.read_bytes() == before_bytes
        assert source_path.stat().st_mtime_ns == before_mtime

    # Reusing a target whose peer counter advanced must never lower that counter.
    with closing(connect("unused-test-dsn")) as target:
        target.execute("UPDATE conversation_generations SET generation=11")
        target.commit()
    assert migration.migrate(source_path, "unused-test-dsn")["complete"]
    with closing(connect("unused-test-dsn")) as target:
        assert target.execute("SELECT generation FROM conversation_generations").fetchone()[0] == 11


def test_migration_rebuilds_display_identity_after_legacy_content_normalization(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STATE_BACKEND", "sqlite")
    source_path = tmp_path / "source.db"
    target_path = tmp_path / "target.db"
    connect = _sqlite_target(monkeypatch, target_path)
    payload = [{"type": "text", "text": "structured request"}]
    with closing(SessionDB(db_path=source_path)) as source:
        source.create_session("session", "cli")
        message_id = source.append_message("session", "user", content=payload)
        source._conn.execute("UPDATE messages SET content=? WHERE id=?", (
            SessionDB._CONTENT_JSON_PREFIX_LEGACY + json.dumps(payload), message_id,
        ))
        source._conn.commit()
        # Exercise the real indexed display path on the legacy representation.
        assert source.get_messages("session", include_compacted=True)[0]["content"] == payload
        indexed = dict(source._conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone())
        assert indexed["display_identity"] is not None
        assert indexed["display_order"] is not None

    summary = migration.migrate(source_path, "unused-test-dsn")
    assert summary["complete"], summary["field_check"]
    with closing(connect("unused-test-dsn")) as target:
        pending = target.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        assert pending["content"] != indexed["content"]
        assert pending["display_identity"] is None
        assert pending["display_order"] == indexed["display_order"]

    with closing(SessionDB(db_path=target_path)) as target:
        assert target.get_messages("session", include_compacted=True)[0]["content"] == payload
        rebuilt = target._conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        assert rebuilt["display_identity"] == target._display_identity(target._display_dedupe_key(rebuilt))
        assert rebuilt["display_order"] == indexed["display_order"]


def test_migration_rejects_matching_counts_with_different_stored_content(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STATE_BACKEND", "sqlite")
    source_path = tmp_path / "source.db"
    target_path = tmp_path / "target.db"
    connect = _sqlite_target(monkeypatch, target_path)
    db = SessionDB(db_path=source_path)
    try:
        db.create_session("session", "cli")
        message_id = db.append_message("session", "user", content="original", _compressed_summary=True)
        db.save_gateway_routing_entry("peer", '{"session_id":"session"}')
    finally:
        db.close()

    with closing(sqlite3.connect(source_path)) as source, closing(connect("unused-test-dsn")) as target:
        source.backup(target)
        # The only changed message column is content: display triggers are removed
        # from this pre-existing target so they cannot manufacture other mismatches.
        for (trigger,) in target.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
            target.execute(f'DROP TRIGGER "{trigger}"')
        target.execute("UPDATE messages SET content='unrelated transcript' WHERE id=?", (message_id,))
        target.execute("UPDATE gateway_routing SET entry_json='{}'")
        target.commit()

    summary = migration.migrate(source_path, "unused-test-dsn")
    assert summary["migrated_sessions"] == summary["source_sessions"]
    assert summary["migrated_messages"] == summary["source_messages"]
    assert not summary["complete"]
    assert not summary["field_check"]["clean"]
    mismatches = summary["field_check"]["field_mismatches"]
    assert any(f"messages[{message_id}].content" in item for item in mismatches)
    assert any("gateway_routing" in item and ".entry_json" in item for item in mismatches)
    with closing(connect("unused-test-dsn")) as target:
        assert target.execute("SELECT content FROM messages WHERE id=?", (message_id,)).fetchone()[0] == "unrelated transcript"
        present = {row[0] for row in target.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not present.intersection(name for name, _, _ in _TOPIC_TABLES)


@pytest.mark.parametrize("target_ahead", [False, True])
def test_migration_never_rewinds_message_sequence_after_rollback(tmp_path, monkeypatch, target_ahead):
    monkeypatch.setenv("HERMES_STATE_BACKEND", "sqlite")
    source_path = tmp_path / "source.db"
    target_path = tmp_path / "target.db"
    with closing(SessionDB(db_path=source_path)) as source:
        source.create_session("session", "cli")
        kept_id = source.append_message("session", "user", content="keep")
        removed_id = source.append_message("session", "assistant", content="later removed")
    # Model sequence allocations from deleted or rolled-back target messages.
    allocated_before = removed_id + 100 if target_ahead else None
    sequence = {"last_value": allocated_before}
    connect = _sqlite_target(monkeypatch, target_path, sequence=sequence)
    advance = pg_import.advance_message_sequence

    def fail_before_commit(conn):
        advance(conn)
        raise RuntimeError("injected failure after sequence advancement")

    with monkeypatch.context() as fault:
        fault.setattr(pg_import, "advance_message_sequence", fail_before_commit)
        with pytest.raises(RuntimeError, match="injected failure"):
            migration.migrate(source_path, "unused-test-dsn")

    after_rollback = sequence["last_value"]
    assert after_rollback >= removed_id
    if allocated_before is not None:
        assert after_rollback >= allocated_before
    with closing(connect("unused-test-dsn")) as target:
        assert target.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert target.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    # A subsequent backup can omit the highest id. Retrying must not recycle the
    # value already consumed by the failed migration's nontransactional setval.
    with closing(sqlite3.connect(source_path)) as source:
        source.execute("DELETE FROM messages WHERE id=?", (removed_id,))
        source.commit()
    assert migration.migrate(source_path, "unused-test-dsn")["complete"]
    assert sequence["last_value"] >= after_rollback
    with closing(connect("unused-test-dsn")) as target:
        assert [row[0] for row in target.execute("SELECT id FROM messages")] == [kept_id]


def test_migration_acquires_writer_lock_before_import_and_releases_it_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STATE_BACKEND", "sqlite")
    source_path = tmp_path / "source.db"
    target_path = tmp_path / "target.db"
    with closing(SessionDB(db_path=source_path)) as source:
        source.create_session("session", "cli")
        source.append_message("session", "user", content="original")
    sequence = {"last_value": None}
    connect = _sqlite_target(monkeypatch, target_path, sequence=sequence)

    def fail_first_import(conn, *_args):
        # Observe exclusion from another real connection before the first row is
        # copied; acquiring the lock after INSERT would fail this invariant.
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        with closing(sqlite3.connect(target_path, timeout=0)) as writer:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.execute("UPDATE messages SET id=id WHERE 0")
        raise RuntimeError("injected failure before first import")

    monkeypatch.setattr(pg_import, "import_rows", fail_first_import)
    with pytest.raises(RuntimeError, match="injected failure"):
        migration.migrate(source_path, "unused-test-dsn")
    assert sequence["last_value"] is None
    with closing(connect("unused-test-dsn")) as target:
        assert target.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert target.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        target.execute("UPDATE messages SET id=id WHERE 0")
        target.commit()
