"""Shared message paths through the PostgreSQL adapter and real transactional rows.

The driver maps PostgreSQL catalog/index operations to SQLite storage. It
rejects SQLite-only catalog reads and executes the PostgreSQL trigger SQL bodies.
"""

import pytest

import hermes_state_postgres as pg
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL
from hermes_state_pg_triggers import install_postgres_message_triggers
from tests.pg_trigger_harness import TriggerDDL
from tests.test_pg_fts_transactions import _AbortingDriver, _Cursor


class _MessageCursor(_Cursor):
    def execute(self, sql, params=()):
        super().execute(sql, params)
        # psycopg buffers RETURNING results before execute returns; SQLite
        # otherwise keeps multi-row INSERTs pending and refuses a savepoint.
        self._rows = iter(self.result.fetchall()) if self.result.description else iter(())
        return self

    def fetchone(self):
        return next(self._rows, None)

    def fetchall(self):
        return list(self._rows)

    def executemany(self, sql, rows):
        for params in rows:
            self.execute(sql, params)
        return self


class _MessageDriver(_AbortingDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_ddl = TriggerDDL(self.db)

    def cursor(self):
        return _MessageCursor(self)

    def execute(self, sql, params=()):
        trigger_result = self.trigger_ddl.execute(sql)
        if trigger_result is not None:
            return trigger_result
        if sql.lstrip().upper().startswith("PRAGMA "):
            raise RuntimeError("PostgreSQL does not accept PRAGMA")
        if "pg_catalog.pg_attribute" in sql and "attname" in sql.split("FROM", 1)[0]:
            sql = "SELECT name FROM pragma_table_info('messages') ORDER BY cid"
        return super().execute(sql.replace(" ILIKE ", " LIKE "), params)


@pytest.fixture
def message_store(tmp_path, monkeypatch):
    raw = _MessageDriver(tmp_path / "rows.db")
    raw.db.execute("DROP TABLE messages")
    raw.db.executescript(SCHEMA_SQL)
    raw.db.execute("ALTER TABLE messages ADD COLUMN fts_content TEXT")
    raw.db.execute("ALTER TABLE messages ADD COLUMN clone_extension TEXT")
    raw.db.executemany(
        "INSERT INTO sessions (id, source, started_at) VALUES (?, 'test', 1)",
        [("parent",), ("child",)],
    )
    conn = pg._PostgresConnection(raw)
    install_postgres_message_triggers(conn)
    monkeypatch.setattr(pg, "maybe_open_postgres", lambda *args, **kwargs: conn)
    db = SessionDB(tmp_path / "nominal.db", postgres_dsn="postgresql://fixture/state")
    try:
        yield db, raw
    finally:
        db.close()


def _rich_tail(db, raw):
    caller = db.append_message(
        "parent", role="assistant", content="tool caller", timestamp=20,
        tool_calls=[{"id": "call-1", "type": "function",
                     "function": {"name": "terminal", "arguments": "{}"}}],
        token_count=17, reasoning="reasoning text", reasoning_content="reasoning sidecar",
        reasoning_details=[{"text": "detail"}], codex_reasoning_items=[{"id": "reason-1"}],
        codex_message_items=[{"id": "message-1"}], platform_message_id="platform-1",
        api_content="exact API content", display_metadata={"source": "test"},
    )
    result = db.append_message(
        "parent", role="tool", content="tool result", timestamp=21,
        tool_call_id="call-1", tool_name="terminal", effect_disposition="applied",
        token_count=9, observed=True,
    )
    raw.db.executemany(
        "UPDATE messages SET clone_extension = ? WHERE id = ?",
        [("caller extension", caller), ("result extension", result)],
    )
    return [caller, result]


def _raw_rows(raw, session_id, *, active_only=False):
    cursor = raw.db.execute(
        "SELECT * FROM messages WHERE session_id = ?"
        + (" AND active = 1" if active_only else "") + " ORDER BY id", (session_id,),
    )
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _payload(row):
    return {key: value for key, value in row.items()
            if key not in {"id", "session_id", "active", "compacted", "display_order"}}


def test_compaction_clones_all_tail_columns_and_rebuilds_display_order(message_store):
    db, raw = message_store
    db.append_message("parent", role="user", content="before summary", timestamp=10)
    watermark = db.get_active_message_watermark("parent")
    tail_ids = _rich_tail(db, raw)
    originals = [row for row in _raw_rows(raw, "parent") if row["id"] in tail_ids]
    summary = [{"role": "assistant", "content": "summary", "timestamp": 30}]

    assert db.archive_and_compact("parent", summary, watermark=watermark) == len(summary) + len(tail_ids)
    live = _raw_rows(raw, "parent", active_only=True)
    clones = live[len(summary):]
    assert [_payload(row) for row in clones] == [_payload(row) for row in originals]
    assert all(row["id"] > max(tail_ids) and row["compacted"] == 0 for row in clones)
    archived = [row for row in _raw_rows(raw, "parent") if row["id"] in tail_ids]
    assert all(row["active"] == row["compacted"] == 0 for row in archived)
    session = db.get_session("parent")
    assert session["message_count"] == len(live)
    assert session["tool_call_count"] == 1

    displayed = db.get_messages("parent", include_compacted=True, limit=50)
    assert [row["content"] for row in displayed] == ["before summary", "summary", "tool caller", "tool result"]
    assert all(row["display_order"] is not None for row in _raw_rows(raw, "parent", active_only=True))


def test_rotation_clone_retargets_every_row_without_mutating_parent(message_store):
    db, raw = message_store
    tail_ids = _rich_tail(db, raw)
    db.get_messages("parent", include_compacted=True, limit=50)
    before = _raw_rows(raw, "parent")

    db._execute_write(lambda conn: db._clone_message_rows(conn, tail_ids, session_id="child"))
    cloned = _raw_rows(raw, "child")
    assert _raw_rows(raw, "parent") == before
    assert [_payload(row) for row in cloned] == [_payload(row) for row in before]
    assert all(row["id"] > max(tail_ids) and row["active"] == 1 and row["compacted"] == 0 for row in cloned)
    appended = db.append_message("child", role="assistant", content="continued", timestamp=22)
    displayed = db.get_messages("child", include_compacted=True, limit=50)
    assert [row["id"] for row in displayed] == [*(row["id"] for row in cloned), appended]
    assert [row["display_order"] for row in _raw_rows(raw, "child")] == [row["id"] for row in displayed]
