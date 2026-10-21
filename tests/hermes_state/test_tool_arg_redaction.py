import json
import sqlite3

from hermes_state import SessionDB


SECRET = "sk_LegacySecret_1234567890"


def _assert_absent_from_fts(conn):
    assert conn.execute(
        "SELECT 1 FROM messages_fts WHERE messages_fts MATCH ?", (f'"{SECRET}"',)
    ).fetchone() is None


def test_fresh_write_keeps_tool_operands_literal_while_redacting_projections(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("fresh", source="cli")
    tool_calls = {
        "id": "call-1",
        "function": {
            "name": "file_write",
            "arguments": {"nested": [{"content": SECRET}]},
        },
    }
    projections = [{"nested": [{"secret": SECRET}]}]
    metadata = {"nested": {"secret": SECRET}}
    try:
        db.append_message(
            "fresh", "assistant", SECRET, tool_calls=tool_calls,
            reasoning_details=projections,
            codex_reasoning_items=projections,
            codex_message_items=projections,
            display_metadata=metadata,
        )
        row = db._conn.execute(
            "SELECT content, tool_calls, reasoning_details, codex_reasoning_items, "
            "codex_message_items, display_metadata FROM messages WHERE session_id = 'fresh'"
        ).fetchone()
        assert SECRET not in " ".join(str(value) for value in (row["content"], row["reasoning_details"], row["codex_reasoning_items"], row["codex_message_items"], row["display_metadata"]))
        stored_tool_calls = json.loads(row["tool_calls"])
        assert tool_calls["function"]["arguments"]["nested"][0]["content"] == SECRET
        assert isinstance(stored_tool_calls["function"]["arguments"], dict)
        assert stored_tool_calls["function"]["arguments"]["nested"][0]["content"] == SECRET
        resumed = db.get_messages("fresh")[0]["tool_calls"]
        assert resumed == tool_calls
        assert SECRET not in json.dumps(db.get_messages("fresh")[0]["display_metadata"])
        # The raw replay operand is also indexed as tool-call JSON; this is the
        # documented storage limitation rather than a display projection.
        assert db._conn.execute(
            "SELECT 1 FROM messages_fts WHERE messages_fts MATCH ?", (f'"{SECRET}"',)
        ).fetchone() is not None
    finally:
        db.close()


def test_fresh_write_fails_closed_for_unserializable_projection(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("malformed", source="cli")
    try:
        db.append_message(
            "malformed", "assistant", "ok",
            tool_calls={"function": {"arguments": {"secret": SECRET, "bad": {object()}}}},
            reasoning_details={"bad": object()},
            display_metadata={"bad": object()},
        )
        row = db._conn.execute(
            "SELECT tool_calls, reasoning_details, display_metadata FROM messages WHERE session_id = 'malformed'"
        ).fetchone()
        assert SECRET not in " ".join(str(value) for value in (row["reasoning_details"], row["display_metadata"]))
        assert SECRET in row["tool_calls"]
    finally:
        db.close()


def test_schema_migration_keeps_legacy_tool_operands_literal(tmp_path):
    db_path = tmp_path / "state.db"
    legacy_tool_calls = {
        "id": "legacy",
        "function": {"name": "file_write", "arguments": {"nested": [{"content": SECRET}]}},
    }
    projections = {"nested": [{"secret": SECRET}]}

    db = SessionDB(db_path)
    db.create_session("legacy", source="cli")
    db.append_message("legacy", role="assistant", content="")
    db.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE messages SET content = ?, tool_calls = ?, reasoning_details = ?, "
            "codex_reasoning_items = ?, codex_message_items = ?, display_metadata = ? WHERE session_id = ?",
            (SECRET, json.dumps(legacy_tool_calls), json.dumps(projections), json.dumps(projections),
             json.dumps(projections), json.dumps(projections), "legacy"),
        )
        conn.execute("UPDATE schema_version SET version = 30")
        assert conn.execute(
            "SELECT 1 FROM messages_fts WHERE messages_fts MATCH ?", (f'"{SECRET}"',)
        ).fetchone() is not None

    migrated = SessionDB(db_path)
    try:
        row = migrated._conn.execute(
            "SELECT content, tool_calls, reasoning_details, codex_reasoning_items, "
            "codex_message_items, display_metadata FROM messages WHERE session_id = 'legacy'"
        ).fetchone()
        assert SECRET not in " ".join(str(value) for value in (row["content"], row["reasoning_details"], row["codex_reasoning_items"], row["codex_message_items"], row["display_metadata"]))
        assert json.loads(row["tool_calls"])["function"]["arguments"]["nested"][0]["content"] == SECRET
    finally:
        migrated.close()


def test_schema_migration_preserves_malformed_tool_operands(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("malformed", source="cli")
    db.append_message("malformed", role="assistant", content="")
    db.close()

    malformed = f'{{"arguments": "{SECRET}"'
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE messages SET tool_calls = ?", (malformed,))
        conn.execute("UPDATE schema_version SET version = 30")

    migrated = SessionDB(db_path)
    try:
        stored = migrated._conn.execute("SELECT tool_calls FROM messages").fetchone()[0]
        assert stored == malformed
    finally:
        migrated.close()
