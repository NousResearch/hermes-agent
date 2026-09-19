"""Read-only replay preserves provenance without migrating legacy stores."""

from contextlib import closing
import sqlite3

import pytest


@pytest.mark.parametrize("stored_format", ["missing-column", None, "structured-v1"])
def test_readonly_conversations_preserve_provenance_without_writes(tmp_path, stored_format):
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    value = None if stored_format == "missing-column" else stored_format
    with SessionDB(path) as writer:
        writer.create_session("s", source="cli")
        writer.append_messages_batch("s", [
            {"role": "user", "content": "Read the log."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call", "type": "function", "function": {
                    "name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call", "tool_name": "terminal",
             "content": "literal [Command interrupted]", "tool_result_format": value},
            {"role": "assistant", "content": "Done."},
        ])
    if stored_format == "missing-column":
        # Reproduce the schema immediately before provenance was introduced.
        with closing(sqlite3.connect(path, isolation_level=None)) as conn:
            conn.execute("ALTER TABLE messages DROP COLUMN tool_result_format")
    before = path.read_bytes()
    try:
        with SessionDB(path, read_only=True) as reader:
            conversation = reader.get_messages_as_conversation("s")
            model, display = reader.get_resume_conversations("s")
            for messages in (conversation, model, display):
                assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
                assert messages[2]["content"] == "literal [Command interrupted]"
                assert messages[2].get("tool_result_format") == value
                if value is None:
                    assert "tool_result_format" not in messages[2]
            rows = reader._fetch_conversation_rows(["s"], "", with_session_id=False)
            assert rows[2]["tool_result_format"] == value
    finally:
        assert path.read_bytes() == before
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    assert ("tool_result_format" in columns) == (stored_format != "missing-column")
