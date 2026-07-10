import json

from hermes_state import SessionDB


def test_update_assistant_tool_calls_replaces_active_preexecution_row(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="s1", source="cli", model="coding")
        original = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"content": "x" * 50_000}),
                },
            }
        ]
        db.append_message("s1", "assistant", tool_calls=original)
        compacted = [
            {
                **original[0],
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"content": "[compacted receipt]"}),
                },
            }
        ]

        assert db.update_assistant_tool_calls("s1", "call-1", compacted) is True
        stored = db._conn.execute(
            "SELECT tool_calls FROM messages WHERE session_id = 's1' AND active = 1"
        ).fetchone()[0]
        assert json.loads(stored) == compacted
        session = db.get_session("s1")
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 1
    finally:
        db.close()
