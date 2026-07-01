from hermes_state import SessionDB
from tools.todo_tool import TodoStore, is_todo_injection_content


def test_todo_injection_is_internal_state_not_user_turn(tmp_path):
    store = TodoStore()
    store.write([
        {"id": "work", "content": "continue implementation", "status": "in_progress"},
    ])
    snapshot = store.format_for_injection()
    assert snapshot
    assert is_todo_injection_content(snapshot)

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s1", source="test")
        # Legacy databases may already contain these snapshots as user rows.
        db.append_message("s1", "user", snapshot)
        db.append_message("s1", "user", "actual user message")

        replay = db.get_messages_as_conversation("s1")

        assert replay[0]["role"] == "system"
        assert replay[0]["content"] == snapshot
        assert replay[1] == {"role": "user", "content": "actual user message", "timestamp": replay[1]["timestamp"]}
    finally:
        db.close()
