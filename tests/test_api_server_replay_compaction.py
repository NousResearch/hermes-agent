"""Read-path counterpart to api_server write compaction.

Sessions persisted before the write-path filter existed still hold raw tool
rows in state.db. api_server replay must skip them on load (and strip
dangling tool_calls from kept assistant rows, which providers reject),
while the default load keeps full fidelity for CLI/audit flows.
"""

from hermes_state import SessionDB


def _seed_legacy_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s1", source="api_server")
    db.append_message("s1", role="user", content="run the build")
    # Legacy empty assistant tool-call stub.
    db.append_message(
        "s1",
        role="assistant",
        content="",
        tool_calls=[{"name": "terminal", "arguments": "{}"}],
    )
    # Legacy narrated assistant row that still carries tool plumbing.
    db.append_message(
        "s1",
        role="assistant",
        content="Checking the build...",
        tool_calls=[{"name": "terminal", "arguments": "{}"}],
    )
    db.append_message(
        "s1",
        role="tool",
        content="large build log",
        tool_name="terminal",
        tool_call_id="call_1",
    )
    db.append_message("s1", role="assistant", content="Build passed.")
    return db


def test_chat_text_only_replay_filters_legacy_tool_rows(tmp_path):
    db = _seed_legacy_session(tmp_path)

    compact = db.get_messages_as_conversation("s1", chat_text_only=True)

    assert [m["role"] for m in compact] == ["user", "assistant", "assistant"]
    assert [m["content"] for m in compact] == [
        "run the build",
        "Checking the build...",
        "Build passed.",
    ]
    for m in compact:
        assert "tool_calls" not in m
        assert "tool_call_id" not in m
        assert "tool_name" not in m


def test_default_replay_keeps_full_fidelity(tmp_path):
    db = _seed_legacy_session(tmp_path)

    full = db.get_messages_as_conversation("s1")

    assert [m["role"] for m in full] == [
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
    ]
    assert full[1]["tool_calls"] == [{"name": "terminal", "arguments": "{}"}]
    assert full[3]["tool_name"] == "terminal"
