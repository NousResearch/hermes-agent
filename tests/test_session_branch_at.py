import json

import pytest

from hermes_state import SessionDB


def _messages(db: SessionDB, session_id: str):
    return db.get_messages_as_conversation(session_id)


def test_get_messages_as_conversation_omits_ids_by_default_for_replay(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("source", source="cli")
    message_id = db.append_message("source", role="user", content="hello", timestamp=10)

    replay = db.get_messages_as_conversation("source")
    display = db.get_messages_as_conversation("source", include_ids=True)

    assert replay == [{"role": "user", "content": "hello", "timestamp": 10}]
    assert display == [{"id": message_id, "session_id": "source", "role": "user", "content": "hello", "timestamp": 10}]


def test_branch_at_user_message_copies_prefix_before_user_and_returns_prefill(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("source", source="cli", model="model-a", model_config={"provider": "test"})
    first_id = db.append_message("source", role="user", content="first question", timestamp=10)
    db.append_message("source", role="assistant", content="first answer", timestamp=11)
    target_id = db.append_message("source", role="user", content="edit me", timestamp=12)
    db.append_message("source", role="assistant", content="tail that must not copy", timestamp=13)

    result = db.branch_at_message("source", target_id, new_session_id="branch")

    assert result == {
        "session_id": "branch",
        "parent_session_id": "source",
        "source_session_id": "source",
        "source_message_id": target_id,
        "cut_mode": "user_prefill_before",
        "copied_message_count": 2,
        "prefill": "edit me",
    }
    assert [m["content"] for m in _messages(db, "source")] == [
        "first question",
        "first answer",
        "edit me",
        "tail that must not copy",
    ]
    assert [m["content"] for m in _messages(db, "branch")] == [
        "first question",
        "first answer",
    ]
    branch_row = db.get_session("branch")
    assert branch_row["parent_session_id"] == "source"
    meta = json.loads(branch_row["model_config"])
    assert meta["provider"] == "test"
    assert meta["_branched_from"] == "source"
    assert meta["_branch_from_message_id"] == target_id
    assert meta["_branch_cut_mode"] == "user_prefill_before"
    listed_ids = {session["id"] for session in db.list_sessions_rich(limit=10)}
    assert "source" in listed_ids
    assert "branch" in listed_ids


def test_branch_at_assistant_message_copies_through_answer_and_preserves_replay_fields(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("source", source="cli")
    db.append_message("source", role="user", content="question", timestamp=20)
    db.append_message(
        "source",
        role="assistant",
        content=None,  # type: ignore[arg-type]
        tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        finish_reason="tool_calls",
        timestamp=21,
    )
    db.append_message("source", role="tool", content="search result", tool_call_id="call_1", timestamp=22)
    target_id = db.append_message(
        "source",
        role="assistant",
        content="answer",
        finish_reason="stop",
        reasoning="private reasoning",
        reasoning_content="provider reasoning",
        reasoning_details=[{"type": "summary", "text": "reasoned"}],
        codex_reasoning_items=[{"id": "r1", "type": "reasoning"}],
        codex_message_items=[{"id": "m1", "type": "message"}],
        timestamp=23,
    )
    db.append_message("source", role="user", content="tail", timestamp=24)

    result = db.branch_at_message("source", target_id, new_session_id="branch")

    assert result["cut_mode"] == "assistant_after"
    assert result["prefill"] is None
    copied = _messages(db, "branch")
    assert [m["role"] for m in copied] == ["user", "assistant", "tool", "assistant"]
    tool_call_assistant = copied[1]
    assert tool_call_assistant["tool_calls"] == [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
    assert tool_call_assistant["finish_reason"] == "tool_calls"
    assert copied[2]["tool_call_id"] == "call_1"
    assistant = copied[3]
    assert assistant["content"] == "answer"
    assert assistant["finish_reason"] == "stop"
    assert assistant["reasoning"] == "private reasoning"
    assert assistant["reasoning_content"] == "provider reasoning"
    assert assistant["reasoning_details"] == [{"type": "summary", "text": "reasoned"}]
    assert assistant["codex_reasoning_items"] == [{"id": "r1", "type": "reasoning"}]
    assert assistant["codex_message_items"] == [{"id": "m1", "type": "message"}]
    assert assistant["timestamp"] == 23


def test_branch_at_rejects_tool_message_cut_points(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("source", source="cli")
    db.append_message("source", role="assistant", content=None, tool_calls=[{"id": "call_1"}])  # type: ignore[arg-type]
    tool_id = db.append_message("source", role="tool", content="result", tool_call_id="call_1")

    with pytest.raises(ValueError, match="cannot branch at tool message"):
        db.branch_at_message("source", tool_id, new_session_id="branch")

    assert db.get_session("branch") is None


def test_branch_at_rejects_tool_call_only_assistant_cut_points(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("source", source="cli")
    target_id = db.append_message("source", role="assistant", content=None, tool_calls=[{"id": "call_1"}])

    with pytest.raises(ValueError, match="tool-call-only assistant"):
        db.branch_at_message("source", target_id, new_session_id="branch")

    assert db.get_session("branch") is None


def test_branch_at_user_retry_include_copies_prefix_and_returns_retry_prefill(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("source", source="cli")
    db.append_message("source", role="user", content="first", timestamp=30)
    db.append_message("source", role="assistant", content="answer", timestamp=31)
    target_id = db.append_message("source", role="user", content="retry this", timestamp=32)
    db.append_message("source", role="assistant", content="stale answer", timestamp=33)

    result = db.branch_at_message(
        "source", target_id, new_session_id="branch", cut_mode="user_retry_include"
    )

    assert result["cut_mode"] == "user_retry_include"
    assert result["copied_message_count"] == 2
    assert result["prefill"] == "retry this"
    assert [m["content"] for m in _messages(db, "branch")] == ["first", "answer"]


def test_branch_at_refuses_to_overwrite_existing_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("source", source="cli")
    target_id = db.append_message("source", role="user", content="branch me")
    db.create_session("branch", source="cli")
    db.append_message("branch", role="user", content="keep me")

    with pytest.raises(ValueError, match="already exists"):
        db.branch_at_message("source", target_id, new_session_id="branch")

    assert [m["content"] for m in _messages(db, "branch")] == ["keep me"]


def test_get_messages_with_ancestors_includes_owner_session_ids(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="cli")
    parent_id = db.append_message("parent", role="user", content="from parent", timestamp=1)
    db.create_session("child", source="cli", parent_session_id="parent")
    child_id = db.append_message("child", role="assistant", content="from child", timestamp=2)

    display = db.get_messages_as_conversation("child", include_ancestors=True, include_ids=True)

    assert [(m["id"], m["session_id"], m["content"]) for m in display] == [
        (parent_id, "parent", "from parent"),
        (child_id, "child", "from child"),
    ]