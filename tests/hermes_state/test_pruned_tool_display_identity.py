"""Pruned tool copies project once while the complete archived output stays exportable (#132939)."""

import json

import pytest

from hermes_state import SessionDB
from hermes_state_timeline import get_session_messages_around


@pytest.fixture
def compacted_db(tmp_path):
    """Persist an original tool output and its carried-forward pruned copy through real compaction."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("chat", source="test")
    db.append_messages_batch("chat", [
        {"role": "user", "content": "start", "timestamp": 100.0},
        {"role": "assistant", "content": "working", "timestamp": 101.0,
         "tool_calls": [{"id": "call", "type": "function", "function": {
             "name": "demo", "arguments": json.dumps({"value": "L" * 4000})}}]},
        {"role": "tool", "content": "R" * 5000, "tool_call_id": "call",
         "tool_name": "demo", "timestamp": 102.0},
        {"role": "assistant", "content": "answer", "timestamp": 103.0},
    ])
    history = db.get_messages_as_conversation("chat", include_row_ids=True)
    history[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"value": "short"})
    history[2]["content"] = "pruned result"
    db.archive_and_compact("chat", history)
    yield db
    db.close()


@pytest.mark.parametrize("view", ["indexed", "old-index", "read-only", "read-only-old-index"])
def test_pruned_tool_occurrence_projects_once_without_losing_full_result(compacted_db, view):
    """Every display surface keeps the original output, while the model keeps its pruned copy."""
    db = compacted_db
    if view in ("old-index", "read-only-old-index"):
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET display_identity = CAST('old-' || id AS BLOB), display_order = id "
            "WHERE session_id = 'chat' AND role = 'tool'"))
        db._execute_write(lambda conn: conn.execute(
            "DELETE FROM state_meta WHERE key = 'tool_display_identity_version'"))
    elif view == "read-only":
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET display_identity = NULL, display_order = NULL WHERE session_id = 'chat'"))
    if view != "indexed":
        db.close()
        db = SessionDB(db.db_path, read_only=view.startswith("read-only"))
    audit_before = [(m["id"], m["content"], m["active"], m["compacted"])
                    for m in db.get_messages("chat", include_inactive=True)]
    try:
        prompt = next(m for m in db.get_messages("chat", include_inactive=True)
                      if m["role"] == "user" and m["active"])
        jumped = get_session_messages_around(db, "chat", prompt["id"])
        model, resumed = db.get_resume_conversations("chat")
        visible = db.get_messages("chat", include_compacted=True)
        assert [m["content"] for m in model if m["role"] == "tool"] == ["pruned result"]
        for messages in (resumed, visible, db.get_messages_as_conversation("chat", include_compacted=True)):
            assert [m["content"] for m in messages if m["role"] == "tool"] == ["R" * 5000]
        assert db.get_messages("chat", include_compacted=True, limit=1, offset=2)[0]["content"] == "R" * 5000
        exported = db.export_session("chat", include_compacted=True)["messages"]
        assert [m["content"] for m in exported if m["role"] == "tool"] == ["R" * 5000]
        if not view.startswith("read-only"):
            assert db.display_message_count("chat") == len(visible)
        assert [m["content"] for m in jumped["messages"] if m["role"] == "tool"] == ["R" * 5000]
        assert [(m["role"], m["content"]) for m in jumped["messages"]] == [
            (m["role"], m["content"]) for m in visible]
        assert [(m["id"], m["content"], m["active"], m["compacted"])
                for m in db.get_messages("chat", include_inactive=True)] == audit_before
    finally:
        if view != "indexed":
            db.close()


@pytest.mark.parametrize("identity", ["distinct-uids", "missing-uids", "merged-index"])
def test_reused_provider_call_id_keeps_distinct_tool_occurrences(tmp_path, identity):
    """Reused provider call IDs cannot merge distinct occurrences, with or without durable uids."""
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("chat", source="test")
        db.append_message("chat", "user", "start", timestamp=100.0)
        results = ("first result", "second result") if identity == "missing-uids" else ("same result", "same result")
        for result in results:
            db.append_messages_batch("chat", [
                {"role": "assistant", "content": "working", "timestamp": 101.0,
                 "tool_calls": [{"id": "call", "type": "function", "function": {
                     "name": "demo", "arguments": "{}"}}]},
                {"role": "tool", "content": result, "timestamp": 102.0,
                 "tool_call_id": "call", "tool_name": "demo"},
            ])
        if identity == "missing-uids":
            db._execute_write(lambda conn: conn.execute(
                "UPDATE messages SET message_uid = NULL, display_identity = NULL, display_order = NULL "
                "WHERE session_id = 'chat'"))
        elif identity == "merged-index":
            db._execute_write(lambda conn: conn.execute(
                "UPDATE messages SET display_order = (SELECT MIN(id) FROM messages WHERE role = 'tool') "
                "WHERE session_id = 'chat' AND role = 'tool'"))
            db._execute_write(lambda conn: conn.execute(
                "DELETE FROM state_meta WHERE key = 'tool_display_identity_version'"))
            db.close()
            db = SessionDB(db.db_path)
        prompt = db.get_messages("chat", include_inactive=True)[0]
        jumped = get_session_messages_around(db, "chat", prompt["id"])["messages"]
        visible = db.get_messages("chat", include_compacted=True)
        for messages in (visible, db.get_resume_conversations("chat")[1], jumped):
            assert [m["content"] for m in messages if m["role"] == "tool"] == list(results)
    finally:
        db.close()
