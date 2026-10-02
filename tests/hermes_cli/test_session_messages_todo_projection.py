"""Desktop's real FastAPI session reader exposes bounded durable Todo authority."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_state import SessionDB

TODOS = [{"id": "next", "content": "Next task", "status": "pending"}]
DATA_URI = "data:image/png;base64," + "a" * 128


@pytest.fixture
def router_client(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    db = SessionDB(home / "state.db")
    db.create_session("todo-chat", "desktop")
    db.close()

    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        yield client, home


def _pair(db, todos, call_id="todo-call"):
    call = {"id": call_id, "type": "function", "function": {"name": "todo", "arguments": "{}"}}
    db.append_message("todo-chat", "assistant", "", tool_calls=[call])
    content = json.dumps({"todos": todos, "revision": 3})
    db.append_message("todo-chat", "tool", content, tool_call_id=call_id)
    return call, content


def _forbid_history(*args, **kwargs):
    raise AssertionError("Todo projection must not load ordinary transcript history")


@pytest.mark.parametrize("scenario", ["empty", "tool", "clear", "carrier", "lineage", "profiles", "ordinary"])
def test_real_router_todo_authority(router_client, monkeypatch, scenario):
    client, home = router_client
    db = SessionDB(home / "state.db")
    try:
        if scenario in {"tool", "clear"}:
            call, result = _pair(db, TODOS)
            db.append_messages_batch("todo-chat", [
                {"role": "assistant", "content": f"ordinary row {index}"} for index in range(600)
            ])
            if scenario == "clear":
                call, result = _pair(db, [], "clear-call")
        elif scenario in {"carrier", "lineage"}:
            db.append_message("todo-chat", "user", "old prefix")
            db.archive_and_compact("todo-chat", [{
                "role": "user", "content": "opaque compaction carrier", "display_kind": "hidden",
                "display_metadata": {"todo_snapshot": {"todos": TODOS}},
            }])
            if scenario == "lineage":
                db.end_session("todo-chat", "compression")
                db.create_session("todo-child", "desktop", parent_session_id="todo-chat")
                db.append_message("todo-child", "user", "child carrier", display_kind="hidden",
                                  display_metadata={"todo_snapshot": {"todos": TODOS}})
        elif scenario == "profiles":
            _pair(db, TODOS)
            other_home = home / "profiles" / "other"
            other_home.mkdir(parents=True)
            other = SessionDB(other_home / "state.db")
            try:
                other.create_session("todo-chat", "desktop")
                _pair(other, [{"id": "other", "content": "Other profile plan", "status": "completed"}])
            finally:
                other.close()
        elif scenario == "ordinary":
            db.append_messages_batch("todo-chat", [
                {"role": "user", "content": [
                    {"type": "text", "text": "image question"},
                    {"type": "image_url", "image_url": {"url": DATA_URI}},
                ]},
                {"role": "user", "content": "[System: recovery scaffold]"},
                {"role": "assistant", "content": "answer"},
            ])
    finally:
        db.close()

    if scenario == "ordinary":
        response = client.get("/api/sessions/todo-chat/messages?limit=2&order=oldest&inline_images=false")
        assert response.status_code == 200
        page = response.json()
        assert page["pagination"] == {"limit": 2, "offset": 0, "order": "oldest", "returned": 2}
        assert "[image]" in page["messages"][0]["content"]
        assert DATA_URI not in str(page["messages"])
        assert page["messages"][1]["display_kind"] == "hidden"
        tail = client.get("/api/sessions/todo-chat/messages?limit=1&offset=2&order=oldest").json()
        assert [row["content"] for row in tail["messages"]] == ["answer"]
        return

    monkeypatch.setattr(SessionDB, "get_messages", _forbid_history)
    for projection in ("todo-state", "todo-state-candidates"):
        params = {"projection": projection}
        if projection == "todo-state-candidates":
            params["limit"] = 0
        response = client.get("/api/sessions/todo-chat/messages", params=params)
        assert response.status_code == 200
        page = response.json()
        expected_count = 0 if scenario == "empty" else 1 if scenario in {"carrier", "lineage"} else 2
        assert page["pagination"] == {
            "limit": 2, "returned": expected_count, "has_more": False,
            "exhausted": True, "next_before_id": None,
        }
        assert page["session_id"] == ("todo-child" if scenario == "lineage" else "todo-chat")
        assert page["profile"] == "default"
        if scenario in {"tool", "clear"}:
            assert page["messages"][0]["tool_calls"] == [call]
            assert page["messages"][1]["content"] == result
            assert json.loads(page["messages"][1]["content"])["todos"] == ([] if scenario == "clear" else TODOS)
        elif scenario in {"carrier", "lineage"}:
            assert page["messages"][0]["display_metadata"] == {"todo_snapshot": {"todos": TODOS}}
            assert page["messages"][0]["display_kind"] == "hidden"
            # Structured authority stores metadata and an empty content field.
            assert page["messages"][0]["content"] == ""
        elif scenario == "profiles":
            other_page = client.get("/api/sessions/todo-chat/messages", params={**params, "profile": "other"}).json()
            assert other_page["profile"] == "other"
            assert json.loads(other_page["messages"][1]["content"])["todos"][0]["id"] == "other"
            again = client.get("/api/sessions/todo-chat/messages", params={**params, "profile": "default"}).json()
            assert json.loads(again["messages"][1]["content"])["todos"] == TODOS


@pytest.mark.parametrize("scenario,query,status,code", [
    ("invalid", "projection=unknown", 400, None),
    ("invalid", "projection=", 400, None),
    ("invalid", "projection=todo-state&limit=0", 400, None),
    ("invalid", "projection=todo-state&limit=2", 400, None),
    ("invalid", "projection=todo-state&offset=1", 400, None),
    ("invalid", "projection=todo-state&order=latest", 400, None),
    ("invalid", "projection=todo-state&before_id=0", 400, None),
    ("invalid", "projection=todo-state-candidates&offset=1", 400, None),
    ("invalid", "projection=todo-state-candidates&order=oldest", 400, None),
    ("invalid", "projection=todo-state-candidates&before_id=1", 400, None),
    ("missing", "projection=todo-state", 404, None),
    ("pending", "projection=todo-state", 503, "todo_state_migration_pending"),
    ("oversized", "projection=todo-state", 413, "todo_state_response_too_large"),
    ("identity", "projection=todo-state", 200, None),
    ("oversized-envelope", "projection=todo-state", 413, "todo_state_response_too_large"),
    ("near-limit", "projection=todo-state", 200, None),
])
def test_real_router_projection_boundaries(router_client, monkeypatch, scenario, query, status, code):
    client, home = router_client
    original = None
    if scenario in {"pending", "oversized", "oversized-envelope", "near-limit"}:
        original = None if scenario == "pending" else [{"role": "user", "content": "界" * 400_000}]
        if scenario == "oversized-envelope":
            # Content is below the cap; response metadata pushes the whole envelope over it.
            original = [{"role": "user", "content": "x" * 1_099_950}]
        elif scenario == "near-limit":
            from agent.message_metadata import stamp_persisted_todo_snapshot
            original = [stamp_persisted_todo_snapshot({
                "role": "user", "content": "x" * 1_099_500, "message_uid": "near-limit-uid"})]
        monkeypatch.setattr(SessionDB, "get_todo_state_messages", lambda self, sid: original)
    elif scenario == "identity":
        db = SessionDB(home / "state.db")
        try:
            _pair(db, TODOS)
            # Durable authority fixtures may already carry canonical identity fields.
            # The router must preserve every field returned by the real validating lookup.
            row = db._conn.execute("SELECT message_id, authority_json FROM todo_authorities WHERE session_id = ?",
                                   ("todo-chat",)).fetchone()
            evidence = json.loads(row["authority_json"])
            evidence[0].update({"message_uid": "assistant-uid", "absorbed_message_uids": ["absorbed-uid"],
                                "tool_call_uids": {"todo-call": "occurrence-uid"}, "_todo_snapshot_provenance": "local-only"})
            evidence[1].update({"message_uid": "result-uid", "tool_call_uid": "occurrence-uid"})
            db._conn.execute("UPDATE todo_authorities SET authority_json = ? WHERE message_id = ?",
                             (json.dumps(evidence), row["message_id"]))
            db._conn.commit()
        finally:
            db.close()

    monkeypatch.setattr(SessionDB, "get_messages", _forbid_history)
    sid = "missing-chat" if scenario == "missing" else "todo-chat"
    response = client.get(f"/api/sessions/{sid}/messages?{query}")
    assert response.status_code == status
    if code:
        assert response.json()["detail"]["code"] == code
        assert len(response.content) < 1_000
    if scenario == "identity":
        rows = response.json()["messages"]
        assert rows[0]["message_uid"] == "assistant-uid"
        assert rows[0]["absorbed_message_uids"] == ["absorbed-uid"]
        assert rows[0]["tool_call_uids"] == {"todo-call": "occurrence-uid"}
        assert rows[1]["message_uid"] == "result-uid"
        assert rows[1]["tool_call_uid"] == "occurrence-uid"
        assert "_todo_snapshot_provenance" not in rows[0]
        db = SessionDB(home / "state.db")
        try:
            assert db.get_todo_state_messages("todo-chat")[0]["_todo_snapshot_provenance"] == "local-only"
        finally:
            db.close()
    if scenario == "oversized":
        assert original == [{"role": "user", "content": "界" * 400_000}]
    if scenario == "near-limit":
        assert len(response.content) <= 1_100_000
        assert response.json()["messages"] == [{
            "role": "user", "content": "x" * 1_099_500, "message_uid": "near-limit-uid"}]
        from agent.message_metadata import has_persisted_todo_snapshot_provenance
        assert has_persisted_todo_snapshot_provenance(original[0])