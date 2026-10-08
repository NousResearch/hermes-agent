"""Real tool executor -> canonical SQLite -> authenticated route and history."""
import json
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from tests.agent.test_tool_call_incremental_persistence import _attach_real_session_db, _make_agent, _mock_tool_call


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_inline_artifact_survives_cold_retrieval_at_exact_tool_position(tmp_path, monkeypatch, mode):
    from hermes_cli import web_server
    from hermes_state import SessionDB
    from tools.inline_artifact_tool import publish_html
    import model_tools
    import tui_gateway.server as gateway

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", None, raising=False)
    agent = _make_agent()
    agent.valid_tool_names = {"publish_html"}
    sid, call_id = "artifact-chat", "artifact-call"
    db = _attach_real_session_db(agent, home / "state.db", sid)
    args = {"title": "Report", "html": "<h1>Original</h1>", "fallback": "Original report"}
    messages = [{"role": "user", "content": "Publish"}, {"role": "assistant", "content": "Before",
                 "tool_calls": [{"id": call_id, "type": "function", "function": {
                     "name": "publish_html", "arguments": json.dumps(args)}}]}]
    agent._flush_messages_to_session_db(messages)
    events = []
    import threading
    persisted_at_emission = []
    monkeypatch.setitem(gateway._sessions, sid, {"agent": agent, "session_key": sid,
        "profile_home": str(home), "history": [], "history_lock": threading.Lock(),
        "running": False, "created_at": 1.0, "last_active": 1.0})
    def record_event(kind, event_sid, name, event_args, payload):
        events.append((kind, payload))
        if kind == "tool.complete":
            with SessionDB(db_path=home / "state.db") as cold:
                persisted_at_emission.extend(cold.get_messages(sid))
    monkeypatch.setattr(gateway, "_emit_tool_lifecycle", record_event)
    monkeypatch.setattr(gateway, "_tool_progress_enabled", lambda _sid: False)
    for key, callback in gateway._agent_cbs(sid).items():
        setattr(agent, key, callback)
    try:
        getattr(agent, f"_execute_tool_calls_{mode}")(
            SimpleNamespace(tool_calls=[_mock_tool_call("publish_html", json.dumps(args), call_id)]), messages, sid)
        row = next(m for m in db.get_messages(sid) if m["role"] == "tool")
        assert "artifact" in json.loads(row["content"]), row["content"]
        artifact = row["display_metadata"]["inline_artifact"]
        assert artifact["html"] == args["html"]
        event = next(p for kind, p in events if kind == "tool.complete")
        assert event["inline_artifact"]["id"] == artifact["id"]
        assert "html" not in event["inline_artifact"]
        assert event["tool_id"] == call_id
        assert next(m for m in persisted_at_emission if m["role"] == "tool")["display_metadata"]["inline_artifact"] == artifact
        # Re-flushing the same occurrence is idempotent under canonical message identity.
        agent._flush_messages_to_session_db(messages)
        assert len([m for m in db.get_messages(sid) if m["role"] == "tool"]) == 1
        db.create_session("unrelated", source="gui")
        db.close()
        with SessionDB(db_path=home / "state.db") as cold:
            history = cold.get_messages_as_conversation(sid, include_row_ids=True)
        projected = gateway._history_to_messages(history)
        assert [m["role"] for m in projected] == ["user", "assistant", "tool"]
        tool = projected[-1]
        assert tool["row_id"] == row["id"]
        assert tool["tool_call_id"] == call_id
        assert tool["display_metadata"]["inline_artifact"] == event["inline_artifact"]
        assert "html" not in tool.get("args", {})
        replay = gateway.handle_request({"id": "replay", "method": "session.history", "params": {"session_id": sid}})
        assert "result" in replay, replay
        assert replay["result"]["messages"][-1]["display_metadata"]["inline_artifact"] == event["inline_artifact"]
        assert replay["result"]["messages"][-1]["row_id"] == row["id"]
        with TestClient(web_server.app) as client:
            path = f"/api/sessions/{sid}/artifacts/{artifact['id']}"
            assert client.get(path).status_code == 401
            assert client.get(path, params={"token": web_server._SESSION_TOKEN}).status_code == 401
            client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
            response = client.get(path)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["html"] == args["html"]
            assert result["association"] == {"session_id": sid, "row_id": row["id"], "tool_call_id": call_id}
            assert result["render_policy"]["sandbox"] == ""
            listing = client.get(f"/api/sessions/{sid}/artifacts")
            assert listing.status_code == 200, listing.text
            assert listing.json()["artifacts"] == [{"artifact": event["inline_artifact"], "association": result["association"]}]
            assert listing.json()["count"] == 1
            assert client.get(path, params={"row_id": row["id"] + 1000}).status_code == 404
            assert "script-src 'none'" in result["render_html"]
            assert response.headers["content-type"].startswith("application/json")
            assert response.headers["cache-control"] == "no-store"
            assert client.get(path.replace(sid, "unrelated")).status_code == 404
            assert client.get(path, params={"profile": "missing"}).status_code == 404
            assert client.get(path.replace(artifact["id"], "html_" + "0" * 64)).status_code == 404
    finally:
        db.close()


@pytest.mark.parametrize("fail_flush", [False, True])
def test_publication_uses_runtime_fence_without_gui_metadata_callback(tmp_path, monkeypatch, fail_flush):
    agent = _make_agent()
    agent.valid_tool_names = {"publish_html"}
    sid = "headless"
    db = _attach_real_session_db(agent, tmp_path / "state.db", sid)
    args = {"title": "Report", "html": "<h1>Headless</h1>", "fallback": "Headless report"}
    messages = [{"role": "user", "content": "Publish"}, {"role": "assistant", "content": "",
        "tool_calls": [{"id": "headless-call", "type": "function", "function": {
            "name": "publish_html", "arguments": json.dumps(args)}}]}]
    agent._flush_messages_to_session_db(messages)
    completions = []
    agent.tool_complete_callback = lambda *a: completions.append(a)
    agent.tool_result_metadata_callback = None
    if fail_flush:
        monkeypatch.setattr(agent, "_flush_messages_to_session_db", lambda _messages: False)
    try:
        agent._execute_tool_calls_sequential(SimpleNamespace(tool_calls=[
            _mock_tool_call("publish_html", json.dumps(args), "headless-call")]), messages, sid)
        rows = [m for m in db.get_messages(sid) if m["role"] == "tool"]
        if fail_flush:
            assert rows == []
            assert completions == []
        else:
            assert len(rows) == 1
            assert rows[0]["display_metadata"]["inline_artifact"]["html"] == args["html"]
            assert json.loads(rows[0]["content"])["text"] == args["fallback"]
            assert len(completions) == 1
    finally:
        db.close()
