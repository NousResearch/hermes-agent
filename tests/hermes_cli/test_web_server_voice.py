"""Tests for dashboard voice-call prototype endpoints."""

from types import SimpleNamespace

import pytest


@pytest.fixture()
def voice_client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_cli.web_server as web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client, web_server


def test_voice_session_requires_dashboard_token(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    client = TestClient(app)
    resp = client.post("/api/voice/session", json={})
    assert resp.status_code == 401


def test_voice_session_returns_ephemeral_session(voice_client, monkeypatch):
    client, web_server = voice_client

    monkeypatch.setattr(
        web_server,
        "_create_openai_realtime_session",
        lambda user, mode="solo": {
            "client_secret": "ek_test",
            "endpoint": "https://api.openai.com/v1/realtime",
            "model": "gpt-realtime",
            "voice": "alloy",
            "expires_at": 123,
        },
    )

    resp = client.post("/api/voice/session", json={})
    assert resp.status_code == 200
    assert resp.json()["client_secret"] == "ek_test"
    assert resp.json()["model"] == "gpt-realtime"


def test_voice_session_config_uses_phone_call_turn_detection(voice_client):
    _client, web_server = voice_client

    config = web_server._voice_session_config(user="deniz")
    assert "dashboard user: deniz" in config["instructions"]
    assert config["audio"]["output"]["voice"] == "cedar"
    tool_names = [tool["name"] for tool in config["tools"]]
    assert "context_lookup" in tool_names
    assert "memory_lookup" in tool_names
    assert "kanban_lookup" in tool_names
    assert "brain_lookup" in tool_names
    assert "session_lookup" in tool_names
    assert "rolly_background" in tool_names
    turn_detection = config["audio"]["input"]["turn_detection"]
    assert turn_detection == {
        "type": "semantic_vad",
        "create_response": True,
        "interrupt_response": True,
    }


def test_voice_session_config_meet_mode_waits_for_hey_rolly(voice_client):
    _client, web_server = voice_client

    config = web_server._voice_session_config(user="arman", mode="meet")

    assert "MEET MODE" in config["instructions"]
    assert "Hey Rolly" in config["instructions"]
    assert "dashboard user: arman" in config["instructions"]
    assert config["audio"]["input"]["turn_detection"]["create_response"] is False


def test_voice_tool_rejects_unknown_tool(voice_client):
    client, _web_server = voice_client

    resp = client.post("/api/voice/tool", json={"name": "shell", "arguments": {}})
    assert resp.status_code == 400


def test_voice_tool_runs_research_bridge(voice_client, monkeypatch):
    client, web_server = voice_client

    monkeypatch.setattr(
        web_server,
        "_run_voice_research",
        lambda question, user, **_kwargs: f"answered: {question}",
    )

    resp = client.post(
        "/api/voice/tool",
        json={"name": "research", "arguments": {"question": "what is Rolly Voice?"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["result"] == "answered: what is Rolly Voice?"
    assert body["error"] is None
    assert body["tool_name"] == "research"


def test_voice_tool_runs_rolly_bridge_with_request_arg(voice_client, monkeypatch):
    client, web_server = voice_client

    monkeypatch.setattr(
        web_server,
        "_run_voice_research",
        lambda question, user, **_kwargs: f"answered: {user}: {question}",
    )

    resp = client.post(
        "/api/voice/tool",
        json={"name": "rolly", "arguments": {"request": "what were we doing yesterday?"}},
        headers={"X-Rolly-User": "deniz"},
    )
    assert resp.status_code == 200
    assert resp.json()["result"] == "answered: deniz: what were we doing yesterday?"


def test_voice_context_endpoint_returns_fast_context(voice_client, monkeypatch):
    client, web_server = voice_client

    monkeypatch.setattr(web_server, "_voice_memory_snapshot", lambda: "memory facts")
    monkeypatch.setattr(web_server, "_voice_kanban_digest", lambda query=None: "kanban facts")
    monkeypatch.setattr(web_server, "_voice_brain_lookup_text", lambda query=None, limit=1800: "brain facts")
    monkeypatch.setattr(web_server, "_voice_recent_sessions", lambda query=None, limit=1600: "session facts")

    resp = client.get("/api/voice/context?debug=true", headers={"X-Rolly-User": "deniz"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["user"] == "deniz"
    assert "memory facts" in body["context"]
    assert "kanban facts" in body["context"]
    assert body["chars"] == len(body["context"])


def test_voice_tool_memory_lookup_does_not_spawn_cli(voice_client, monkeypatch):
    client, web_server = voice_client

    monkeypatch.setattr(web_server, "_voice_memory_snapshot", lambda: "remembered preference")
    monkeypatch.setattr(web_server.subprocess, "run", lambda *args, **kwargs: pytest.fail("slow CLI should not run"))

    resp = client.post("/api/voice/tool", json={"name": "memory_lookup", "arguments": {"query": "preference"}})

    assert resp.status_code == 200
    assert resp.json()["result"] == "remembered preference"
    assert resp.json()["tool_name"] == "memory_lookup"


def test_voice_lookup_tools_return_explicit_no_match_text(voice_client, monkeypatch):
    client, web_server = voice_client

    monkeypatch.setattr(web_server, "_voice_kanban_digest", lambda query=None: "")
    monkeypatch.setattr(web_server, "_voice_recent_sessions", lambda query=None: "")

    kanban = client.post("/api/voice/tool", json={"name": "kanban_lookup", "arguments": {"query": "blocked mix card"}})
    sessions = client.post("/api/voice/tool", json={"name": "session_lookup", "arguments": {"query": "recent mix message"}})

    assert kanban.status_code == 200
    assert kanban.json()["result"] == "Kanban: no matching results found for query: blocked mix card."
    assert sessions.status_code == 200
    assert sessions.json()["result"] == "Sessions: no matching results found for query: recent mix message."


def test_voice_tool_dedupes_same_realtime_call_id(voice_client, monkeypatch):
    client, web_server = voice_client
    calls = []

    def fake_run(question, user=None, **_kwargs):
        calls.append((question, user))
        return "answer once"

    monkeypatch.setattr(web_server, "_run_voice_research", fake_run)
    payload = {
        "name": "rolly",
        "arguments": {"request": "status"},
        "realtime_call_id": "call_same",
    }

    first = client.post("/api/voice/tool", json=payload, headers={"X-Rolly-User": "deniz"})
    second = client.post("/api/voice/tool", json=payload, headers={"X-Rolly-User": "deniz"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(calls) == 1
    assert second.json()["cached"] is True


def test_voice_transcript_persists_jsonl(voice_client):
    client, _web_server = voice_client

    resp = client.post(
        "/api/voice/transcript",
        json={
            "call_id": "call/1",
            "role": "user",
            "text": "hello",
            "user": "deniz",
            "sequence": 3,
            "elapsed_ms": 1200,
            "metadata": {"source": "test"},
        },
    )

    assert resp.status_code == 200
    path = resp.json()["path"]
    assert path.endswith("voice-transcripts/call_1.jsonl")
    with open(path, encoding="utf-8") as fh:
        body = fh.read()
    assert '"user": "deniz"' in body
    assert '"text": "hello"' in body
    assert '"sequence": 3' in body
    assert '"elapsed_ms": 1200' in body
    assert '"source": "test"' in body


def test_run_voice_research_uses_cli_bridge(monkeypatch, voice_client):
    _client, web_server = voice_client
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="session_id: abc\nspoken answer\n", stderr="")

    monkeypatch.setattr(web_server.subprocess, "run", fake_run)
    result = web_server._run_voice_research("who am I?", user="deniz")

    assert result == "spoken answer"
    assert calls[0][0][1:5] == ["-m", "hermes_cli.main", "chat", "-q"]
    assert "Dashboard voice user: deniz" in calls[0][0][5]
    assert calls[0][0][-3:] == ["--source", "dashboard-voice", "-Q"]
    assert calls[0][1]["timeout"] == 90


def test_run_voice_research_background_timeout_is_longer_and_concise(monkeypatch, voice_client):
    _client, web_server = voice_client

    def fake_run(_args, **_kwargs):
        raise web_server.subprocess.TimeoutExpired(cmd=["python", "-m", "hermes_cli.main", "chat", "huge prompt"], timeout=600)

    monkeypatch.setattr(web_server.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        web_server._run_voice_research("long task", user="deniz", source="dashboard-voice-background")

    message = str(exc_info.value)
    assert message == "Rolly CLI tool timed out after 600s"
    assert "huge prompt" not in message


def test_voice_transcript_creates_state_session(voice_client):
    client, _web_server = voice_client

    resp = client.post(
        "/api/voice/transcript",
        json={
            "call_id": "state-call",
            "role": "user",
            "text": "remember this voice line",
            "user": "deniz",
            "event_type": "transcript",
            "sequence": 1,
            "metadata": {"mode": "solo"},
        },
    )

    assert resp.status_code == 200
    from hermes_state import SessionDB

    db = SessionDB()
    session = db.get_session("dashboard_voice_state-call")
    assert session is not None
    assert session["source"] == "dashboard-voice"
    assert session["user_id"] == "deniz"
    messages = db.get_messages("dashboard_voice_state-call")
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "remember this voice line"


def test_voice_background_tool_starts_real_task_contract(voice_client, monkeypatch):
    client, web_server = voice_client
    started = []

    def fake_start(call_id, request_text, user):
        task = web_server.VoiceTask("vt_test", call_id, user, request_text, "voice_task_vt_test")
        started.append(task)
        return task

    monkeypatch.setattr(web_server, "_voice_start_background_task", fake_start)
    resp = client.post(
        "/api/voice/tool",
        json={"name": "rolly_background", "call_id": "voice-call", "arguments": {"request": "do the thing"}},
        headers={"X-Rolly-User": "deniz"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["tool_name"] == "rolly_background"
    assert body["data"]["task_id"] == "vt_test"
    assert body["data"]["status"] == "queued"
    assert started[0].call_id == "voice-call"
    assert started[0].request == "do the thing"


def test_voice_context_lookup_reports_live_voice_task_status(voice_client, monkeypatch):
    client, web_server = voice_client
    task = web_server.VoiceTask("vt_live", "call-live", "deniz", "do work", "voice_task_vt_live")
    task.mark("failed", "Rolly background task failed: timeout", error="timeout")
    with web_server._VOICE_TASKS_LOCK:
        web_server._VOICE_TASKS["vt_live"] = task

    monkeypatch.setattr(web_server, "_voice_memory_snapshot", lambda: "")
    monkeypatch.setattr(web_server, "_voice_kanban_digest", lambda query=None: "")
    monkeypatch.setattr(web_server, "_voice_brain_lookup_text", lambda query=None, limit=1800: "")
    monkeypatch.setattr(web_server, "_voice_recent_sessions", lambda query=None, limit=1600: "")

    resp = client.post(
        "/api/voice/tool",
        json={"name": "context_lookup", "arguments": {"query": "status for vt_live", "sources": ["sessions"]}},
        headers={"X-Rolly-User": "deniz"},
    )

    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "Voice tasks:" in result
    assert "vt_live: failed" in result
    assert "timeout" in result


def test_voice_task_status_lookup_reports_unavailable_for_missing_id(voice_client):
    _client, web_server = voice_client

    result = web_server._voice_task_status_lookup("check vt_missing")

    assert "vt_missing: status unavailable" in result


def test_voice_session_config_reports_speaking_rate_support(voice_client, monkeypatch):
    _client, web_server = voice_client
    monkeypatch.setenv("HERMES_VOICE_SPEAKING_RATE", "1.08")

    config = web_server._voice_session_config(user="deniz")

    assert "slightly faster" in config["instructions"]
    rate = config["metadata"]["speaking_rate"]
    assert rate["requested"] == 1.08
    assert rate["explicit_control_supported"] is False
    assert rate["provider"] == "openai-realtime-webrtc"


def test_voice_invite_requires_feature_flag(voice_client, monkeypatch):
    client, _web_server = voice_client
    monkeypatch.delenv("HERMES_VOICE_MEET_INVITES", raising=False)

    resp = client.post("/api/voice/meet/invite", json={"call_id": "voice-call", "user": "deniz"})

    assert resp.status_code == 404


def test_voice_invite_returns_share_url_when_enabled(voice_client, monkeypatch):
    client, _web_server = voice_client
    monkeypatch.setenv("HERMES_VOICE_MEET_INVITES", "1")

    resp = client.post(
        "/api/voice/meet/invite",
        json={"call_id": "voice-call", "user": "deniz"},
        headers={"host": "dashboard.local:9119"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["mode"] == "meet"
    assert body["call_id"] == "voice-call"
    assert "mode=meet" in body["invite_url"]
    assert "call_id=voice-call" in body["invite_url"]


def test_voice_call_end_marks_task_and_sends_single_post_call_notification(voice_client, monkeypatch):
    client, web_server = voice_client
    sent = []
    task = web_server.VoiceTask("vt_done", "call-ended", "deniz", "do work", "voice_task_vt_done")
    with web_server._VOICE_TASKS_LOCK:
        web_server._VOICE_TASKS[task.task_id] = task

    monkeypatch.setattr(web_server, "_voice_send_post_call_notification", lambda task: sent.append(task.task_id) or True)

    end_resp = client.post(
        "/api/voice/transcript",
        json={"call_id": "call-ended", "role": "system", "text": "ended", "event_type": "call_end", "user": "deniz"},
    )
    assert end_resp.status_code == 200

    web_server._voice_maybe_notify_post_call(task, "complete")
    web_server._voice_maybe_notify_post_call(task, "complete")

    assert sent == ["vt_done"]
    assert task.to_dict()["call_ended"] is True
    assert task.to_dict()["post_call_notification"]["status"] == "sent"
