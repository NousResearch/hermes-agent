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
        lambda user: {
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
    assert config["tools"][0]["name"] == "rolly"
    turn_detection = config["audio"]["input"]["turn_detection"]
    assert turn_detection == {
        "type": "semantic_vad",
        "create_response": True,
        "interrupt_response": True,
    }


def test_voice_tool_rejects_unknown_tool(voice_client):
    client, _web_server = voice_client

    resp = client.post("/api/voice/tool", json={"name": "shell", "arguments": {}})
    assert resp.status_code == 400


def test_voice_tool_runs_research_bridge(voice_client, monkeypatch):
    client, web_server = voice_client

    monkeypatch.setattr(
        web_server,
        "_run_voice_research",
        lambda question, user: f"answered: {question}",
    )

    resp = client.post(
        "/api/voice/tool",
        json={"name": "research", "arguments": {"question": "what is Rolly Voice?"}},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "result": "answered: what is Rolly Voice?",
        "error": None,
    }


def test_voice_tool_runs_rolly_bridge_with_request_arg(voice_client, monkeypatch):
    client, web_server = voice_client

    monkeypatch.setattr(
        web_server,
        "_run_voice_research",
        lambda question, user: f"answered: {user}: {question}",
    )

    resp = client.post(
        "/api/voice/tool",
        json={"name": "rolly", "arguments": {"request": "what were we doing yesterday?"}},
        headers={"X-Rolly-User": "deniz"},
    )
    assert resp.status_code == 200
    assert resp.json()["result"] == "answered: deniz: what were we doing yesterday?"


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
