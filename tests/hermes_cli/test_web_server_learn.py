from __future__ import annotations

from fastapi.testclient import TestClient


def _client():
    from hermes_cli import web_server

    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


def test_learn_status_endpoint_uses_profile_home(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    client = _client()
    try:
        response = client.get("/api/learn/status")
    finally:
        client.close()

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "off"
    assert data["state"] == "stopped"
    assert data["hermes_home"] == str(home)
    assert data["storage_path"] == str(home / "learn")


def test_learn_start_pause_resume_stop_and_delete_data(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    client = _client()
    try:
        started = client.post("/api/learn/start", json={"mode": "learn"}).json()
        assert started["mode"] == "learn"
        assert started["state"] == "running"

        paused = client.post("/api/learn/pause").json()
        assert paused["state"] == "paused"

        resumed = client.post("/api/learn/resume").json()
        assert resumed["state"] == "running"

        stopped = client.post("/api/learn/stop").json()
        assert stopped["state"] == "stopped"

        events_file = home / "learn" / "events.jsonl"
        events_file.write_text('{"kind":"app"}\n', encoding="utf-8")
        deleted = client.delete("/api/learn/data").json()
    finally:
        client.close()

    assert deleted["mode"] == "learn"
    assert deleted["state"] == "stopped"
    assert deleted["collected_event_count"] == 0
    assert not (home / "learn" / "events.jsonl").exists()


def test_learn_runtime_controls_and_review_suggestions(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import analyzer, runtime

    calls = []
    monkeypatch.setattr(runtime, "ensure_running", lambda: calls.append("start"))
    monkeypatch.setattr(runtime, "stop_runtime", lambda: calls.append("stop"))
    monkeypatch.setattr(
        analyzer,
        "create_usage_suggestions",
        lambda: [{"id": "learn1", "title": "Daily communication follow-up summary", "status": "pending"}],
    )

    client = _client()
    try:
        assert client.post("/api/learn/start", json={"mode": "learn"}).status_code == 200
        assert client.post("/api/learn/pause").status_code == 200
        assert client.post("/api/learn/resume").status_code == 200
        assert client.post("/api/learn/stop").status_code == 200
        reviewed = client.post("/api/learn/suggestions").json()
    finally:
        client.close()

    assert calls == ["start", "stop", "start", "stop"]
    assert reviewed["created_count"] == 1
    assert reviewed["suggestions"][0]["status"] == "pending"


def test_learn_config_endpoint_updates_collection_controls(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    client = _client()
    try:
        response = client.put(
            "/api/learn/config",
            json={
                "allowlist": ["code.exe", "chrome.exe"],
                "denylist": ["slack.exe"],
                "retention_days": 21,
            },
        )
    finally:
        client.close()

    assert response.status_code == 200
    data = response.json()
    assert data["allowlist"] == ["code.exe", "chrome.exe"]
    assert data["denylist"] == ["slack.exe"]
    assert data["retention_days"] == 21
    assert data["storage_path"] == str(home / "learn")


def test_learn_start_rejects_unknown_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))

    client = _client()
    try:
        response = client.post("/api/learn/start", json={"mode": "screenshots"})
    finally:
        client.close()

    assert response.status_code == 400
    assert "mode" in response.json()["detail"]


def test_learn_start_rejects_future_modes_until_implemented(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))

    client = _client()
    try:
        response = client.post("/api/learn/start", json={"mode": "teach"})
    finally:
        client.close()

    assert response.status_code == 400
    assert "learn" in response.json()["detail"]
