import pytest


class FakeProcess:
    pid = 1234

    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


def test_harnesses_listing_and_confirmation_contract(monkeypatch, tmp_path):
    from starlette.testclient import TestClient
    import hermes_cli.web_server as web_server
    import hermes_cli.web_routers.harnesses as harnesses
    import hermes_cli.web_server_gateway as gateway

    harnesses._ACTIONS.clear()
    monkeypatch.setattr(harnesses, "_profile_rows", lambda: [("alpha", tmp_path)])
    monkeypatch.setattr(harnesses, "_runtime_for", lambda _home: {"platforms": {"telegram": {"state": "connected"}}})
    monkeypatch.setattr(harnesses, "_gateway_pid", lambda _home: 4321)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    response = client.get("/api/harnesses")
    assert response.status_code == 200
    assert response.json()["profiles"][0]["name"] == "alpha"
    assert response.json()["profiles"][0]["platforms"] == [{"name": "telegram", "state": "connected"}]

    response = client.post("/api/harnesses/alpha/actions/start", json={"confirmed": False})
    assert response.status_code == 400
    assert response.json()["detail"] == "confirmation_required"

    response = client.post("/api/harnesses/missing/actions/start", json={"confirmed": True})
    assert response.status_code == 404


def test_harness_action_status_is_invocation_bound(monkeypatch, tmp_path):
    from starlette.testclient import TestClient
    import hermes_cli.web_server as web_server
    import hermes_cli.web_routers.harnesses as harnesses
    import hermes_cli.web_server_gateway as gateway

    harnesses._ACTIONS.clear()
    process = FakeProcess()
    calls = []
    monkeypatch.setattr(harnesses, "_profile_rows", lambda: [("alpha", tmp_path)])
    monkeypatch.setattr(gateway, "multiplexed_profile_refusal", lambda *_args: None)
    monkeypatch.setattr(
        harnesses, "_spawn_hermes_action",
        lambda argv, name: calls.append((argv, name)) or process,
    )
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    response = client.post("/api/harnesses/alpha/actions/start", json={"confirmed": True})
    assert response.status_code == 200
    assert calls == [(["-p", "alpha", "gateway", "start"], "harness:alpha:start")]
    invocation_id = response.json()["invocation_id"]
    assert len(invocation_id) == 32

    assert client.get(
        "/api/harnesses/alpha/actions/start/status?invocation_id=wrong"
    ).json()["status"] == "unknown"
    assert client.get(
        f"/api/harnesses/alpha/actions/start/status?invocation_id={invocation_id}"
    ).json()["status"] == "running"

    response = client.post("/api/harnesses/alpha/actions/stop", json={"confirmed": True})
    assert response.status_code == 409
    assert response.json()["detail"] == "action_in_progress"


def test_harness_spawn_registers_extension_action_log(monkeypatch, tmp_path):
    import hermes_cli.web_server_gateway as gateway

    class SpawnedProcess:
        pid = 4321

        def poll(self):
            return None

    monkeypatch.setattr(gateway, "_ACTION_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(gateway, "_dashboard_spawn_executable", lambda: "/usr/bin/python3")
    monkeypatch.setattr(gateway, "_profile_action_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(gateway.subprocess, "Popen", lambda *args, **kwargs: SpawnedProcess())

    proc = gateway._spawn_hermes_action(
        ["-p", "alpha", "gateway", "start"], "harness:alpha:start"
    )
    assert proc.pid == 4321
    assert list((tmp_path / "logs").glob("*.log"))


def test_default_profile_action_environment_is_pinned(monkeypatch, tmp_path):
    import hermes_cli.web_server_gateway as gateway
    import hermes_cli.web_server_profiles as profiles

    target_home = tmp_path / "default"
    target_home.mkdir()
    monkeypatch.setattr(profiles, "_resolve_profile_dir", lambda _name: target_home)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "dashboard-profile"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dashboard-secret-must-not-inherit")

    action_env = gateway._profile_action_environment(
        ["-p", "default", "gateway", "start"]
    )
    assert action_env["HERMES_HOME"] == str(target_home)
    assert action_env["HERMES_NONINTERACTIVE"] == "1"
    assert "TELEGRAM_BOT_TOKEN" not in action_env
