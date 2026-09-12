"""Authenticated service intents must guard process creation, including retries."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def service_client(monkeypatch, _isolate_hermes_home, tmp_path):
    import hermes_state
    import hermes_cli.web_server as server
    import hermes_cli.web_server_gateway as gateway
    import hermes_cli.web_routers.actions as actions
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    for name in ("_ACTION_PROCS", "_ACTION_COMMANDS", "_ACTION_IDS", "_ACTION_RESULTS"):
        monkeypatch.setattr(gateway, name, {})
    monkeypatch.setattr(gateway, "_ACTION_LOG_DIR", tmp_path / "actions")
    monkeypatch.setattr(server, "_LAST_GATEWAY_RESTART", None)
    monkeypatch.setattr(actions, "_SERVICE_MUTATION_INTENTS", {}, raising=False)
    monkeypatch.setattr("hermes_cli.gateway._reap_unsupervised_gateway_orphans", lambda: None)
    monkeypatch.setattr("hermes_cli.web_server_files._dashboard_local_update_managed_externally", lambda: False)
    monkeypatch.setattr("hermes_cli.update_contract.evaluate_update_admission", lambda *_: None)
    (home / "profiles" / "coder").mkdir(parents=True, exist_ok=True)
    (home / "profiles" / "coder" / "config.yaml").write_text("{}\n", encoding="utf-8")
    spawned = []

    def spawn(argv, **kwargs):
        proc = SimpleNamespace(pid=1000 + len(spawned), returncode=None, argv=argv, kwargs=kwargs)
        proc.poll = lambda: proc.returncode
        spawned.append(proc)
        return proc

    monkeypatch.setattr(gateway.subprocess, "Popen", spawn)
    client = TestClient(server.app)
    client.headers[server._SESSION_HEADER_NAME] = server._SESSION_TOKEN

    def audit():
        path = home / "logs" / "dashboard-auth.log"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    yield SimpleNamespace(client=client, spawned=spawned, gateway=gateway, server=server, audit=audit)
    client.close()


def intent(action, key="confirmed-request-123456"):
    return {"confirmation": "RESTART" if action == "gateway-restart" else "UPDATE", "idempotency_key": key}


def endpoint(action):
    return "/api/gateway/restart" if action == "gateway-restart" else "/api/hermes/update"


@pytest.mark.parametrize("action", ["gateway-restart", "hermes-update"])
@pytest.mark.parametrize("payload", [None, [], "RESTART", {}, {"confirmation": "wrong"}, {"confirmation": "RESTART", "idempotency_key": "short"}])
def test_service_actions_reject_incomplete_intent_before_spawning(service_client, action, payload):
    ctx = service_client
    response = ctx.client.post(endpoint(action), json=payload)
    assert response.status_code == 400
    assert ctx.spawned == []
    event = ctx.audit()[-1]
    assert event["event"] == "dashboard_mutation"
    assert event["action"] == action
    assert event["result"] == "rejected"
    assert event["actor"] == "local-dashboard"


@pytest.mark.parametrize("action", ["gateway-restart", "hermes-update"])
@pytest.mark.parametrize("scenario", ["retry", "different-key", "other-action", "profile", "completed", "spawn-failure", "unsupported", "concurrent-same", "concurrent-other", "status-respawn"])
def test_service_intents_bind_retries_and_conflicts_to_the_running_action(service_client, monkeypatch, action, scenario):
    ctx = service_client
    url = endpoint(action)
    if scenario == "status-respawn":
        first = ctx.client.post(url, json=intent(action))
        assert first.status_code == 200
        old = ctx.spawned[0]
        old.returncode = 0
        entered = threading.Event()
        resume = threading.Event()

        def wait_old(*_args, **_kwargs):
            entered.set()
            assert resume.wait(5)
            return 0

        old.wait = wait_old
        if action == "gateway-restart":
            started_at, proc, command = ctx.server._LAST_GATEWAY_RESTART
            monkeypatch.setattr(ctx.server, "_LAST_GATEWAY_RESTART", (started_at - 11, proc, command))
        with ThreadPoolExecutor(max_workers=1) as pool:
            status = pool.submit(ctx.client.get, f"/api/actions/{action}/status")
            try:
                assert entered.wait(5)
                second = ctx.client.post(url, json=intent(action, "replacement-request-12345"))
                assert second.status_code == 200
                assert len(ctx.spawned) == 2
            finally:
                resume.set()
            assert status.result(timeout=5).status_code == 200
        retry = ctx.client.post(url, json=intent(action, "replacement-request-12345"))
        assert retry.status_code == 200
        assert retry.json()["pid"] == second.json()["pid"]
        assert len(ctx.spawned) == 2
        assert ctx.gateway._ACTION_PROCS[action] is ctx.spawned[-1]
        return
    if scenario.startswith("concurrent"):
        original_spawn = ctx.gateway.subprocess.Popen
        start = threading.Barrier(2)

        def delayed_spawn(*args, **kwargs):
            time.sleep(0.05)  # leave a real overlap between request threads
            return original_spawn(*args, **kwargs)

        def request(key):
            start.wait(timeout=5)
            return ctx.client.post(url, json=intent(action, key))

        monkeypatch.setattr(ctx.gateway.subprocess, "Popen", delayed_spawn)
        second_key = "another-confirmed-request-123" if scenario == "concurrent-other" else "confirmed-request-123456"
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(request, "confirmed-request-123456")
            second = pool.submit(request, second_key)
            responses = [first.result(timeout=5), second.result(timeout=5)]
        assert sorted(response.status_code for response in responses) == ([200, 409] if scenario == "concurrent-other" else [200, 200])
        assert len(ctx.spawned) == 1
        return
    if scenario == "spawn-failure":
        def fail_spawn(*_args, **_kwargs):
            raise OSError("fixture spawn failure")
        monkeypatch.setattr(ctx.gateway.subprocess, "Popen", fail_spawn)
        response = ctx.client.post(url, json=intent(action))
        assert response.status_code == 500
        assert ctx.audit()[-1]["result"] == "failed"
        return
    if scenario == "unsupported":
        if action == "gateway-restart":
            pytest.skip("Update admission applies only to the updater")
        monkeypatch.setattr("hermes_cli.web_server_files._dashboard_local_update_managed_externally", lambda: True)
        response = ctx.client.post(url, json=intent(action))
        assert response.status_code == 200 and response.json()["ok"] is False
        assert ctx.spawned == []
        assert ctx.audit()[-1]["result"] == "unsupported"
        return

    first = ctx.client.post(url, json={**intent(action), "secret": "do-not-audit-this-value"})
    assert first.status_code == 200
    assert len(ctx.spawned) == 1
    other_action = "hermes-update" if action == "gateway-restart" else "gateway-restart"
    next_url = endpoint(other_action) if scenario == "other-action" else url
    next_body = intent(other_action) if scenario == "other-action" else intent(action)
    if scenario == "different-key":
        next_body = intent(action, "another-confirmed-request-123")
    if scenario == "profile":
        if action == "hermes-update":
            next_url = endpoint(other_action) + "?profile=coder"
            next_body = intent(other_action)
        else:
            next_url += "?profile=coder"
    if scenario == "completed":
        ctx.spawned[0].returncode = 0
        # The restart cooldown remains authoritative after the child exits.
        next_body = intent(action, "another-confirmed-request-123")

    second = ctx.client.post(next_url, json=next_body)
    conflict = scenario in {"different-key", "other-action", "profile"}
    assert second.status_code == (409 if conflict else 200)
    assert len(ctx.spawned) == (2 if scenario == "completed" and action == "hermes-update" else 1)
    if scenario == "retry" or (scenario == "completed" and action == "gateway-restart"):
        assert second.json()["pid"] == first.json()["pid"]
    if scenario == "retry" and action == "hermes-update":
        assert second.json()["action_id"] == first.json()["action_id"]
    events = ctx.audit()
    assert events[0]["result"] == "started"
    assert events[-1]["result"] == ("conflict" if conflict else "started" if len(ctx.spawned) == 2 else "reused")
    assert "do-not-audit-this-value" not in json.dumps(events)
    assert "confirmed-request-123456" not in json.dumps(events)
