"""Route-boundary security and process-restart recovery with real temporary state."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_state import SessionDB
from hermes_cli import web_server
import tui_gateway.server  # Import bootstrap before the test's real-home I/O guard is installed.
from tools.inline_artifact_tool import publish_html


def seed(home, sid="chat", text="Original", call_id="publish-call"):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("{}", encoding="utf-8")
    artifact = json.loads(publish_html({"title": "Report", "html": f"<h1>{text}</h1>", "fallback": text},
                                      session_id=sid))["artifact"]
    with SessionDB(db_path=home / "state.db") as db:
        if not db.get_session(sid):
            db.create_session(sid, source="gui")
        row_id = db.append_message(sid, "tool", content=json.dumps({"artifact": artifact}),
                                   tool_name="publish_html", tool_call_id=call_id,
                                   display_metadata={"inline_artifact": artifact})
    return artifact, row_id


@pytest.fixture
def client_state(tmp_path, monkeypatch):
    from hermes_cli import web_server
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: home / "profiles")
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", None, raising=False)
    with TestClient(web_server.app) as client:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        yield client, home


def test_cross_profile_and_exact_repeated_message_association(client_state):
    client, home = client_state
    artifact, first = seed(home)
    same, second = seed(home, call_id="second-call")
    assert artifact == same and first != second
    other, other_row = seed(home / "profiles" / "beta", text="Other profile")
    route = f"/api/sessions/chat/artifacts/{artifact['id']}"
    assert client.get(route, params={"profile": "beta"}).status_code == 404
    assert client.get(f"/api/sessions/chat/artifacts/{other['id']}").status_code == 404
    beta = client.get(f"/api/sessions/chat/artifacts/{other['id']}", params={"profile": "beta"})
    assert beta.status_code == 200, beta.text
    assert beta.json()["html"] == other["html"]
    listing = client.get("/api/sessions/chat/artifacts").json()
    assert listing["count"] == 2
    assert [item["association"]["row_id"] for item in listing["artifacts"]] == [first, second]
    response = client.get(route, params={"row_id": second})
    assert response.json()["association"]["tool_call_id"] == "second-call"
    assert response.json()["association"]["row_id"] == second
    assert client.get(route, params={"row_id": second + 100}).status_code == 404
    assert client.post(route, json={"html": "replace"}).status_code == 405


def test_deleted_session_and_rewind_do_not_leak_artifacts(client_state):
    client, home = client_state
    artifact, row_id = seed(home)
    route = f"/api/sessions/chat/artifacts/{artifact['id']}"
    assert client.get(route).status_code == 200
    with SessionDB(db_path=home / "state.db") as db:
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET active=0, compacted=0 WHERE id=?", (row_id,)))
    assert client.get(route).status_code == 404
    assert client.get("/api/sessions/chat/artifacts").json()["count"] == 0
    assert client.get(route.replace("chat", "missing")).status_code == 404


def test_compacted_artifact_is_retrievable_without_old_runtime(client_state):
    client, home = client_state
    artifact, row_id = seed(home)
    with SessionDB(db_path=home / "state.db") as db:
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET active=0, compacted=1 WHERE id=?", (row_id,)))
    result = client.get(f"/api/sessions/chat/artifacts/{artifact['id']}")
    assert result.status_code == 200, result.text
    assert result.json()["html"] == artifact["html"]
    assert client.get("/api/sessions/chat/artifacts").json()["count"] == 1


def test_artifact_route_uses_existing_oauth_cookie_gate(client_state, monkeypatch):
    _client, home = client_state
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
    from tests.hermes_cli.test_dashboard_auth_middleware import _complete_stub_login
    artifact, row_id = seed(home)
    clear_providers()
    register_provider(StubAuthProvider())
    monkeypatch.setattr(web_server.app.state, "auth_required", True)
    monkeypatch.setattr(web_server.app.state, "bound_host", "artifact.example.test")
    monkeypatch.setattr(web_server.app.state, "bound_port", 443, raising=False)
    path = f"/api/sessions/chat/artifacts/{artifact['id']}"
    try:
        with TestClient(web_server.app, base_url="https://artifact.example.test") as client:
            client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
            assert client.get(path).status_code == 401
            assert client.get("/api/sessions/chat/artifacts").status_code == 401
            del client.headers[web_server._SESSION_HEADER_NAME]
            _complete_stub_login(client)
            response = client.get(path)
            assert response.status_code == 200, response.text
            assert response.json()["association"]["row_id"] == row_id
            assert response.json()["html"] == artifact["html"]
    finally:
        clear_providers()


def test_invalid_canonical_metadata_is_not_a_renderable_artifact(client_state):
    client, home = client_state
    artifact, row_id = seed(home)
    with SessionDB(db_path=home / "state.db") as db:
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET display_metadata=? WHERE id=?",
            (json.dumps({"inline_artifact": {"id": artifact["id"]}}), row_id)))
    assert client.get("/api/sessions/chat/artifacts").json()["count"] == 0
    assert client.get(f"/api/sessions/chat/artifacts/{artifact['id']}").status_code == 404


def test_process_restart_reopens_canonical_artifact_and_does_not_resurrect_delegates(tmp_path):
    home = tmp_path / "restart"
    artifact, row_id = seed(home)
    # A fresh process imports a fresh web app, SessionDB and delegation registry.
    script = '''
import json, sys
output = sys.stdout
from hermes_cli import web_server
from starlette.testclient import TestClient
from tools.delegate_tool_registry import list_active_subagents
web_server.app.state.auth_required = False
web_server.app.state.bound_host = None
with TestClient(web_server.app) as client:
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    response = client.get(sys.argv[1])
    print(json.dumps({"status": response.status_code, "body": response.json(), "delegates": list_active_subagents()}), file=output)
'''
    env = {**os.environ, "HERMES_HOME": str(home)}
    path = f"/api/sessions/chat/artifacts/{artifact['id']}"
    for _ in range(2):
        proc = subprocess.run([sys.executable, "-c", script, path], cwd=Path(__file__).resolve().parents[2],
                              env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == 200, result
        assert result["body"]["html"] == artifact["html"]
        assert result["body"]["association"]["row_id"] == row_id
        assert result["delegates"] == []
