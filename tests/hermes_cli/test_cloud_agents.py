"""``GET /api/cloud/agents``: the Webapp lists the signed-in account's Hermes Cloud agents through
this install's own Nous sign-in, since a browser tab holds no portal session of its own.

A real HTTP server stands in for Nous Portal; the route, the token resolution and the HTTP client
are the production ones.
"""

import base64
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def _jwt() -> str:
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return ".".join([b64({"alg": "none"}), b64({"sub": "user-1", "exp": int(time.time()) + 3600}), "sig"])


@pytest.fixture
def portal():
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append({"path": self.path, "auth": self.headers.get("Authorization", "")})
            if "org=" not in self.path and getattr(self.server, "multi_team", False):
                self._send(409, {"error": "org_selection_required",
                                 "orgs": [{"id": "o1", "slug": "a", "name": "A"}, {"id": "o2", "name": "B"}]})
                return
            self._send(200, {"org": {"id": "o1", "slug": "a", "name": "A"}, "agents": [
                {"id": "ag1", "name": "work", "status": "RUNNING", "dashboardUrl": "https://work.example.test",
                 "dashboardGatewayState": "active", "secretField": "never relayed"},
                {"id": "ag2", "name": "new", "status": "PROVISIONING"}]})

        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, seen, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def client(tmp_path, monkeypatch, portal):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from hermes_cli.web_routers import status

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", portal[2])
    expires = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    token = _jwt()
    (home / "auth.json").write_text(json.dumps({"version": 1, "active_provider": "nous", "providers": {"nous": {
        "portal_base_url": portal[2], "client_id": "hermes-cli", "token_type": "Bearer",
        "access_token": token, "refresh_token": "rt", "expires_at": expires, "obtained_at": expires,
        "agent_key": token, "agent_key_expires_at": expires}}}))
    app = FastAPI()
    app.include_router(status.router)
    return TestClient(app), token


def test_lists_the_accounts_agents_with_the_installs_own_sign_in(client, portal):
    http, token = client
    response = http.get("/api/cloud/agents")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["org"]["slug"] == "a"
    assert body["agents"] == [
        {"id": "ag1", "name": "work", "status": "RUNNING", "dashboardUrl": "https://work.example.test",
         "dashboardGatewayState": "active"},
        {"id": "ag2", "name": "new", "status": "PROVISIONING", "dashboardUrl": None, "dashboardGatewayState": "unknown"}]
    assert portal[1][-1] == {"path": "/api/agents", "auth": f"Bearer {token}"}


def test_a_multi_team_account_gets_the_team_picker_then_the_chosen_teams_agents(client, portal):
    http, _token = client
    portal[0].multi_team = True
    picker = http.get("/api/cloud/agents").json()
    assert picker == {"needsOrgSelection": True, "orgs": [
        {"id": "o1", "slug": "a", "name": "A", "isPersonal": False, "role": "MEMBER"},
        {"id": "o2", "slug": None, "name": "B", "isPersonal": False, "role": "MEMBER"}]}
    assert [a["id"] for a in http.get("/api/cloud/agents?org=a").json()["agents"]] == ["ag1", "ag2"]
    assert portal[1][-1]["path"] == "/api/agents?org=a"


def test_signed_out_install_is_told_to_sign_in_not_a_server_error(client, tmp_path):
    http, _token = client
    (tmp_path / "hermes" / "auth.json").unlink()
    response = http.get("/api/cloud/agents")
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "portal_signin_required"
