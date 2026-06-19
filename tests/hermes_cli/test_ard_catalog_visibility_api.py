from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server

pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")


@pytest.fixture
def client_loopback():
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _catalog(visibility: str):
    entries = []
    if visibility == "private":
        entries.append({"identifier": "urn:ai:test:mcp:local", "url": "stdio:local", "type": "application/mcp-server-card+json"})
    return {"schemaVersion": "1.0", "visibility": visibility, "entries": entries}


def test_api_ard_catalog_accepts_private_visibility(client_loopback, monkeypatch):
    calls = []

    def fake_generate_ard_catalog(*, visibility="public", **kwargs):
        calls.append(visibility)
        return _catalog(visibility)

    monkeypatch.setattr("tools.skills_hub.generate_ard_catalog", fake_generate_ard_catalog)
    r = client_loopback.get(
        "/api/ard/catalog?visibility=private",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
    )
    assert r.status_code == 200
    assert calls == ["private"]
    body = r.json()
    assert body["visibility"] == "private"
    assert body["entries"][0]["url"].startswith("stdio:")


def test_api_ard_catalog_rejects_invalid_visibility(client_loopback):
    r = client_loopback.get(
        "/api/ard/catalog?visibility=internal",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
    )
    assert r.status_code == 400


def test_well_known_catalog_stays_public_only(client_loopback, monkeypatch, tmp_path):
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    calls = []

    def fake_publish_ard_catalog(*, visibility="public", **kwargs):
        calls.append(("publish", visibility))
        p = tmp_path / ".well-known" / "ai-catalog.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        import json
        p.write_text(json.dumps(_catalog(visibility)))
        return p

    def fake_generate_ard_catalog(*, visibility="public", **kwargs):
        calls.append(("generate", visibility))
        return _catalog(visibility)

    monkeypatch.setattr("tools.skills_hub.publish_ard_catalog", fake_publish_ard_catalog)
    monkeypatch.setattr("tools.skills_hub.generate_ard_catalog", fake_generate_ard_catalog)
    r = client_loopback.get("/.well-known/ai-catalog.json?visibility=private")
    assert r.status_code == 200
    assert r.json()["visibility"] == "public"
    assert all(visibility == "public" for _, visibility in calls)


def test_api_ard_publish_accepts_private_visibility_and_output(client_loopback, monkeypatch, tmp_path):
    calls = []
    out = tmp_path / "private.json"

    def fake_publish_ard_catalog(*, domain="hermes.local", visibility="public", output_path=None):
        calls.append({"domain": domain, "visibility": visibility, "output_path": str(output_path)})
        out.write_text('{"entries": [{"type": "application/mcp-server-card+json"}]}')
        return out

    monkeypatch.setattr("tools.skills_hub.publish_ard_catalog", fake_publish_ard_catalog)
    r = client_loopback.post(
        "/api/ard/publish",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        json={"domain": "example.test", "visibility": "private", "output_path": str(out)},
    )
    assert r.status_code == 200
    assert calls == [{"domain": "example.test", "visibility": "private", "output_path": str(out)}]
    assert r.json()["visibility"] == "private"


def test_api_ard_publish_rejects_invalid_visibility(client_loopback):
    r = client_loopback.post(
        "/api/ard/publish",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        json={"visibility": "internal"},
    )
    assert r.status_code == 400


def test_api_ard_inspect_returns_entry_risk_and_source(client_loopback, monkeypatch):
    def fake_inspect_identifier(identifier):
        return {
            "ok": True,
            "identifier": identifier,
            "source_catalog": "/tmp/catalog.json",
            "entry": {"identifier": identifier, "displayName": "demo"},
            "risk": {"decision": "allow", "risk": "low", "next_action": "manual_register_after_review"},
            "install_performed": False,
        }

    monkeypatch.setattr("scripts.ard_inspect.inspect_identifier", fake_inspect_identifier)
    r = client_loopback.post(
        "/api/ard/inspect",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        json={"identifier": "urn:ai:test:demo"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["entry"]["displayName"] == "demo"
    assert body["risk"]["decision"] == "allow"
    assert body["install_performed"] is False


def test_api_ard_inspect_requires_identifier(client_loopback):
    r = client_loopback.post(
        "/api/ard/inspect",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        json={},
    )
    assert r.status_code == 400
