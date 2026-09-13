"""Only authenticated owner renewal may extend a still-valid viewer ticket."""
from pathlib import Path
import runpy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PLUGIN = Path(__file__).resolve().parents[2] / "plugins/hermes-realms"
load = runpy.run_path(str(PLUGIN / "realms/_binding.py"))["load_runtime"]


def test_renewal_preserves_scope_privilege_expiry_and_revocation():
    now = [0.0]
    tickets = load("viewer_auth").Tickets(clock=lambda: now[0])
    token = tickets.issue("owner", "generation", ttl=10, can_control=False)
    now[0] = 9
    assert not tickets.renew(token, "other", "generation", ttl=10)
    assert not tickets.renew(token, "owner", "replacement", ttl=10)
    assert tickets.renew(token, "owner", "generation", ttl=10)
    now[0] = 11
    assert tickets.check(token, "owner", "generation")
    assert not tickets.check(token, "owner", "generation", control=True)
    now[0] = 19
    assert not tickets.renew(token, "owner", "generation", ttl=10)
    assert not tickets.check(token, "owner", "generation")
    token = tickets.issue("owner", "generation", ttl=10)
    tickets.revoke("owner")
    assert not tickets.renew(token, "owner", "generation", ttl=10)


@pytest.mark.linux_only
@pytest.mark.parametrize("kind,realm_id", [("realm", "r-fixture"), ("omarchy-vm", "v-fixture")])
def test_renew_route_preserves_owner_locality_and_generation(tmp_path, monkeypatch, kind, realm_id):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    api = runpy.run_path(str(PLUGIN / "dashboard/plugin_api.py"))
    service = api["get_integration"]()
    service.bind(session_id="owner", stored_session_id="history")
    service.bind(session_id="other", stored_session_id="other-history")
    record = {"id": realm_id, "kind": kind, "session_id": "owner", "status": "running", "generation": "original"}
    monkeypatch.setattr(service.manager, "list", lambda: [record] if kind == "realm" else [])
    monkeypatch.setattr(service.vm, "list", lambda: [record] if kind == "omarchy-vm" else [])
    bridge_module = load("bridge")
    viewer = bridge_module.ViewerServer(lambda _: record)
    now = [0.0]
    viewer.tickets = load("viewer_auth").Tickets(clock=lambda: now[0])
    monkeypatch.setattr(bridge_module, "get_profile_viewer", lambda home: viewer)
    token = viewer.issue(realm_id, ttl=10, can_control=True)
    app = FastAPI()
    app.include_router(api["router"])
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1234)) as client:
        path = f"/realms/{realm_id}/renew"
        body = {"stored_session_id": "history", "viewer_token": token}
        assert client.post(path, json={**body, "stored_session_id": "other-history"}).status_code == 403
        assert client.post(path, json=body, headers={"forwarded": "for=external"}).status_code == 503
        now[0] = 9
        response = client.post(path, json=body)
        assert response.status_code == 200
        assert response.json() == {"renewed": True}
        assert response.headers["cache-control"] == "no-store"
        now[0] = 11
        assert viewer.tickets.check(token, realm_id, viewer._ticket_generation(record), control=True)
        record["generation"] = "replacement"
        assert client.post(path, json=body).status_code == 403
        record["generation"] = "original"
        viewer.revoke(realm_id)
        assert client.post(path, json=body).status_code == 403
        assert client.post(path, json={**body, "stored_session_id": "unknown"}).status_code == 403
