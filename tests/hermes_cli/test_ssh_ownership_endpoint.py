from fastapi.testclient import TestClient

from hermes_cli import web_server


def test_ssh_ownership_endpoint_requires_token_and_returns_exact_nonce(monkeypatch):
    token = "t" * 64
    nonce = "0123456789abcdef"
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(web_server, "_SSH_OWNER_NONCE", nonce)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)

    assert client.get("/api/ssh/ownership").status_code == 401
    response = client.get(
        "/api/ssh/ownership",
        headers={"X-Hermes-Session-Token": token},
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "sshOwnerNonce": nonce,
        "protocolVersion": 1,
    }


def test_ssh_ownership_endpoint_is_absent_without_owner_nonce(monkeypatch):
    token = "t" * 64
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(web_server, "_SSH_OWNER_NONCE", None)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)

    response = client.get(
        "/api/ssh/ownership",
        headers={"X-Hermes-Session-Token": token},
    )
    assert response.status_code == 404
