from fastapi.testclient import TestClient

from app.asgi import app


SIDECAR_TOKEN_HEADER = "X-Hermes-MoneyPrinter-Token"


def test_managed_sidecar_protects_api_and_task_files(monkeypatch):
    monkeypatch.setenv("MONEYPRINTER_HERMES_TOKEN", "managed-test-token")
    client = TestClient(app)

    assert client.get("/api/v1/tasks").status_code == 401
    assert client.get("/tasks/missing.mp4").status_code == 401


def test_managed_sidecar_identity_requires_token(monkeypatch):
    monkeypatch.setenv("MONEYPRINTER_HERMES_TOKEN", "managed-test-token")
    client = TestClient(app)

    assert client.get("/api/v1/hermes/health").status_code == 401

    response = client.get(
        "/api/v1/hermes/health",
        headers={SIDECAR_TOKEN_HEADER: "managed-test-token"},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "managed": True,
        "protocol_version": 1,
        "service": "moneyprinterturbo",
        "version": "1.3.0",
    }


def test_managed_sidecar_keeps_docs_available_for_diagnostics(monkeypatch):
    monkeypatch.setenv("MONEYPRINTER_HERMES_TOKEN", "managed-test-token")
    client = TestClient(app)

    assert client.get("/docs").status_code == 200
