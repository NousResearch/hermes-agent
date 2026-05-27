"""Security regression tests for the Hermes dashboard FastAPI app."""

from fastapi.testclient import TestClient

from hermes_cli import web_server


def test_dashboard_auth_uses_raw_asgi_path_for_badhost_malformed_host(monkeypatch):
    """Malformed Host headers must not poison the API auth path check.

    CVE-2026-48710 / BadHost affects Starlette < 1.0.1 when middleware bases
    authorization on ``request.url.path``. A Host value containing ``?`` can make
    Starlette reconstruct ``request.url.path`` as a public path while the ASGI
    router still dispatches the real protected path. The dashboard auth gate
    must use the raw ASGI path instead.
    """

    monkeypatch.setattr(web_server.app.state, "bound_host", "0.0.0.0", raising=False)
    client = TestClient(web_server.app)

    response = client.get(
        "/api/config",
        headers={"Host": "127.0.0.1/api/config/defaults?"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_dashboard_auth_still_allows_public_api_without_token(monkeypatch):
    """The BadHost fix must not accidentally close intended public endpoints."""

    monkeypatch.setattr(web_server.app.state, "bound_host", "0.0.0.0", raising=False)
    client = TestClient(web_server.app)

    response = client.get("/api/config/defaults", headers={"Host": "127.0.0.1"})

    assert response.status_code == 200
    assert isinstance(response.json(), dict)
