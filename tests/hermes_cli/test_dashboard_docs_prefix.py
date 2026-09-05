"""FastAPI's auto-generated API docs must live under the auth-gated ``/api/``
prefix.

By default FastAPI serves Swagger UI at ``/docs``, ReDoc at ``/redoc`` and the
raw OpenAPI schema at ``/openapi.json`` — all at the root, where the dashboard's
``auth_middleware`` (which only gates ``/api/*``) does not reach them. On a bind
to ``0.0.0.0`` or behind a reverse proxy that exposes the full interactive API
explorer and schema to anyone who can reach the host.

``web_server.app`` is constructed with ``docs_url``/``redoc_url``/``openapi_url``
relocated under ``/api/`` so all three sit behind the same gate as every other
``/api/*`` route. These tests assert the root paths are gone (404) and that the
relocated paths are now subject to the dashboard auth gate.
"""
import pytest


@pytest.fixture
def client():
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - exercised only when extras missing
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    return TestClient(app)


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_not_served_at_root(client, path):
    """The interactive docs and schema are no longer exposed at the root.

    The root paths fall outside the ``/api/*`` gate, so leaving them mapped
    would serve the API explorer and schema unauthenticated. They must 404.
    """
    resp = client.get(path)
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "path", ["/api/docs", "/api/redoc", "/api/openapi.json"]
)
def test_docs_gated_under_api_prefix_without_token(client, path):
    """Under the ``/api/`` prefix the docs are gated by ``auth_middleware``.

    Without a valid session token the gate returns 401 — the security win:
    the docs and schema are no longer reachable unauthenticated. They are
    deliberately absent from the public-paths allowlist.
    """
    from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS

    assert path not in PUBLIC_API_PATHS

    resp = client.get(path, headers={"X-Hermes-Session-Token": "wrong-token"})
    assert resp.status_code == 401


@pytest.mark.parametrize(
    "path", ["/api/docs", "/api/redoc", "/api/openapi.json"]
)
def test_docs_reachable_under_api_prefix_with_token(client, path):
    """With a valid session token the relocated docs/schema are served.

    Confirms the relocation didn't break the docs for an authenticated caller
    (e.g. the dashboard SPA / desktop shell): the route still exists, it's just
    behind the gate now.
    """
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

    resp = client.get(path, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    assert resp.status_code == 200
