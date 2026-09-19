"""Tests for CORS preflight ordering on the dashboard web server.

Regression: ``CORSMiddleware`` was registered *before* the
``@app.middleware("http")`` auth decorators, making it the innermost
layer in Starlette's onion model.  A cross-origin ``OPTIONS`` preflight
hit the auth middleware first → 401, and the CORS headers were never
sent.  Moving the ``add_middleware(CORSMiddleware, …)`` call *after*
all ``@app.middleware("http")`` registrations makes CORS outermost so
preflights are answered before auth runs.

See: https://github.com/NousResearch/hermes-agent/issues/59052
"""

from __future__ import annotations

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(web_server.app)


# ---- CORS preflight (OPTIONS) must not be blocked by auth ------------


def test_options_preflight_returns_cors_headers(client):
    """OPTIONS with Origin + Access-Control-Request-Method → 200 + CORS."""
    resp = client.options(
        "/api/skills",
        headers={
            "Origin": "http://127.0.0.1:8092",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-hermes-session-token",
        },
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers
    assert resp.headers["access-control-allow-origin"] == "http://127.0.0.1:8092"
    assert "access-control-allow-methods" in resp.headers


def test_options_preflight_allows_localhost_origin(client):
    """Preflight from a different localhost port is allowed."""
    resp = client.options(
        "/api/status",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_options_preflight_rejects_disallowed_origin(client):
    """Preflight from a non-localhost origin is rejected (no CORS headers)."""
    resp = client.options(
        "/api/skills",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette CORSMiddleware returns 400 for disallowed origins on preflight
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


def test_get_without_origin_has_no_cors_headers(client):
    """A same-origin GET (no Origin header) has no CORS headers."""
    resp = client.get(
        "/api/status",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_get_with_matching_origin_has_cors_headers(client):
    """A cross-origin GET from an allowed origin has CORS headers."""
    resp = client.get(
        "/api/status",
        headers={
            "Origin": "http://127.0.0.1:8092",
            web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:8092"
