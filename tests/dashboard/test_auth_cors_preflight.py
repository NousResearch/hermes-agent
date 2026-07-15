"""Regression tests for CORS-preflight handling in the dashboard auth gate.

Issue #59052: ``CORSMiddleware`` is registered first, so it ends up
*innermost* and runs after ``auth_middleware``. A browser CORS preflight
(``OPTIONS``) carries no session token, so the token check returned 401
before ``CORSMiddleware`` could answer with the preflight headers, and the
SPA's cross-origin requests to protected ``/api/`` routes failed.

The contract these tests pin down:

  * A genuine preflight (``OPTIONS`` + ``Access-Control-Request-Method``)
    to a protected route short-circuits the token check and is answered by
    ``CORSMiddleware`` with the CORS headers (not 401).
  * A non-``OPTIONS`` request to a protected route without a token still
    returns 401 (the auth gate is otherwise unchanged).
  * A bare ``OPTIONS`` with no preflight header does NOT bypass auth — it
    falls through to the normal gate, so it cannot be used to probe
    protected routes without credentials.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server

# A route under ``/api/`` that is NOT in ``PUBLIC_API_PATHS`` — i.e. the auth
# gate enforces the session token on it.
_PROTECTED_PATH = "/api/config"
_ORIGIN = "http://localhost:3000"


@pytest.fixture
def client():
    """A TestClient with the legacy (loopback) auth gate active.

    ``auth_required`` False keeps the OAuth gate a no-op so ``auth_middleware``
    is the authority — the path exercised by the fix. Restored afterwards so
    the flag does not leak into other tests.
    """
    saved = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    yield TestClient(web_server.app)
    web_server.app.state.auth_required = saved


def test_genuine_preflight_is_answered_by_cors_not_401(client):
    resp = client.request(
        "OPTIONS",
        _PROTECTED_PATH,
        headers={"Origin": _ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == _ORIGIN


def test_protected_route_without_token_still_401(client):
    resp = client.get(_PROTECTED_PATH)
    assert resp.status_code == 401


def test_bare_options_without_preflight_header_does_not_bypass_auth(client):
    # No Access-Control-Request-Method → not a real preflight, so it must not
    # short-circuit the token check and hand back a 200.
    resp = client.request("OPTIONS", _PROTECTED_PATH)
    assert resp.status_code == 401
