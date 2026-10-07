"""``/api/auth/providers`` must be public under BOTH dashboard gates.

Native clients (Hermes Conduit, and the Desktop's own pre-login probes) call
``GET /api/auth/providers`` as the *first* step of their auth flow — before any
credential exists. Under the OAuth gate it was already public
(``_GATE_PUBLIC_PREFIXES``), but the legacy session-token gate consulted only
``PUBLIC_API_PATHS``, so a loopback-bound dashboard (``auth_required=false``,
no session token) answered 401 and discovery could never complete (#134345).

The endpoint is login bootstrap data by design: provider ``name`` /
``display_name`` / ``supports_password`` flags (or 503 when none is
registered). It must stay in the shared allowlist so the two gates cannot
drift apart again.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def loopback_client():
    """Loopback bind with no session token — the legacy gate's territory."""
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8080
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:8080")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def test_providers_endpoint_is_in_the_shared_public_allowlist():
    """The allowlist both gates consult is the single source of truth."""
    assert "/api/auth/providers" in PUBLIC_API_PATHS


def test_providers_endpoint_answers_without_a_session_token_on_loopback(loopback_client):
    """The discovery call that runs *before* login must not require a token.

    Before the fix this returned 401 ``Unauthorized`` from the legacy
    middleware, so a native client's very first request failed and no
    credential could ever get past it.
    """
    r = loopback_client.get("/api/auth/providers", follow_redirects=False)
    assert r.status_code != 401, (
        "/api/auth/providers returned 401 without a session token — the "
        "pre-login discovery step is unreachable"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Bootstrap shape only: names and capability flags, never secrets.
    assert [p["name"] for p in body["providers"]] == ["stub"]
    assert all(set(p) <= {"name", "display_name", "supports_password"}
               for p in body["providers"])


def test_unrelated_api_routes_stay_gated_on_loopback(loopback_client):
    """Widening the allowlist must not open anything else on the legacy gate."""
    r = loopback_client.get("/api/sessions")
    assert r.status_code == 401
