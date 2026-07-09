"""Regression guard for PR #61281 (mobile/hosted dashboard OAuth).

The PR removed the *client-side* ``X-Hermes-Session-Token`` requirement from
the dashboard OAuth mutation calls (``web/src/lib/api.ts``) so that
cookie-authenticated hosted/mobile sessions can start provider logins. The
safety of that change rests entirely on the *server* still gating those
endpoints: in gated mode the ``gated_auth_middleware`` verifies the session
cookie before the handler runs, and ``_require_token`` defers to it.

These tests pin that server-side gate for the exact endpoints whose
client-side token gate was removed. Without them, a future change that
re-broke ``_require_token``'s gated-mode branch (e.g. letting it fall through
without a session) would still pass the PR's ``api.test.ts`` suite, because
those tests only mock ``fetch`` and never touch the server.
"""

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def gated_app():
    """A gated (``auth_required``) dashboard with no session cookie set."""
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


class TestOAuthMutationEndpointsGatedWithoutCookie:
    """No cookie in gated mode -> 401 on every endpoint whose client-side
    session-token gate PR #61281 removed."""

    def test_env_reveal_requires_cookie(self, gated_app):
        r = gated_app.post("/api/env/reveal", json={"key": "OPENAI_API_KEY"})
        assert r.status_code == 401

    def test_oauth_disconnect_requires_cookie(self, gated_app):
        r = gated_app.delete("/api/providers/oauth/anthropic")
        assert r.status_code == 401

    def test_oauth_start_requires_cookie(self, gated_app):
        r = gated_app.post("/api/providers/oauth/anthropic/start", json={})
        assert r.status_code == 401

    def test_oauth_submit_requires_cookie(self, gated_app):
        r = gated_app.post(
            "/api/providers/oauth/anthropic/submit",
            json={"session_id": "sid", "code": "abc"},
        )
        assert r.status_code == 401

    def test_oauth_cancel_session_requires_cookie(self, gated_app):
        r = gated_app.delete("/api/providers/oauth/sessions/sid")
        assert r.status_code == 401
