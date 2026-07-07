"""Regression tests for the password-only-provider NotImplementedError crash.

https://github.com/NousResearch/hermes-agent/pull/56886

Covers the two broken entry points:
1. First unauthenticated dashboard hit with a single password-only provider
   should land on /login (credential form), not auto-redirect into
   /auth/login which calls start_login() and raises NotImplementedError.
2. Direct /auth/login?provider=<password-provider> (bookmark, manual URL,
   logout path) should redirect to /login, not call start_login() and 500.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    clear_providers,
    register_provider,
)
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
from tests.hermes_cli.test_dashboard_auth_password_login import PasswordProvider


@pytest.fixture
def pw_only_app():
    clear_providers()
    register_provider(PasswordProvider())
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    web_server.app.state.auth_required = prev_required


class TestAutoSSOBypassForPasswordProvider:
    """First unauthenticated dashboard hit with a single password-only
    provider should land on /login (credential form), not auto-redirect
    into /auth/login which calls start_login() and raises NotImplementedError.
    """

    def test_first_hit_lands_on_login_not_500(self, pw_only_app):
        resp = pw_only_app.get("/", follow_redirects=False)
        assert resp.status_code in (302, 303)
        assert "/login" in resp.headers["location"]
        assert "/auth/login" not in resp.headers["location"]

    def test_first_hit_preserves_next_param(self, pw_only_app):
        resp = pw_only_app.get("/sessions", follow_redirects=False)
        assert resp.status_code in (302, 303)
        location = resp.headers["location"]
        assert "/login" in location
        assert "next=" in location

    def test_first_hit_followed_redirect_renders_login_page(self, pw_only_app):
        resp = pw_only_app.get("/", follow_redirects=True)
        assert resp.status_code == 200
        assert "provider-form" in resp.text or "password" in resp.text.lower()


class TestAuthLoginRecoveryForPasswordProvider:
    """Direct /auth/login?provider=<password-provider> (bookmark, manual URL,
    logout path) should redirect to /login, not call start_login() and 500.
    """

    def test_direct_auth_login_redirects_to_login_not_500(self, pw_only_app):
        resp = pw_only_app.get(
            "/auth/login?provider=testpw", follow_redirects=False
        )
        assert resp.status_code in (302, 303)
        assert "/login" in resp.headers["location"]

    def test_direct_auth_login_with_safe_next(self, pw_only_app):
        resp = pw_only_app.get(
            "/auth/login?provider=testpw&next=/sessions",
            follow_redirects=False,
        )
        assert resp.status_code in (302, 303)
        location = resp.headers["location"]
        assert "/login" in location
        assert "next=%2Fsessions" in location or "next=/sessions" in location

    def test_direct_auth_login_with_open_redirect_next_drops_it(
        self, pw_only_app
    ):
        resp = pw_only_app.get(
            "/auth/login?provider=testpw&next=https://evil.example/phish",
            follow_redirects=False,
        )
        assert resp.status_code in (302, 303)
        location = resp.headers["location"]
        assert "/login" in location
        assert "evil.example" not in location

    def test_direct_auth_login_with_protocol_relative_next_drops_it(
        self, pw_only_app
    ):
        resp = pw_only_app.get(
            "/auth/login?provider=testpw&next=//evil.example",
            follow_redirects=False,
        )
        assert resp.status_code in (302, 303)
        location = resp.headers["location"]
        assert "/login" in location
        assert "evil.example" not in location

    def test_direct_auth_login_no_next_no_query(self, pw_only_app):
        resp = pw_only_app.get(
            "/auth/login?provider=testpw",
            follow_redirects=False,
        )
        assert resp.status_code in (302, 303)
        location = resp.headers["location"]
        assert "/login" in location
        assert "next=" not in location

    def test_oauth_provider_still_works_via_auth_login(self):
        """Ensure the OAuth path is unaffected -- StubAuthProvider
        (supports_password=False) should still start_login normally.
        """
        clear_providers()
        register_provider(StubAuthProvider())
        prev = getattr(web_server.app.state, "auth_required", None)
        web_server.app.state.auth_required = True
        try:
            client = TestClient(
                web_server.app, base_url="https://fly-app.fly.dev"
            )
            resp = client.get(
                "/auth/login?provider=stub", follow_redirects=False
            )
            assert resp.status_code == 302
            assert "/login" not in resp.headers["location"]
            assert "code=stub_code" in resp.headers["location"]
        finally:
            clear_providers()
            web_server.app.state.auth_required = prev
