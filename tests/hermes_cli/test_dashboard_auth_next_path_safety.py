"""Regression tests for the post-login ``next=`` same-origin gate (#136462).

``is_safe_next_path`` is the single gate under every ``next=`` entry point — the auth
routes' ``_validate_post_login_target`` (after ``unquote()``) and the gate middleware's
``_safe_next_target``. A string-prefix-only check let ``/\\evil.example`` through: browsers
normalise ``\\`` to ``/`` in special-scheme URLs (WHATWG URL), so
``window.location.assign("/\\evil.example")`` resolves to ``https://evil.example/`` — an
open redirect off the dashboard origin immediately after authentication. ``unquote()``
re-creates the same state from the URL-safe form ``?next=/%5Cevil.example``.

The matrix mirrors the bypass classes from the report: raw ``//``, backslash forms
(literal, %-encoded, repeated), control characters WHATWG parsing strips (tab/newline),
plus the previously-rejected and legitimately-accepted controls. A space never moves the
authority (the parser percent-encodes it in the path), so decoded ``%20`` deep links survive.
"""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.request_utils import is_safe_next_path
from hermes_cli.dashboard_auth.routes import (
    _reset_password_rate_limit,
    _validate_post_login_target,
)
from tests.hermes_cli.test_dashboard_auth_password_login import PasswordProvider

# Values that must be rejected: each resolves off-origin (or into the auth/API flow)
# once a browser parses it, so none may survive the gate as a post-login target.
UNSAFE_NEXT_PATHS = [
    "//evil.example",  # protocol-relative
    "/\\evil.example",  # backslash form: WHATWG collapses \ to /
    "\\\\evil.example",  # double backslash
    "/\t/x",  # tab is stripped before URL parsing -> //x
    "/\n/x",  # newline is stripped before URL parsing -> //x
    "/\r/x",  # carriage return is stripped before URL parsing -> //x
    "/\t\\x",  # tab + backslash
    "https://evil.example",  # absolute URL
    "evil.example",  # not root-relative
    "/sessions\\x",  # backslash anywhere, not only at the prefix
    "/login",
    "/auth/login",
    "/api",
    "/api/status",
]


class TestIsSafeNextPath:
    @pytest.mark.parametrize("path", UNSAFE_NEXT_PATHS)
    def test_rejects_off_origin_and_auth_flow_forms(self, path: str):
        assert is_safe_next_path(path) is False

    @pytest.mark.parametrize(
        "path", ["/", "/sessions", "/sessions?tab=all", "/sessions?q=a b", "/ /x"]
    )
    def test_accepts_same_origin_paths(self, path: str):
        assert is_safe_next_path(path) is True


class TestValidatePostLoginTarget:
    """The routes gate decodes ``unquote()`` BEFORE validating, so %-encoded variants of the
    bypass forms must collapse back into (and be rejected as) their decoded selves."""

    @pytest.mark.parametrize(
        "raw",
        [
            "/%5Cevil.example",  # -> /\evil.example
            "/%5c/x",  # lowercase hex
            "%2F%5Cevil.example",  # encoded slash + backslash
            "%2f%2fevil.example",  # encoded protocol-relative
            "%5C%5Cevil.example",  # encoded double backslash
            "/%09/x",  # encoded tab
            "/%0A/x",  # encoded newline
            "/%0d/x",  # encoded carriage return
            "/%5C%09evil.example",  # encoded backslash + tab
        ],
    )
    def test_encoded_bypass_forms_are_rejected(self, raw: str):
        assert _validate_post_login_target(raw) == ""

    def test_legitimate_encoded_path_survives_decoding(self):
        # Encoded query punctuation round-trips; an encoded slash does not — it decodes
        # into the (rejected) protocol-relative prefix, which is the fail-closed intent.
        assert (
            _validate_post_login_target("/sessions%3Ftab%3Dall") == "/sessions?tab=all"
        )
        assert _validate_post_login_target("/%2Fsessions") == ""

    def test_empty_and_blank_fall_back(self):
        assert _validate_post_login_target("") == ""


class TestExternalizedTargetIsValidatedComposed:
    """The prefix is joined AFTER ``next`` is validated; the browser gets the composed string,
    so a prefix that would move the authority must never reach it, whatever its source."""

    @pytest.mark.parametrize("prefix", ["/\\evil.example", "/\tx", "\\evil.example"])
    def test_authority_moving_prefix_is_dropped(self, prefix: str):
        from hermes_cli.dashboard_auth.routes import _externalize_post_login_target

        assert _externalize_post_login_target("/sessions", prefix) == "/sessions"

    def test_ordinary_prefix_still_applies(self):
        from hermes_cli.dashboard_auth.routes import _externalize_post_login_target

        assert _externalize_post_login_target("/sessions", "/hermes") == "/hermes/sessions"


class TestPasswordLoginNextTainted:
    @pytest.fixture
    def authed_client(self):
        """The gated dashboard app with the in-test password provider (admin/hunter2),
        mirroring the password-login flow's E2E harness."""
        clear_providers()
        register_provider(PasswordProvider())
        _reset_password_rate_limit()
        prev_host = getattr(web_server.app.state, "bound_host", None)
        prev_port = getattr(web_server.app.state, "bound_port", None)
        prev_required = getattr(web_server.app.state, "auth_required", None)
        web_server.app.state.bound_host = "fly-app.fly.dev"
        web_server.app.state.bound_port = 443
        web_server.app.state.auth_required = True
        yield TestClient(web_server.app, base_url="https://fly-app.fly.dev")
        clear_providers()
        _reset_password_rate_limit()
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required

    def test_backslash_escaped_next_falls_back_to_root(self, authed_client: TestClient):
        """The reported end-to-end repro: POST the URL-safe form of ``/\\evil.example`` and the
        JSON body must NOT hand it to ``window.location.assign``."""
        resp = authed_client.post(
            "/auth/password-login",
            json={
                "provider": "testpw",
                "username": "admin",
                "password": "hunter2",
                "next": "/%5Cevil.example",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "next": "/"}

    def test_malformed_forwarded_prefix_cannot_override_safe_post_login_next(
        self, authed_client: TestClient
    ):
        """Reject a proxy mount with a backslash before composing the browser landing URL."""
        response = authed_client.post(
            "/auth/password-login",
            headers={"x-forwarded-prefix": "/\\evil.example"},
            json={
                "provider": "testpw",
                "username": "admin",
                "password": "hunter2",
                "next": "/sessions",
            },
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True, "next": "/sessions"}

    def test_backslash_forwarded_prefix_never_reaches_the_login_form(
        self, authed_client: TestClient
    ):
        """The password form's script POSTs to ``data-prefix + '/auth/password-login'``, so a
        backslash prefix must be dropped at ``normalise_prefix``, not only at the landing join."""
        page = authed_client.get("/login", headers={"x-forwarded-prefix": "/\\evil.example"})
        assert page.status_code == 200
        assert "evil.example" not in page.text
