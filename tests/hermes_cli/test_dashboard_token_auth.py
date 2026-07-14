"""Contract tests for the generic non-interactive (bearer-token) auth seam.

Covers Task 2.0a: the reusable token-auth capability in the dashboard auth
framework — NOT the drain plugin (that's 2.0b/2.1). Asserts the ABC capability
flag, the registry filter, bearer extraction, provider stacking (verify_token),
and the route-agnostic middleware seam's fail-closed / 503 / pass-through
behaviour.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    LoginStart,
    Session,
    TokenPrincipal,
    clear_providers,
    list_providers,
    list_session_providers,
    list_token_providers,
    register_provider,
)
from hermes_cli.dashboard_auth.base import ProviderError
from hermes_cli.dashboard_auth import token_auth


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class _OAuthOnly(DashboardAuthProvider):
    """A pure interactive provider — never token-authable."""

    name = "oauth-only"
    display_name = "OAuth Only"

    def start_login(self, *, redirect_uri):
        return LoginStart(redirect_url="x", cookie_payload={})

    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        return Session("u", "e", "n", "o", self.name, 0, "a", "r")

    def verify_session(self, *, access_token):
        return None

    def refresh_session(self, *, refresh_token):
        return Session("u", "e", "n", "o", self.name, 0, "a", "r")

    def revoke_session(self, *, refresh_token):
        return None


class _TokenProvider(_OAuthOnly):
    """A token provider that accepts exactly one secret."""

    name = "tok"
    display_name = "Token Provider"
    supports_token = True

    def __init__(self, *, secret: str = "good-secret", scopes=("drain",)):
        self._secret = secret
        self._scopes = tuple(scopes)

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        if token == self._secret:
            return TokenPrincipal(
                principal=self.name, provider=self.name, scopes=self._scopes
            )
        return None


class _UnreachableTokenProvider(_OAuthOnly):
    name = "tok-down"
    display_name = "Unreachable Token Provider"
    supports_token = True

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        raise ProviderError("backing store down")


class _BuggyTokenProvider(_OAuthOnly):
    name = "tok-buggy"
    display_name = "Buggy Token Provider"
    supports_token = True

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        raise RuntimeError("kaboom")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_state():
    clear_providers()
    token_auth.clear_token_routes()
    yield
    clear_providers()
    token_auth.clear_token_routes()


class _FakeURL:
    def __init__(self, path):
        self.path = path


class _FakeClient:
    host = "1.2.3.4"


class _FakeRequest:
    """Minimal Request stand-in for the seam (no real Starlette needed)."""

    def __init__(self, path="/api/gateway/drain", headers=None):
        self.url = _FakeURL(path)
        self.headers = headers or {}
        self.client = _FakeClient()

        class _State:
            pass

        self.state = _State()


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# ABC + registry
# --------------------------------------------------------------------------


def test_oauth_provider_defaults_supports_token_false():
    assert _OAuthOnly().supports_token is False


def test_oauth_provider_verify_token_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        _OAuthOnly().verify_token(token="x")


def test_list_token_providers_filters_to_supports_token():
    register_provider(_OAuthOnly())
    register_provider(_TokenProvider())
    names = [p.name for p in list_token_providers()]
    assert names == ["tok"]


def test_list_token_providers_empty_when_none_registered():
    register_provider(_OAuthOnly())
    assert list_token_providers() == []


class _NonInteractiveProvider(_TokenProvider):
    """A token-only credential with no interactive session."""

    name = "svc-cred"
    display_name = "Service Credential"
    supports_session = False


def test_oauth_provider_defaults_supports_session_true():
    # Interactive providers participate in cookie sessions by default.
    assert _OAuthOnly().supports_session is True


def test_list_session_providers_excludes_non_interactive():
    # Token-only providers stay out of the interactive set. Mirror of
    # list_token_providers.
    register_provider(_OAuthOnly())
    register_provider(_NonInteractiveProvider())
    assert {p.name for p in list_providers()} == {"oauth-only", "svc-cred"}
    assert [p.name for p in list_session_providers()] == ["oauth-only"]


# --------------------------------------------------------------------------
# Bearer extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header,expected",
    [
        ("Bearer abc123", "abc123"),
        ("bearer abc123", "abc123"),
        ("BEARER abc123", "abc123"),
        ("Bearer   spaced  ", "spaced"),
        ("Basic abc123", ""),
        ("abc123", ""),
        ("", ""),
    ],
)
def test_extract_bearer_token(header, expected):
    req = _FakeRequest(headers={"authorization": header} if header else {})
    assert token_auth.extract_bearer_token(req) == expected


# --------------------------------------------------------------------------
# authenticate_token (provider stacking)
# --------------------------------------------------------------------------


def test_authenticate_token_accepts_valid():
    register_provider(_TokenProvider(secret="good-secret"))
    req = _FakeRequest(headers={"authorization": "Bearer good-secret"})
    principal, unreachable = token_auth.authenticate_token(req)
    assert unreachable is None
    assert principal is not None
    assert principal.provider == "tok"
    assert principal.scopes == ("drain",)


def test_authenticate_token_rejects_wrong_secret():
    register_provider(_TokenProvider(secret="good-secret"))
    req = _FakeRequest(headers={"authorization": "Bearer wrong"})
    principal, unreachable = token_auth.authenticate_token(req)
    assert principal is None
    assert unreachable is None


def test_authenticate_token_no_token_returns_none():
    register_provider(_TokenProvider())
    req = _FakeRequest(headers={})
    principal, unreachable = token_auth.authenticate_token(req)
    assert principal is None and unreachable is None


def test_authenticate_token_stacks_first_match_wins():
    register_provider(_TokenProvider(secret="aaa"))
    second = _TokenProvider(secret="bbb")
    second.name = "tok2"
    register_provider(second)
    req = _FakeRequest(headers={"authorization": "Bearer bbb"})
    principal, _ = token_auth.authenticate_token(req)
    assert principal is not None and principal.provider == "tok2"


def test_authenticate_token_unreachable_remembered():
    register_provider(_UnreachableTokenProvider())
    req = _FakeRequest(headers={"authorization": "Bearer anything"})
    principal, unreachable = token_auth.authenticate_token(req)
    assert principal is None
    assert unreachable == "tok-down"


def test_authenticate_token_unreachable_then_valid_provider_wins():
    register_provider(_UnreachableTokenProvider())
    register_provider(_TokenProvider(secret="good"))
    req = _FakeRequest(headers={"authorization": "Bearer good"})
    principal, unreachable = token_auth.authenticate_token(req)
    # A later provider accepting the token beats the earlier outage.
    assert principal is not None and principal.provider == "tok"
    assert unreachable is None


def test_authenticate_token_buggy_provider_does_not_crash():
    register_provider(_BuggyTokenProvider())
    register_provider(_TokenProvider(secret="good"))
    req = _FakeRequest(headers={"authorization": "Bearer good"})
    principal, unreachable = token_auth.authenticate_token(req)
    assert principal is not None and principal.provider == "tok"


# --------------------------------------------------------------------------
# Middleware seam (route-agnostic)
# --------------------------------------------------------------------------


async def _call_next_ok(request):
    from fastapi.responses import JSONResponse

    return JSONResponse({"ok": True}, status_code=200)


def test_seam_passthrough_for_unregistered_route():
    register_provider(_TokenProvider())
    req = _FakeRequest(path="/api/something-else")
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert getattr(req.state, "token_authenticated", False) is False


def test_seam_accepts_valid_token_on_registered_route():
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/gateway/drain")
    req = _FakeRequest(
        path="/api/gateway/drain",
        headers={"authorization": "Bearer good"},
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert req.state.token_authenticated is True
    assert req.state.token_principal.provider == "tok"


def test_seam_rejects_missing_token_401():
    register_provider(_TokenProvider())
    token_auth.register_token_route("/api/gateway/drain")
    req = _FakeRequest(path="/api/gateway/drain", headers={})
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 401


def test_seam_rejects_wrong_token_401():
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/gateway/drain")
    req = _FakeRequest(
        path="/api/gateway/drain", headers={"authorization": "Bearer bad"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 401


def test_seam_fails_closed_when_no_token_provider():
    # Route registered but NO supports_token provider → 401, never open.
    register_provider(_OAuthOnly())
    token_auth.register_token_route("/api/gateway/drain")
    req = _FakeRequest(
        path="/api/gateway/drain", headers={"authorization": "Bearer anything"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 401


def test_seam_503_on_provider_unreachable():
    register_provider(_UnreachableTokenProvider())
    token_auth.register_token_route("/api/gateway/drain")
    req = _FakeRequest(
        path="/api/gateway/drain", headers={"authorization": "Bearer x"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 503


# --------------------------------------------------------------------------
# Prefix registration
# --------------------------------------------------------------------------


def test_prefix_route_matches_parameterised_paths():
    register_provider(_TokenProvider(secret="good", scopes=("kanban",)))
    token_auth.register_token_route_prefix("/api/plugins/kanban/")
    for path in (
        "/api/plugins/kanban/tasks",
        "/api/plugins/kanban/tasks/t_abc123/complete",
    ):
        assert token_auth.is_token_route(path)
        req = _FakeRequest(path=path, headers={"authorization": "Bearer good"})
        resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
        assert resp.status_code == 200
        assert req.state.token_authenticated is True


def test_prefix_route_does_not_match_outside_prefix():
    token_auth.register_token_route_prefix("/api/plugins/kanban/")
    # Sibling plugin and the bare prefix root are untouched pass-throughs.
    for path in ("/api/plugins/other/tasks", "/api/plugins/kanban"):
        assert not token_auth.is_token_route(path)
        req = _FakeRequest(path=path)
        resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
        assert resp.status_code == 200
        assert getattr(req.state, "token_authenticated", False) is False


def test_prefix_route_excluded_subtree_stays_on_session_gate():
    register_provider(_TokenProvider(secret="good", scopes=("kanban",)))
    token_auth.register_token_route_prefix(
        "/api/plugins/kanban/",
        exclude=("/api/plugins/kanban/dashboard",),
    )
    # The interactive dashboard subtree is NOT owned by the seam: no token
    # demanded, request passes through to the cookie/session gates untouched.
    for path in (
        "/api/plugins/kanban/dashboard",
        "/api/plugins/kanban/dashboard/tasks/t_abc123",
    ):
        assert not token_auth.is_token_route(path)
        req = _FakeRequest(path=path)
        resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
        assert resp.status_code == 200
        assert getattr(req.state, "token_authenticated", False) is False
    # ...while the external surface next to it is token-only.
    assert token_auth.is_token_route("/api/plugins/kanban/tasks")
    req = _FakeRequest(path="/api/plugins/kanban/tasks", headers={})
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 401


def test_clear_token_routes_drops_prefixes():
    token_auth.register_token_route_prefix("/api/plugins/kanban/")
    token_auth.clear_token_routes()
    assert not token_auth.is_token_route("/api/plugins/kanban/tasks")


# --------------------------------------------------------------------------
# Scope enforcement
# --------------------------------------------------------------------------


def test_scoped_route_accepts_principal_with_scope():
    register_provider(_TokenProvider(secret="good", scopes=("drain",)))
    token_auth.register_token_route("/api/gateway/drain", scope="drain")
    req = _FakeRequest(
        path="/api/gateway/drain", headers={"authorization": "Bearer good"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert req.state.token_authenticated is True


def test_scoped_route_rejects_principal_without_scope_403():
    # A valid credential for ANOTHER surface must not open this one: the
    # kanban-scoped secret authenticates but lacks the drain capability.
    register_provider(_TokenProvider(secret="good", scopes=("kanban",)))
    token_auth.register_token_route("/api/gateway/drain", scope="drain")
    req = _FakeRequest(
        path="/api/gateway/drain", headers={"authorization": "Bearer good"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 403
    assert getattr(req.state, "token_authenticated", False) is False


def test_scoped_prefix_rejects_foreign_credential_403():
    register_provider(_TokenProvider(secret="drain-secret", scopes=("drain",)))
    token_auth.register_token_route_prefix("/api/plugins/kanban/", scope="kanban")
    req = _FakeRequest(
        path="/api/plugins/kanban/tasks",
        headers={"authorization": "Bearer drain-secret"},
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 403


def test_unscoped_route_accepts_any_verified_principal():
    # Backward compatibility: no scope requirement → any verified principal
    # passes, exactly the pre-scope behaviour.
    register_provider(_TokenProvider(secret="good", scopes=()))
    token_auth.register_token_route("/api/gateway/drain")
    req = _FakeRequest(
        path="/api/gateway/drain", headers={"authorization": "Bearer good"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
