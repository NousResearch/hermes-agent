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

    def __init__(
        self,
        *,
        secret: str = "good-secret",
        scopes=("drain",),
        name: str = "tok",
    ):
        self.name = name
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
    def __init__(self, host="1.2.3.4"):
        self.host = host


class _FakeRequest:
    """Minimal Request stand-in for the seam (no real Starlette needed)."""

    def __init__(
        self,
        path="/api/gateway/drain",
        headers=None,
        *,
        method="GET",
        bound_host=None,
        client_host="1.2.3.4",
    ):
        self.url = _FakeURL(path)
        self.headers = headers or {}
        self.method = method
        self.client = _FakeClient(client_host)

        class _State:
            pass

        self.state = _State()

        class _AppState:
            pass

        self.app = type("_App", (), {})()
        self.app.state = _AppState()
        self.app.state.bound_host = bound_host


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# ABC + registry
# --------------------------------------------------------------------------


def test_oauth_provider_defaults_supports_token_false():
    assert _OAuthOnly().supports_token is False




class _NonInteractiveProvider(_TokenProvider):
    """A token-only credential with no interactive session."""

    name = "svc-cred"
    display_name = "Service Credential"
    supports_session = False


# --------------------------------------------------------------------------
# Bearer extraction
# --------------------------------------------------------------------------




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


def test_authenticate_token_stacks_first_match_wins():
    register_provider(_TokenProvider(secret="aaa"))
    second = _TokenProvider(secret="bbb")
    second.name = "tok2"
    register_provider(second)
    req = _FakeRequest(headers={"authorization": "Bearer bbb"})
    principal, _ = token_auth.authenticate_token(req)
    assert principal is not None and principal.provider == "tok2"


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


def test_scoped_auth_continues_after_same_secret_principal_without_scope():
    register_provider(
        _TokenProvider(secret="same", scopes=("drain",), name="first")
    )
    register_provider(
        _TokenProvider(secret="same", scopes=("kanban.read",), name="second")
    )
    token_auth.register_token_route(
        "/read", required_scope="kanban.read"
    )
    req = _FakeRequest(path="/read", headers={"authorization": "Bearer same"})

    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))

    assert resp.status_code == 200
    assert req.state.token_principal.provider == "second"


# --------------------------------------------------------------------------
# Middleware seam (route-agnostic)
# --------------------------------------------------------------------------


async def _call_next_ok(request):
    from fastapi.responses import JSONResponse

    return JSONResponse({"ok": True}, status_code=200)






def test_seam_rejects_wrong_token_401():
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/gateway/drain")
    req = _FakeRequest(
        path="/api/gateway/drain", headers={"authorization": "Bearer bad"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 401


def test_legacy_route_registration_remains_method_agnostic_and_unscoped():
    register_provider(_TokenProvider(secret="good", scopes=("anything",)))
    token_auth.register_token_route("/legacy")
    assert token_auth.is_token_route("/legacy")
    assert token_auth.is_token_route("/legacy", "GET")
    assert token_auth.is_token_route("/legacy", "POST")

    req = _FakeRequest(
        path="/legacy", method="PATCH", headers={"authorization": "Bearer good"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert req.state.token_authenticated is True


def test_route_method_matching_falls_through_for_other_methods():
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/read", methods=("get",))
    assert token_auth.is_token_route("/read", "GET")
    assert not token_auth.is_token_route("/read", "POST")

    req = _FakeRequest(
        path="/read", method="POST", headers={"authorization": "Bearer good"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert not hasattr(req.state, "token_authenticated")


def test_required_scope_allows_matching_principal():
    register_provider(_TokenProvider(secret="good", scopes=("kanban.read",)))
    token_auth.register_token_route(
        "/read", methods=("GET",), required_scope="kanban.read"
    )
    req = _FakeRequest(
        path="/read", headers={"authorization": "Bearer good"}, bound_host="127.0.0.1"
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert req.state.token_authenticated is True


def test_missing_required_scope_returns_403_without_token_authentication(monkeypatch):
    register_provider(_TokenProvider(secret="good", scopes=("drain",)))
    token_auth.register_token_route(
        "/read", methods=("GET",), required_scope="kanban.read"
    )
    events = []
    monkeypatch.setattr(
        token_auth,
        "audit_log",
        lambda event, **fields: events.append((event, fields)),
    )
    req = _FakeRequest(
        path="/read", headers={"authorization": "Bearer good"}, bound_host="127.0.0.1"
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 403
    assert getattr(req.state, "token_authenticated", False) is False
    assert events[0][0].value == "token_auth_failure"
    assert events[0][1]["reason"] == "missing_scope"


def test_loopback_only_route_requires_known_loopback_bind():
    register_provider(_TokenProvider(secret="good", scopes=("kanban.read",)))
    token_auth.register_token_route(
        "/read", methods=("GET",), required_scope="kanban.read", loopback_only=True
    )
    for bound_host in (None, "10.0.0.7"):
        req = _FakeRequest(
            path="/read",
            headers={"authorization": "Bearer good"},
            bound_host=bound_host,
            client_host="127.0.0.1",
        )
        resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
        assert resp.status_code == 200
        assert not hasattr(req.state, "token_authenticated")

    req = _FakeRequest(
        path="/read",
        headers={"authorization": "Bearer good"},
        bound_host="localhost",
        client_host="::1",
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert req.state.token_authenticated is True


@pytest.mark.parametrize("client_host", [None, "10.0.0.7", "not-an-ip"])
def test_loopback_only_route_requires_known_loopback_peer_and_ignores_forwarded_for(
    client_host,
):
    register_provider(_TokenProvider(secret="good", scopes=("kanban.read",)))
    token_auth.register_token_route(
        "/read", required_scope="kanban.read", loopback_only=True
    )
    req = _FakeRequest(
        path="/read",
        headers={
            "authorization": "Bearer good",
            "x-forwarded-for": "127.0.0.1",
        },
        bound_host="127.0.0.1",
        client_host=client_host,
    )

    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))

    assert resp.status_code == 200
    assert not hasattr(req.state, "token_authenticated")


def test_token_route_can_fall_through_to_native_session_gate_without_bearer():
    register_provider(_TokenProvider(secret="good", scopes=("kanban.read",)))
    token_auth.register_token_route(
        "/read",
        required_scope="kanban.read",
        allow_session_fallback=True,
    )
    req = _FakeRequest(path="/read", headers={})

    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))

    assert resp.status_code == 200
    assert not hasattr(req.state, "token_authenticated")


def test_token_route_session_fallback_does_not_accept_invalid_bearer():
    register_provider(_TokenProvider(secret="good", scopes=("kanban.read",)))
    token_auth.register_token_route(
        "/read",
        required_scope="kanban.read",
        allow_session_fallback=True,
    )
    req = _FakeRequest(
        path="/read", headers={"authorization": "Bearer wrong"}
    )

    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))

    assert resp.status_code == 401


def test_duplicate_identical_route_is_idempotent_but_conflict_fails_closed():
    token_auth.register_token_route(
        "/read", methods=("get",), required_scope="kanban.read", loopback_only=True
    )
    token_auth.register_token_route(
        "/read", methods=("GET",), required_scope="kanban.read", loopback_only=True
    )
    with pytest.raises(ValueError, match="conflicting token route registration"):
        token_auth.register_token_route(
            "/read", methods=("POST",), required_scope="kanban.read", loopback_only=True
        )


def test_route_clearing_removes_method_rules():
    token_auth.register_token_route("/read", methods=("GET",))
    assert token_auth.is_token_route("/read")
    token_auth.clear_token_routes()
    assert not token_auth.is_token_route("/read")
