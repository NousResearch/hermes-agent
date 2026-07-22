"""Contract tests for the generic non-interactive (bearer-token) auth seam.

Covers Task 2.0a: the reusable token-auth capability in the dashboard auth
framework — NOT the drain plugin (that's 2.0b/2.1). Asserts the ABC capability
flag, the registry filter, bearer extraction, provider stacking (verify_token),
and the route-agnostic middleware seam's fail-closed / 503 / pass-through
behaviour.
"""
from __future__ import annotations

import asyncio
import re
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

    def __init__(self, path="/api/gateway/drain", headers=None, method="GET"):
        self.url = _FakeURL(path)
        self.headers = headers or {}
        self.client = _FakeClient()
        self.method = method

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
# register_token_route(match=...) — additive prefix support
# --------------------------------------------------------------------------


def test_register_token_route_default_is_exact():
    token_auth.register_token_route("/api/sessions")
    assert token_auth.is_token_route("/api/sessions") is True
    assert token_auth.is_token_route("/api/sessions/abc123") is False


def test_register_token_route_rejects_invalid_match():
    with pytest.raises(ValueError):
        token_auth.register_token_route("/api/sessions", match="glob")


def test_register_token_route_prefix_matches_dynamic_segment():
    token_auth.register_token_route("/api/sessions/", match="prefix")
    assert token_auth.is_token_route("/api/sessions/abc123") is True
    assert token_auth.is_token_route("/api/sessions/abc123/messages") is True
    assert token_auth.is_token_route("/api/sessions") is False


def test_register_token_route_exact_takes_precedence_over_prefix():
    # A route that happens to fall under a registered prefix, but was ALSO
    # exactly registered elsewhere, must still match — precedence is a
    # property of is_token_route, not registration order.
    token_auth.register_token_route("/api/sessions/", match="prefix")
    token_auth.register_token_route("/api/sessions/search")
    assert token_auth.is_token_route("/api/sessions/search") is True
    assert token_auth.is_token_route("/api/sessions/anything-else") is True


def test_register_token_route_prefix_does_not_widen_existing_exact_routes():
    # Registering an unrelated prefix must not affect a pre-existing exact
    # registration's behaviour — additive only, per drain's own contract.
    token_auth.register_token_route("/api/gateway/drain")
    token_auth.register_token_route("/api/sessions/", match="prefix")
    assert token_auth.is_token_route("/api/gateway/drain") is True
    assert token_auth.is_token_route("/api/gateway/drain/extra") is False


def test_clear_token_routes_clears_both_exact_and_prefix():
    token_auth.register_token_route("/api/gateway/drain")
    token_auth.register_token_route("/api/sessions/", match="prefix")
    token_auth.clear_token_routes()
    assert token_auth.is_token_route("/api/gateway/drain") is False
    assert token_auth.is_token_route("/api/sessions/abc123") is False


def test_seam_accepts_valid_token_on_prefix_registered_route():
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/sessions/", match="prefix")
    req = _FakeRequest(
        path="/api/sessions/abc123/messages",
        headers={"authorization": "Bearer good"},
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert req.state.token_authenticated is True


# --------------------------------------------------------------------------
# register_token_route(match="regex") — additive, pattern-constrained dynamic
# segment (for a sibling-literal-route collision "prefix" can't avoid, e.g.
# /api/sessions/{session_id} vs. /api/sessions/search)
# --------------------------------------------------------------------------

_SESSION_ID_PATTERN = r"/api/sessions/\d{8}_\d{6}_[0-9a-f]{6,8}"
_SESSION_MESSAGES_PATTERN = r"/api/sessions/\d{8}_\d{6}_[0-9a-f]{6,8}/messages"


def test_register_token_route_regex_matches_constrained_shape():
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3d4") is True
    # 6-hex branch/thread session ids (gateway/slash_commands.py's shorter
    # uuid slice) must also match — not just the 8-hex "new session" shape.
    assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3") is True


def test_register_token_route_regex_rejects_sibling_literal_routes():
    # The exact collision this mode exists to prevent: a blanket prefix would
    # sweep these in, a shape-constrained regex must not.
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    for sibling in ("search", "stats", "bulk-delete", "prune"):
        assert token_auth.is_token_route(f"/api/sessions/{sibling}") is False
    assert token_auth.is_token_route("/api/sessions/empty/count") is False


def test_register_token_route_regex_requires_full_match_not_substring():
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    # fullmatch semantics: trailing/leading junk must not sneak a match in.
    assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3d4/messages") is False
    assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3d4x") is False
    assert token_auth.is_token_route("x/api/sessions/20260702_143022_a1b2c3d4") is False


def test_register_token_route_regex_separate_pattern_for_messages_suffix():
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    token_auth.register_token_route(_SESSION_MESSAGES_PATTERN, match="regex")
    assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3d4/messages") is True
    assert token_auth.is_token_route("/api/sessions/search/messages") is False


def test_register_token_route_rejects_invalid_regex():
    with pytest.raises(re.error):
        token_auth.register_token_route("/api/sessions/[", match="regex")


def test_register_token_route_regex_is_idempotent():
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    assert len(token_auth._token_route_regexes) == 1


def test_register_token_route_exact_takes_precedence_over_regex():
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    token_auth.register_token_route("/api/sessions/20260702_143022_a1b2c3d4")
    assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3d4") is True


def test_register_token_route_regex_does_not_widen_existing_exact_or_prefix_routes():
    token_auth.register_token_route("/api/gateway/drain")
    token_auth.register_token_route("/api/sessions/", match="prefix")
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    assert token_auth.is_token_route("/api/gateway/drain") is True
    assert token_auth.is_token_route("/api/sessions/abc123") is True  # unaffected prefix behaviour


def test_clear_token_routes_clears_regex_too():
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    token_auth.clear_token_routes()
    assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3d4") is False


def test_seam_accepts_valid_token_on_regex_registered_route():
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    req = _FakeRequest(
        path="/api/sessions/20260702_143022_a1b2c3d4",
        headers={"authorization": "Bearer good"},
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert req.state.token_authenticated is True


def test_seam_passthrough_for_sibling_literal_route_not_swept_by_regex():
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex")
    req = _FakeRequest(path="/api/sessions/bulk-delete", headers={})
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    # Not a registered token route at all -> pass through to whatever the
    # existing cookie/session gate does (here, the fake call_next -> 200).
    assert resp.status_code == 200
    assert not hasattr(req.state, "token_authenticated")


# --------------------------------------------------------------------------
# register_token_route(required=...) — optional (additive, non-exclusive)
# token auth for routes also reachable via the existing cookie/session gate.
#
# NOTE: the load-bearing claim here — "no Authorization header on a
# required=False route actually falls through to and is decided by the
# downstream cookie/session gate, inside the real Starlette middleware
# stack" — is NOT verified by this file's _FakeRequest-based unit tests
# below; a fake call_next always "succeeds" so it can't distinguish
# fallthrough from every other path also returning 200. That claim is
# verified separately, against the real app and real TestClient, in
# tests/hermes_cli/test_web_server.py
# (TestTokenAuthOptionalFallthroughIntegration).
# --------------------------------------------------------------------------


def test_register_token_route_default_required_true():
    token_auth.register_token_route("/api/gateway/drain")
    assert token_auth.is_token_route_required("/api/gateway/drain") is True


def test_register_token_route_required_false_exact():
    token_auth.register_token_route("/api/status", required=False)
    assert token_auth.is_token_route("/api/status") is True
    assert token_auth.is_token_route_required("/api/status") is False


def test_register_token_route_required_false_prefix():
    token_auth.register_token_route("/api/sessions/", match="prefix", required=False)
    assert token_auth.is_token_route_required("/api/sessions/abc123") is False


def test_register_token_route_required_false_regex():
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex", required=False)
    assert token_auth.is_token_route_required(
        "/api/sessions/20260702_143022_a1b2c3d4"
    ) is False


def test_register_token_route_required_merge_stricter_wins_exact():
    # Same exact path registered twice with different required values: the
    # stricter (True) value must win regardless of registration order, so a
    # fail-closed registration can never be silently loosened later.
    token_auth.register_token_route("/api/status", required=False)
    token_auth.register_token_route("/api/status", required=True)
    assert token_auth.is_token_route_required("/api/status") is True


def test_register_token_route_required_merge_stricter_wins_exact_reverse_order():
    token_auth.register_token_route("/api/status", required=True)
    token_auth.register_token_route("/api/status", required=False)
    assert token_auth.is_token_route_required("/api/status") is True


def test_register_token_route_required_merge_stricter_wins_across_overlapping_prefixes():
    # Two different prefix registrations, one strict, one not, both matching
    # the same path — the stricter one must win.
    token_auth.register_token_route("/api/sessions/", match="prefix", required=False)
    token_auth.register_token_route("/api/sessions/admin", match="prefix", required=True)
    assert token_auth.is_token_route_required("/api/sessions/admin/x") is True
    assert token_auth.is_token_route_required("/api/sessions/other") is False


def test_is_token_route_required_exact_wins_over_looser_regex():
    # An exact registration's required flag is authoritative even if a
    # looser (required=False) regex also matches the same path.
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex", required=False)
    token_auth.register_token_route(
        "/api/sessions/20260702_143022_a1b2c3d4", required=True
    )
    assert token_auth.is_token_route_required(
        "/api/sessions/20260702_143022_a1b2c3d4"
    ) is True


def test_seam_no_token_falls_through_on_required_false_route():
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/status", required=False)
    req = _FakeRequest(path="/api/status", headers={})
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    # Fell through untouched — the seam never ran a decision, so it must not
    # have attached any auth state at all.
    assert not hasattr(req.state, "token_authenticated")
    assert not hasattr(req.state, "token_principal")


def test_seam_valid_token_still_decided_by_seam_on_required_false_route():
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/status", required=False)
    req = _FakeRequest(
        path="/api/status", headers={"authorization": "Bearer good"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert req.state.token_authenticated is True
    assert req.state.token_principal.provider == "tok"


def test_seam_invalid_token_still_401s_on_required_false_route():
    # A PRESENTED (even if wrong) token on a required=False route is still
    # fully decided by this seam alone — only a genuinely absent
    # Authorization header falls through.
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/status", required=False)
    req = _FakeRequest(
        path="/api/status", headers={"authorization": "Bearer wrong"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 401


def test_seam_unreachable_provider_still_503s_on_required_false_route():
    register_provider(_UnreachableTokenProvider())
    token_auth.register_token_route("/api/status", required=False)
    req = _FakeRequest(
        path="/api/status", headers={"authorization": "Bearer x"}
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 503


def test_seam_no_token_still_401s_on_required_true_route():
    # required=True (the default, drain's contract) is completely unchanged:
    # no token still means an immediate 401, never a fallthrough.
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/gateway/drain")
    req = _FakeRequest(path="/api/gateway/drain", headers={})
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 401


def test_clear_token_routes_clears_required_flags():
    token_auth.register_token_route("/api/status", required=False)
    token_auth.clear_token_routes()
    token_auth.register_token_route("/api/status")
    # Post-clear re-registration must not inherit the old required=False —
    # confirms clear_token_routes() actually drops the flag, not just the key.
    assert token_auth.is_token_route_required("/api/status") is True


# --------------------------------------------------------------------------
# register_token_route(methods=...) — per-method restriction. The security
# boundary that keeps a read-only registration from authenticating a mutating
# verb sharing the same path (GET vs POST /api/skills, GET vs DELETE
# /api/sessions/{id}, …). See token_auth.py's registry comment.
# --------------------------------------------------------------------------


def test_methods_default_none_allows_every_method():
    token_auth.register_token_route("/api/gateway/drain")  # methods=None
    for m in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
        assert token_auth.is_token_route_method_allowed("/api/gateway/drain", m) is True


def test_methods_restricted_exact_allows_only_listed():
    token_auth.register_token_route("/api/skills", required=False, methods=("GET", "HEAD"))
    assert token_auth.is_token_route_method_allowed("/api/skills", "GET") is True
    assert token_auth.is_token_route_method_allowed("/api/skills", "HEAD") is True
    assert token_auth.is_token_route_method_allowed("/api/skills", "POST") is False
    assert token_auth.is_token_route_method_allowed("/api/skills", "PUT") is False
    assert token_auth.is_token_route_method_allowed("/api/skills", "DELETE") is False


def test_methods_are_case_insensitive_on_registration_and_query():
    token_auth.register_token_route("/api/skills", required=False, methods=("get",))
    assert token_auth.is_token_route_method_allowed("/api/skills", "GET") is True
    assert token_auth.is_token_route_method_allowed("/api/skills", "get") is True
    assert token_auth.is_token_route_method_allowed("/api/skills", "POST") is False


def test_methods_restricted_regex_allows_only_listed():
    token_auth.register_token_route(_SESSION_ID_PATTERN, match="regex", methods=("GET", "HEAD"))
    sid = "/api/sessions/20260702_100000_aaaaaaaa"
    assert token_auth.is_token_route_method_allowed(sid, "GET") is True
    assert token_auth.is_token_route_method_allowed(sid, "DELETE") is False
    assert token_auth.is_token_route_method_allowed(sid, "PATCH") is False


def test_methods_union_across_repeat_registrations():
    # A method is authable if ANY registration of the path permits it.
    token_auth.register_token_route("/api/thing", methods=("GET",))
    token_auth.register_token_route("/api/thing", methods=("POST",))
    assert token_auth.is_token_route_method_allowed("/api/thing", "GET") is True
    assert token_auth.is_token_route_method_allowed("/api/thing", "POST") is True
    assert token_auth.is_token_route_method_allowed("/api/thing", "DELETE") is False


def test_methods_none_registration_widens_restricted_one_to_all():
    # None ("all") is the universal set: unioning it with a restricted set
    # yields "all", matching the OR semantics of repeat registration.
    token_auth.register_token_route("/api/thing", methods=("GET",))
    token_auth.register_token_route("/api/thing")  # methods=None
    assert token_auth.is_token_route_method_allowed("/api/thing", "DELETE") is True


def test_methods_empty_set_is_treated_as_all_not_deny_all():
    # A degenerate empty method set is almost certainly a caller bug; it must
    # not silently deny every method (which would break the route). Treated
    # as "all", the same as None.
    token_auth.register_token_route("/api/thing", methods=())
    assert token_auth.is_token_route_method_allowed("/api/thing", "GET") is True
    assert token_auth.is_token_route_method_allowed("/api/thing", "POST") is True


def test_seam_falls_through_on_disallowed_method_even_with_valid_token():
    # The end-to-end security property at the unit level: a valid token on a
    # DISALLOWED method must NOT be authenticated by the seam — it falls
    # through (call_next) WITHOUT setting token_authenticated, so the
    # downstream cookie gate decides. (required=False so a bare fallthrough
    # is the expected no-op path; the point is that a *presented, valid*
    # token still does not flip token_authenticated for a write verb.)
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/skills", required=False, methods=("GET", "HEAD"))
    req = _FakeRequest(
        path="/api/skills",
        headers={"authorization": "Bearer good"},
        method="POST",
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200  # _call_next_ok stands in for the downstream gate
    assert getattr(req.state, "token_authenticated", False) is False
    assert getattr(req.state, "token_principal", None) is None


def test_seam_authenticates_valid_token_on_allowed_method():
    # Contrast: the SAME token on an ALLOWED method (GET) is authenticated.
    register_provider(_TokenProvider(secret="good"))
    token_auth.register_token_route("/api/skills", required=False, methods=("GET", "HEAD"))
    req = _FakeRequest(
        path="/api/skills",
        headers={"authorization": "Bearer good"},
        method="GET",
    )
    resp = _run(token_auth.token_auth_middleware(req, _call_next_ok))
    assert resp.status_code == 200
    assert req.state.token_authenticated is True


def test_clear_token_routes_clears_method_restrictions():
    token_auth.register_token_route("/api/skills", methods=("GET",))
    token_auth.clear_token_routes()
    token_auth.register_token_route("/api/skills")  # methods=None
    assert token_auth.is_token_route_method_allowed("/api/skills", "POST") is True
