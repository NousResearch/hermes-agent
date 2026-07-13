"""Contract test for the StubAuthProvider used in dashboard-auth E2E tests.

Phase 2 of the dashboard-OAuth plan. Validates the stub against the
provider protocol so subsequent phases that depend on its behavior
have a guarantee.
"""
from __future__ import annotations

import pytest

from hermes_cli.dashboard_auth.base import (
    InvalidCodeError, RefreshExpiredError, assert_protocol_compliance,
)
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


def _pkce_payload(ls) -> dict:
    """Parse ``state=...;verifier=...`` out of the LoginStart cookie payload."""
    return dict(
        item.split("=", 1)
        for item in ls.cookie_payload["hermes_session_pkce"].split(";")
    )


def test_stub_complies_with_protocol():
    assert assert_protocol_compliance(StubAuthProvider) is None


def test_stub_start_login_returns_callback_redirect():
    p = StubAuthProvider()
    ls = p.start_login(redirect_uri="https://x.fly.dev/auth/callback")
    assert "code=stub_code" in ls.redirect_url
    assert "state=" in ls.redirect_url
    assert "hermes_session_pkce" in ls.cookie_payload


def test_stub_complete_login_with_matching_state_succeeds():
    p = StubAuthProvider()
    ls = p.start_login(redirect_uri="https://x.fly.dev/auth/callback")
    payload = _pkce_payload(ls)
    sess = p.complete_login(
        code="stub_code",
        state=payload["state"],
        code_verifier=payload["verifier"],
        redirect_uri="https://x.fly.dev/auth/callback",
    )
    assert sess.user_id == "stub-user-1"
    assert sess.email == "stub@example.test"
    assert sess.display_name == "Stub User"
    assert sess.org_id == "stub-org-1"
    assert sess.provider == "stub"
    assert sess.access_token and sess.refresh_token


def test_stub_complete_login_rejects_mismatched_state():
    p = StubAuthProvider()
    p.start_login(redirect_uri="https://x.fly.dev/auth/callback")
    with pytest.raises(InvalidCodeError):
        p.complete_login(
            code="stub_code",
            state="WRONG",
            code_verifier="anything",
            redirect_uri="https://x.fly.dev/auth/callback",
        )


def test_stub_complete_login_rejects_wrong_code():
    p = StubAuthProvider()
    ls = p.start_login(redirect_uri="https://x.fly.dev/auth/callback")
    payload = _pkce_payload(ls)
    with pytest.raises(InvalidCodeError):
        p.complete_login(
            code="BAD",
            state=payload["state"],
            code_verifier=payload["verifier"],
            redirect_uri="https://x.fly.dev/auth/callback",
        )


def test_stub_verify_session_round_trips():
    p = StubAuthProvider()
    ls = p.start_login(redirect_uri="https://x.fly.dev/auth/callback")
    payload = _pkce_payload(ls)
    sess = p.complete_login(
        code="stub_code",
        state=payload["state"],
        code_verifier=payload["verifier"],
        redirect_uri="https://x.fly.dev/auth/callback",
    )
    verified = p.verify_session(access_token=sess.access_token)
    assert verified is not None
    assert verified.user_id == "stub-user-1"
    assert verified.org_id == "stub-org-1"


def test_stub_verify_expired_session_returns_none():
    p = StubAuthProvider(default_ttl=0)
    ls = p.start_login(redirect_uri="https://x/auth/callback")
    payload = _pkce_payload(ls)
    sess = p.complete_login(
        code="stub_code",
        state=payload["state"],
        code_verifier=payload["verifier"],
        redirect_uri="https://x/auth/callback",
    )
    # default_ttl=0 means the access token is born already expired
    # (verify uses ``<=`` so exp == now counts as expired).
    assert p.verify_session(access_token=sess.access_token) is None


def test_stub_verify_tampered_token_returns_none():
    p = StubAuthProvider()
    assert p.verify_session(access_token="garbage-not-a-real-token") is None


def test_stub_refresh_round_trips():
    p = StubAuthProvider()
    ls = p.start_login(redirect_uri="https://x/auth/callback")
    payload = _pkce_payload(ls)
    sess = p.complete_login(
        code="stub_code",
        state=payload["state"],
        code_verifier=payload["verifier"],
        redirect_uri="https://x/auth/callback",
    )
    refreshed = p.refresh_session(refresh_token=sess.refresh_token)
    # Refresh must return a valid Session for the same identity. (Tokens
    # may compare equal byte-for-byte if the refresh happens within the
    # same wall-clock second as the original — payload contents are
    # otherwise identical and HMAC is deterministic. The behavioural
    # invariant is just "refresh succeeds and identity survives".)
    assert refreshed.user_id == "stub-user-1"
    assert refreshed.access_token  # non-empty
    assert refreshed.refresh_token  # non-empty
    # And the refreshed access_token is still verifiable.
    verified = p.verify_session(access_token=refreshed.access_token)
    assert verified is not None
    assert verified.user_id == "stub-user-1"


def test_stub_refresh_expired_raises():
    p = StubAuthProvider()
    with pytest.raises(RefreshExpiredError):
        p.refresh_session(refresh_token="garbage")


def test_stub_revoke_is_silent():
    p = StubAuthProvider()
    # Best-effort; must never raise.
    p.revoke_session(refresh_token="anything")
