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


def test_stub_verify_tampered_token_returns_none():
    p = StubAuthProvider()
    assert p.verify_session(access_token="garbage-not-a-real-token") is None


