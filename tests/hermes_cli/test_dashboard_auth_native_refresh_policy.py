"""Refresh retry policy for the RFC 8252 native-app refresh path (Refs #98338).

Defects 2–3: ``POST /auth/native/refresh`` fanned every inbound attempt out to
the Portal token endpoint with no classification and no throttle — a dead
refresh token rejected with 403 was reported as ``ProviderError`` (503,
"transient"), so the client kept retrying at up to 3 req/s for 17h45m.

Contract pinned here:
  * a 401/403 carrying an OAuth error envelope is a *permanent* credential
    rejection (``bad_request_exc`` — ``RefreshExpiredError`` on refresh), so a
    dead token answers 401 ``session_expired`` instead of 503;
  * a 401/403 *without* an envelope stays ``ProviderError`` (may be a WAF /
    proxy, transient — never force a re-login on an ambiguous signal);
  * 429 / 5xx / transport failures stay ``ProviderError`` (transient);
  * inbound attempts are throttled per credential-hash (never per IP — NAT
    households share IPs, see #98338 request #6), answering 429 with
    ``Retry-After`` once over budget so one looping client cannot storm Portal.

Run: scripts/run_tests.sh tests/hermes_cli/test_dashboard_auth_native_refresh_policy.py
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth import native_flow
from hermes_cli.dashboard_auth.base import ProviderError, RefreshExpiredError
from hermes_cli.dashboard_auth.routes import (
    _REFRESH_RATE_MAX_BUCKETS,
    _native_refresh_rate_limited,
    _refresh_attempts,
    _reset_native_refresh_rate_limit,
)
from plugins.dashboard_auth._shared import exchange_token
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


def _mock_post(status_code: int, body, *, ctype: str = "application/json"):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    if isinstance(body, dict):
        resp.text = json.dumps(body)
        resp.json = MagicMock(return_value=body)
    else:
        resp.text = body
        resp.json = MagicMock(side_effect=ValueError("not json"))
    resp.headers = {"content-type": ctype}
    return resp


def _exchange(status_code: int, body, **kwargs):
    with patch(
        "plugins.dashboard_auth._shared.httpx.post",
        return_value=_mock_post(status_code, body, **kwargs),
    ):
        return exchange_token(
            "https://portal.example.test/api/oauth/token",
            {"grant_type": "refresh_token"},
            bad_request_exc=RefreshExpiredError,
            idp="Portal",
            endpoint="Portal token endpoint",
            token_key="access_token",
            missing_msg="missing access_token",
        )


# ---------------------------------------------------------------------------
# Rejection classification (Defect 2 root cause)
# ---------------------------------------------------------------------------


class TestRefreshRejectionClassification:
    def test_403_with_oauth_envelope_is_permanent_rejection(self):
        with pytest.raises(RefreshExpiredError):
            _exchange(403, {"error": "invalid_grant"})

    def test_401_with_oauth_envelope_is_permanent_rejection(self):
        with pytest.raises(RefreshExpiredError):
            _exchange(401, {"error": "invalid_token"})

    def test_403_with_unknown_envelope_code_stays_transient(self):
        # A WAF/proxy JSON envelope without a known token-rejection code must
        # not force a re-login — nor may a non-JSON block page.
        with pytest.raises(ProviderError):
            _exchange(403, {"error": "forbidden"})
        with pytest.raises(ProviderError):
            _exchange(403, "<html>blocked</html>", ctype="text/html")

    def test_rejection_code_match_is_case_insensitive(self):
        with pytest.raises(RefreshExpiredError):
            _exchange(403, {"error": "Invalid_Grant"})

    def test_429_and_500_stay_transient(self):
        with pytest.raises(ProviderError):
            _exchange(429, {"error": "rate_limited"})
        with pytest.raises(ProviderError):
            _exchange(500, {"error": "server_error"})


# ---------------------------------------------------------------------------
# Per-credential inbound throttle (Defect 2 storm stop)
# ---------------------------------------------------------------------------


class _CountingStubProvider(StubAuthProvider):
    """Stub that counts refresh attempts so tests can prove Portal fan-out
    stays bounded under a retry storm."""

    def __init__(self):
        super().__init__()
        self.refresh_calls = 0

    def refresh_session(self, *, refresh_token: str):
        self.refresh_calls += 1
        return super().refresh_session(refresh_token=refresh_token)


@pytest.fixture(autouse=True)
def _reset_state():
    native_flow._reset_for_tests()
    _reset_native_refresh_rate_limit()
    prev_required = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    yield
    native_flow._reset_for_tests()
    _reset_native_refresh_rate_limit()
    clear_providers()
    web_server.app.state.auth_required = prev_required
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port


@pytest.fixture
def storm_client():
    provider = _CountingStubProvider()
    clear_providers()
    register_provider(provider)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(
        web_server.app, base_url="https://fly-app.fly.dev", follow_redirects=False
    )
    yield client, provider
    clear_providers()


class TestNativeRefreshThrottle:
    def test_storm_from_one_credential_is_capped_with_429(self, storm_client):
        client, provider = storm_client
        last = None
        for _ in range(25):
            last = client.post(
                "/auth/native/refresh",
                json={"refresh_token": "dead-token-same", "provider": "stub"},
            )
        assert last is not None
        assert last.status_code == 429
        assert last.json()["error"] == "rate_limited"
        assert int(last.headers["retry-after"]) > 0
        # Portal fan-out stays at the budget: the storm never reaches providers.
        assert provider.refresh_calls <= 10

    def test_distinct_credential_unaffected_by_storm(self, storm_client):
        client, provider = storm_client
        for _ in range(25):
            client.post(
                "/auth/native/refresh",
                json={"refresh_token": "dead-token-same", "provider": "stub"},
            )
        r = client.post(
            "/auth/native/refresh",
            json={"refresh_token": "other-dead-token", "provider": "stub"},
        )
        # A different credential is a different bucket: rejected on its own
        # merits (401, dead token), not throttled by the first storm.
        assert r.status_code == 401
        assert r.json()["error"] == "session_expired"

    def test_bucket_table_stays_bounded(self):
        # Token-rotation abuse must not grow the bucket table without limit.
        for i in range(_REFRESH_RATE_MAX_BUCKETS + 500):
            _native_refresh_rate_limited(f"rotated-token-{i}")
        assert len(_refresh_attempts) <= _REFRESH_RATE_MAX_BUCKETS + 1
        # The limiter still works after pruning.
        limited, _ = _native_refresh_rate_limited("fresh-token-after-prune")
        assert limited is False
