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

import hashlib

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


class TestRetryAfterIsDeliveredNotJustComputed:
    """The header a client reads must be the value the limiter computed.

    ``_native_refresh_rate_limited``'s return value is unit-tested, but nothing compared
    it to what the route actually puts on the wire: a hardcoded "1" satisfies a
    presence check and tells a client to come back after one second against a 60s window.
    """

    def test_429_header_matches_the_computed_remainder(self, storm_client, monkeypatch):
        from hermes_cli.dashboard_auth import routes as routes_mod

        client, provider = storm_client
        monkeypatch.setattr(routes_mod, "_REFRESH_RATE_MAX_ATTEMPTS", 2)
        monkeypatch.setattr(routes_mod, "_REFRESH_RATE_WINDOW_SEC", 600.0)
        _reset_native_refresh_rate_limit()

        def _refresh(t):
            return client.post("/auth/native/refresh",
                               json={"refresh_token": t, "provider": "stub"})

        refused = None
        for _ in range(4):
            r = _refresh("retry-probe")
            if r.status_code == 429:
                refused = r
                break
        assert refused is not None, "the throttle never refused"
        # Recompute against the same table the route just wrote to. The limited path
        # returns before appending, so this does not spend budget.
        _, expected = _native_refresh_rate_limited("retry-probe")
        assert int(refused.headers["Retry-After"]) == int(expected), (
            "header %s does not match the computed remainder %.1f"
            % (refused.headers.get("Retry-After"), expected)
        )
        assert int(refused.headers["Retry-After"]) > 1, (
            "a 60s window must not report a 1s wait"
        )


class TestEvictionDoesNotForgiveAStorm:
    """A flood of one-shot credentials must not reset a storming client's budget.

    Evicting oldest-inserted drops whichever bucket arrived first — which is the one
    spending its budget — so a flood of throwaway tokens could forgive a client
    mid-attack and hand it a fresh allowance.
    """

    def test_storming_credential_keeps_its_budget_across_eviction(self, storm_client, monkeypatch):
        from hermes_cli.dashboard_auth import routes as routes_mod

        client, provider = storm_client
        monkeypatch.setattr(routes_mod, "_REFRESH_RATE_MAX_ATTEMPTS", 5)
        monkeypatch.setattr(routes_mod, "_REFRESH_RATE_WINDOW_SEC", 600.0)
        monkeypatch.setattr(routes_mod, "_REFRESH_RATE_MAX_BUCKETS", 8)
        _reset_native_refresh_rate_limit()

        def _refresh(token):
            return client.post("/auth/native/refresh",
                               json={"refresh_token": token, "provider": "stub"})

        # The stormer arrives first and spends its entire budget, so every bucket it
        # competes with holds a single one-shot attempt. Oldest-inserted eviction picks it
        # precisely because it arrived first; depth-preferring eviction does not, because
        # it is spending far more than anything else in the table.
        for _ in range(5):
            assert _refresh("stormer").status_code == 401
        for i in range(40):  # one-shot tokens push the table past the cap
            _refresh("oneshot-%d" % i)

        r = _refresh("stormer")
        assert r.status_code == 429, (
            "the storming credential's budget was reset by bucket eviction (status %s)"
            % r.status_code
        )
        # Still throttled directly at the limiter: the route response is not what carried it.
        assert _native_refresh_rate_limited("stormer")[0], (
            "the limiter forgot the storming credential"
        )


class TestBucketKeyHandlesUntrustedBytes:
    """The bucket key hashes whatever string arrives; none of it may raise here.

    Note the endpoint still has a pre-existing 500 on a lone surrogate: main's
    refresh_singleflight._refresh_provider encodes the token before this runs, so the
    request never reaches the limiter. That is upstream's bug, tracked separately; this
    guard keeps the limiter's own keying total.
    """

    def test_bucket_key_survives_a_lone_surrogate(self):
        from hermes_cli.dashboard_auth.routes import _refresh_token_bucket

        assert len(_refresh_token_bucket("\ud800")) == 64

    def test_valid_tokens_hash_unchanged(self):
        from hermes_cli.dashboard_auth.routes import _refresh_token_bucket

        assert _refresh_token_bucket("normal-token") == hashlib.sha256(
            b"normal-token").hexdigest()


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


class TestPermanentTokenErrorCodes:
    """`refresh_token_reused` is what Portal returns once a rotated RT is replayed. The
    first version of this PR listed invalid_grant/invalid_token/expired_token and omitted
    it, so a reused grant was still answered 503 "try later" — the retry-storm response.
    The set now reuses the repo's canonical dead-grant frozenset, so the two cannot drift."""

    @pytest.mark.parametrize("code", [
        "invalid_grant", "invalid_token", "expired_token", "refresh_token_reused",
    ])
    def test_dead_grant_codes_are_credential_verdicts(self, code):
        with pytest.raises(RefreshExpiredError):
            _exchange(401, {"error": code})
        with pytest.raises(RefreshExpiredError):
            _exchange(403, {"error": code})

    @pytest.mark.parametrize("code", ["forbidden", "unauthorized", "server_error",
                                      "temporarily_unavailable", "invalid_client"])
    def test_other_codes_stay_transient(self, code):
        """A WAF page or a proxy envelope must never force a re-login."""
        with pytest.raises(ProviderError):
            _exchange(403, {"error": code})

    def test_non_string_error_payload_stays_transient(self):
        """A JSON body with a non-string `error` must not crash or force a re-login."""
        with pytest.raises(ProviderError):
            _exchange(401, {"error": 42})

    def test_set_is_derived_from_the_canonical_one(self):
        from hermes_cli.auth import _OAUTH_GRANT_DEAD_CODES as canonical
        from plugins.dashboard_auth import _shared

        actual = _shared._PERMANENT_TOKEN_ERRORS
        assert canonical <= actual, (
            "the dead-grant set drifted from hermes_cli.auth._OAUTH_GRANT_DEAD_CODES: %r"
            % sorted(canonical - actual)
        )
        assert "expired_token" in actual


class TestBudgetExpires:
    """A throttled credential must recover once its window passes. Nothing pinned that:
    dropping the `while bucket and bucket[0] < cutoff` expiry entirely left all 8 tests green,
    which would mean a single burst locks a legitimate client out permanently."""

    def test_budget_recovers_after_the_window(self, monkeypatch):
        from hermes_cli.dashboard_auth import routes as routes_mod

        routes_mod._reset_native_refresh_rate_limit()
        token = "expiring-budget-token"
        # Exhaust the real budget, but keep the loop bound independent of the constant so
        # mutating it cannot spin this test instead of failing it.
        for _ in range(min(routes_mod._REFRESH_RATE_MAX_ATTEMPTS, 1000)):
            limited, _ = routes_mod._native_refresh_rate_limited(token)
            assert not limited
        limited, _ = routes_mod._native_refresh_rate_limited(token)
        assert limited, "the budget should be exhausted"

        # Advance past the window without sleeping: rewind the recorded attempts.
        real_monotonic = routes_mod.time.monotonic
        monkeypatch.setattr(routes_mod.time, "monotonic",
                            lambda: real_monotonic() + routes_mod._REFRESH_RATE_WINDOW_SEC + 1)
        limited, retry = routes_mod._native_refresh_rate_limited(token)
        assert not limited, (
            "a throttled credential stays blocked forever once its window has passed "
            "(retry_after=%r)" % retry
        )
        routes_mod._reset_native_refresh_rate_limit()


class TestBucketKeyIsNeverTheCredential:
    """The comment claims "the raw token never leaves this module as a dict key — only its
    hash". Nothing enforced that: keying the table by the raw token would park plaintext
    refresh tokens in a process-local dict and every test stayed green. Keying by IP instead
    would throttle a whole NAT household (#98338 request 6), so the hash is load-bearing on
    both sides."""

    def test_every_bucket_key_is_a_hex_digest_not_the_token(self):
        from hermes_cli.dashboard_auth.routes import _refresh_attempts, _refresh_token_bucket

        token = "sk-super-secret-refresh-token-value"
        _refresh_token_bucket(token)
        _native_refresh_rate_limited(token)
        assert _refresh_attempts, "the limiter recorded no bucket"
        for key in _refresh_attempts:
            assert key != token, "a refresh token is being used as a dict key: %r" % key
            assert len(key) == 64 and all(c in "0123456789abcdef" for c in key), (
                "bucket keys must be sha256 hex digests, got %r" % key
            )
        assert _refresh_token_bucket(token) != token

    def test_different_tokens_get_different_buckets(self):
        from hermes_cli.dashboard_auth.routes import _refresh_token_bucket

        assert _refresh_token_bucket("token-a") != _refresh_token_bucket("token-b")

    def test_retry_after_reflects_the_window_remainder(self):
        """Not just "> 0": a client that honours Retry-After must be told roughly when its
        budget frees up. A hardcoded 1.0 would pass a bare positive-value assertion."""
        from hermes_cli.dashboard_auth.routes import _REFRESH_RATE_MAX_ATTEMPTS, _REFRESH_RATE_WINDOW_SEC

        token = "retry-after-probe-token"
        for _ in range(_REFRESH_RATE_MAX_ATTEMPTS):
            limited, _retry = _native_refresh_rate_limited(token)
            assert not limited
        limited, retry_after = _native_refresh_rate_limited(token)
        assert limited
        assert retry_after > 1.0, (
            "Retry-After must reflect the window remainder, not a fixed 1.0s: %r" % retry_after
        )
        assert retry_after <= _REFRESH_RATE_WINDOW_SEC, retry_after
