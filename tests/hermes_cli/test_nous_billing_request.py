"""Tests for the hermes_cli.nous_billing HTTP client's response handling.

Focus: a 2xx response with a NON-JSON body (e.g. a reverse-proxy / SPA fallback
HTML page when a route isn't actually serving the billing API) must surface as a
typed BillingError, NOT a raw json.JSONDecodeError that escapes the typed-error
contract and reads downstream as "not logged in".
"""

from __future__ import annotations

import io
import json
import socket
from contextlib import contextmanager

import pytest

from hermes_cli import nous_billing as nb


class _FakeResp(io.BytesIO):
    """Minimal urlopen() context-manager stand-in with a .status attribute."""

    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _http_error(status: int, body: bytes | dict[str, object] = b"{}", headers=None):
    """Build the real urllib HTTPError object _request catches."""
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    return nb.urllib.error.HTTPError(
        "https://portal.example/api/billing/state",
        status,
        "HTTP Error",
        headers or {},
        fp=io.BytesIO(body),
    )


def _sequence(monkeypatch, *outcomes, resolver=None):
    """Stub urlopen with ordered outcomes and record each Request."""
    seen: list[dict[str, object]] = []
    monkeypatch.setattr(nb, "_token_cache", {}, raising=False)
    monkeypatch.setattr(
        nb,
        "_resolve_token_and_base",
        resolver or (lambda **kw: ("tok", "https://portal.example")),
    )

    def _fake_urlopen(req, timeout=None):
        seen.append(
            {
                "method": req.get_method(),
                "url": req.full_url,
                "data": json.loads(req.data.decode()) if req.data else None,
                "headers": {k.lower(): v for k, v in req.header_items()},
            }
        )
        outcome = outcomes[len(seen) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(nb.urllib.request, "urlopen", _fake_urlopen)
    return seen


@contextmanager
def _stub(monkeypatch, body: bytes, status: int = 200):
    # Bypass auth/token resolution entirely — we only exercise response parsing.
    monkeypatch.setattr(nb, "_resolve_token_and_base", lambda **kw: ("tok", "https://portal.example"))
    monkeypatch.setattr(nb, "_token_cache", {}, raising=False)
    monkeypatch.setattr(nb.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(body, status))
    yield








# ---------------------------------------------------------------------------
# Subscription change (V3): the request the client actually puts on the wire.
# ---------------------------------------------------------------------------


@contextmanager
def _capture(monkeypatch, body: bytes = b"{}", status: int = 200):
    """Stub urlopen, recording the urllib.request.Request the client built."""
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        nb, "_resolve_token_and_base", lambda **kw: ("tok", "https://portal.example")
    )

    def _fake_urlopen(req, timeout=None):
        seen["method"] = req.get_method()
        seen["url"] = req.full_url
        seen["data"] = json.loads(req.data.decode()) if req.data else None
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _FakeResp(body, status)

    monkeypatch.setattr(nb.urllib.request, "urlopen", _fake_urlopen)
    yield seen










# ---------------------------------------------------------------------------
# Wire-layer HTTP error mapping through _request.
# ---------------------------------------------------------------------------


def test_401_refreshes_token_and_retries_successfully(monkeypatch):
    # A cached-token 401 is retried once with a freshly resolved token.
    def _resolver(*, use_cache=True):
        token = "tok-fresh" if not use_cache else "tok-stale"
        return token, "https://portal.example"

    seen = _sequence(
        monkeypatch,
        _http_error(401),
        _FakeResp(b'{"ok": true}'),
        resolver=_resolver,
    )

    assert nb.get_billing_state() == {"ok": True}
    assert len(seen) == 2
    assert seen[0]["headers"]["authorization"] == "Bearer tok-stale"
    assert seen[1]["headers"]["authorization"] == "Bearer tok-fresh"








def test_403_remote_spending_revoked_maps_through_request(monkeypatch):
    # The wire discriminator keeps spend-revocation distinct from missing scope.
    _sequence(
        monkeypatch,
        _http_error(
            403,
            {
                "error": "remote_spending_revoked",
                "actor": "self",
                "recovery": "reconnect",
            },
        ),
    )

    with pytest.raises(nb.BillingRemoteSpendingRevoked) as ei:
        nb.get_billing_state()

    assert ei.value.actor == "self"
    assert ei.value.recovery == "reconnect"




def test_429_retry_after_header_maps_to_rate_limited(monkeypatch):
    # 429 is rate-limited and reads Retry-After from headers.
    _sequence(monkeypatch, _http_error(429, headers={"Retry-After": "15"}))

    with pytest.raises(nb.BillingRateLimited) as ei:
        nb.get_billing_state()

    assert ei.value.retry_after == 15


def test_non_json_second_401_maps_to_auth_error_not_session_revoked(monkeypatch):
    # The terminal 401 must not become session_revoked without a JSON discriminator.
    _sequence(
        monkeypatch,
        _http_error(401),
        _http_error(401, b"<html>Unauthorized</html>"),
    )

    with pytest.raises(nb.BillingAuthError) as ei:
        nb.get_billing_state()

    assert not isinstance(ei.value, nb.BillingSessionRevoked)


def test_404_get_charge_status_maps_to_generic_billing_error(monkeypatch):
    # get_charge_status should surface unexpected 404s as typed generic errors.
    _sequence(monkeypatch, _http_error(404))

    with pytest.raises(nb.BillingError) as ei:
        nb.get_charge_status("ch_404")

    assert type(ei.value) is nb.BillingError
    assert ei.value.status == 404








def test_billing_token_cache_is_scoped_per_profile_home(monkeypatch, tmp_path):
    """The 30s (token, base) memo must not hand profile A's Portal bearer to profile B under a
    multiplex gateway, where the per-turn HERMES_HOME override selects the auth.json."""
    import hermes_constants
    import hermes_cli.auth as auth

    monkeypatch.setattr(nb, "_token_cache", {}, raising=False)
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda provider: {})
    monkeypatch.setattr(
        auth, "resolve_nous_access_token",
        lambda **kw: f"tok-{hermes_constants.get_hermes_home().name}")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    seen = []
    for home in (a, b):
        tok = hermes_constants.set_hermes_home_override(str(home))
        try:
            seen.append(nb._resolve_token_and_base()[0])
        finally:
            hermes_constants.reset_hermes_home_override(tok)
    assert seen == ["tok-a", "tok-b"]
