"""MAS OAuth2 + legacy refresh tests for the Matrix adapter (#94096 v2).

matrix.org runs MAS: the legacy ``/_matrix/client/v3/refresh`` endpoint
rejects MAS-issued tokens, so the adapter must fall back to the OAuth2
token endpoint. Live verification (vlify, 2026-08-27) showed the
homeserver-side OIDC well-known 404s on matrix.org and the issuer host
differs from the homeserver, so v2 adds ``MATRIX_OAUTH_TOKEN_ENDPOINT``
and ``MATRIX_OIDC_ISSUER`` overrides, an MSC2965 homeserver-pointer
fallback, and proxy support.
"""
import asyncio
import json
import types

import pytest

from plugins.platforms.matrix import adapter as matrix_adapter


class _FakeApi:
    def __init__(self):
        self.token = "old-access"
        self.legacy_calls = 0

    async def request(self, method, path, body):
        self.legacy_calls += 1
        raise RuntimeError("M_UNKNOWN_TOKEN Invalid refresh token")


class _Ctx:
    """Context-manager wrapper for stubbed urlopen responses."""

    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_adapter(
    client_id="my-client",
    token_endpoint_override="",
    issuer_override="",
    homeserver="https://matrix-client.matrix.org",
):
    a = object.__new__(matrix_adapter.MatrixAdapter)
    a._refresh_token = "rt-1"
    a._homeserver = homeserver
    a._access_token = "old-access"
    a._refresh_lock = asyncio.Lock()
    a._oidc_client_id = client_id
    a._oidc_token_endpoint = None
    a._oidc_token_endpoint_override = token_endpoint_override
    a._oidc_issuer_override = issuer_override
    a._oidc_hint_logged = False
    a._refresh_attempts = 0
    return a


# ---------------------------------------------------------------- legacy ok

@pytest.mark.asyncio
async def test_legacy_refresh_success_applies_tokens():
    a = _make_adapter()
    api = _FakeApi()

    async def req(method, path, body):
        return {"access_token": "acc-2", "refresh_token": "rt-2"}

    api.request = req
    assert await a._refresh_access_token(api, reason="test") is True
    assert a._access_token == "acc-2"
    assert api.token == "acc-2"
    assert a._refresh_token == "rt-2"


# --------------------------------------------- legacy rejected -> MAS oauth

@pytest.mark.asyncio
async def test_legacy_rejection_probes_homeserver_oidc_then_msc2965(monkeypatch):
    """vlify live verification (Aug 27): the homeserver
    matrix-client.matrix.org has no /.well-known/openid-configuration
    (404) and no /.well-known/matrix/client either, so both probes
    return empty and we fall through to the clear hint log. This locks
    in the exact-URL discovery behavior the mock test was previously
    silently passing.
    """
    import urllib.request as _ureq

    a = _make_adapter(client_id="pub-client")
    api = _FakeApi()
    probed = []

    def fail_404(req, timeout=10):
        url = req if isinstance(req, str) else req.full_url
        probed.append(url)
        raise FileNotFoundError("404")

    monkeypatch.setattr(_ureq, "urlopen", fail_404, raising=True)
    monkeypatch.delenv("MATRIX_OAUTH_TOKEN_ENDPOINT", raising=False)
    monkeypatch.delenv("MATRIX_OIDC_ISSUER", raising=False)

    assert await a._refresh_access_token(api, reason="test") is False
    # Both homeserver probes attempted in order.
    assert any(".well-known/openid-configuration" in u for u in probed)
    assert any(".well-known/matrix/client" in u for u in probed)
    # The exact homeserver base must be the one we constructed.
    assert any(u.startswith("https://matrix-client.matrix.org/") for u in probed)


# ----------------------------------------------------- direct token override

@pytest.mark.asyncio
async def test_oauth_token_endpoint_override_skips_discovery(monkeypatch):
    """``MATRIX_OAUTH_TOKEN_ENDPOINT`` bypasses discovery entirely: the
    config override is the documented matrix.org escape hatch when the
    homeserver has no machine-readable well-known pointer. The LEGACY
    ``/v3/refresh`` attempt still runs first (it must reject to fall
    through); only the OIDC discovery step is skipped."""
    import urllib.request as _ureq

    a = _make_adapter(
        client_id="pub",
        token_endpoint_override="https://account.matrix.org/oauth2/token",
    )
    api = _FakeApi()  # legacy request raises M_UNKNOWN_TOKEN -> fall through
    probed = []

    def routed(req, timeout=15):
        url = req if isinstance(req, str) else req.full_url
        probed.append(url)
        if "openid-configuration" in url or "well-known" in url:
            raise AssertionError("discovery must be skipped when override set")
        return _Ctx({
            "access_token": "ov-acc",
            "refresh_token": "ov-rt",
        })

    monkeypatch.setattr(_ureq, "urlopen", routed, raising=True)

    assert await a._refresh_access_token(api, reason="test") is True
    # No discovery/well-known URL was probed.
    assert not any("well-known" in u for u in probed)
    # The token POST went straight to the override endpoint.
    assert "https://account.matrix.org/oauth2/token" in probed
    assert a._access_token == "ov-acc"
    assert a._refresh_token == "ov-rt"


# -------------------------------------------------------- issuer override

@pytest.mark.asyncio
async def test_oidc_issuer_override_discovers_at_issuer(monkeypatch):
    """``MATRIX_OIDC_ISSUER`` (e.g. ``https://account.matrix.org``) makes
    the discover probe hit the issuer directly, skipping the homeserver
    well-known entirely."""
    import urllib.request as _ureq

    a = _make_adapter(
        client_id="pub",
        issuer_override="https://account.matrix.org",
    )
    api = _FakeApi()
    captured = {"probed": []}

    def routed(req, timeout=10):
        url = req if isinstance(req, str) else req.full_url
        captured["probed"].append(url)
        if "openid-configuration" in url:
            return _Ctx(
                {"token_endpoint": "https://account.matrix.org/oauth2/token"}
            )
        # Token POST.
        captured["token_url"] = url
        return _Ctx({
            "access_token": "iss-acc",
            "refresh_token": "iss-rt",
        })

    monkeypatch.setattr(_ureq, "urlopen", routed, raising=True)
    monkeypatch.delenv("MATRIX_OAUTH_TOKEN_ENDPOINT", raising=False)

    assert await a._refresh_access_token(api, reason="test") is True
    # No homeserver probe was attempted.
    assert all("matrix-client.matrix.org" not in u for u in captured["probed"])
    # The discovery probe hit the issuer.
    assert any("account.matrix.org" in u for u in captured["probed"])
    assert captured["token_url"] == "https://account.matrix.org/oauth2/token"


# -------------------------------------------------------------- proxy

@pytest.mark.asyncio
async def test_http_proxy_honored_for_mas_refresh(monkeypatch):
    """The MAS refresh path must honor ``MATRIX_PROXY`` so deployments
    behind an outbound proxy work (the legacy aiohttp path does)."""
    import urllib.request as _ureq

    a = _make_adapter(
        client_id="pub",
        token_endpoint_override="https://account.matrix.org/oauth2/token",
    )
    api = _FakeApi()
    captured = {"proxies": None}

    def record_proxy_handler(proxies=None):
        captured["proxies"] = proxies
        return object()  # unused; build_opener is stubbed

    class _Opener:
        def open(self, req, timeout=15):
            return _Ctx({
                "access_token": "px-acc",
                "refresh_token": "px-rt",
            })

    monkeypatch.setattr(_ureq, "ProxyHandler", record_proxy_handler)
    monkeypatch.setattr(_ureq, "build_opener", lambda *handlers: _Opener())
    monkeypatch.setenv("MATRIX_PROXY", "http://proxy.local:7897")
    monkeypatch.delenv("MATRIX_OAUTH_TOKEN_ENDPOINT", raising=False)

    assert await a._refresh_access_token(api, reason="test") is True
    assert captured["proxies"] == {
        "http": "http://proxy.local:7897",
        "https": "http://proxy.local:7897",
    }
    assert a._access_token == "px-acc"


# --------------------------------------------------------- bound / hint

@pytest.mark.asyncio
async def test_refresh_attempts_bounded(monkeypatch):
    import urllib.request as _ureq

    a = _make_adapter(client_id="pub")
    api = _FakeApi()
    monkeypatch.setenv("MATRIX_OIDC_CLIENT_ID", "pub")

    def always_fail(req, timeout=10):
        raise RuntimeError("M_UNKNOWN_TOKEN Invalid refresh token")

    monkeypatch.setattr(_ureq, "urlopen", always_fail, raising=True)

    for _ in range(matrix_adapter._MAX_REFRESH_ATTEMPTS + 3):
        ok = await a._refresh_access_token(api, reason="test")
        assert ok is False

    calls_after_cap = api.legacy_calls
    assert await a._refresh_access_token(api, reason="test") is False
    assert api.legacy_calls == calls_after_cap
    assert a._refresh_attempts >= matrix_adapter._MAX_REFRESH_ATTEMPTS


@pytest.mark.asyncio
async def test_success_resets_attempt_counter_chain_survives(monkeypatch):
    """32h live repro (#94096 review): MAS rotates tokens every ~4h, so a
    LIFETIME cap of 8 attempts exhausts a perfectly valid refresh chain
    after ~32h and the adapter gives up forever. The counter must reset
    on every successful refresh — only consecutive failures trip it."""
    a = _make_adapter()
    api = _FakeApi()

    async def ok_req(method, path, body):
        return {"access_token": "acc-1", "refresh_token": "rt-1"}

    api.request = ok_req

    # Simulate many success/fail cycles spanning days of runtime: each
    # success must reset the counter so the next failure sequence starts
    # from zero.
    async def fail_req(method, path, body):
        raise RuntimeError("M_UNKNOWN_TOKEN Invalid refresh token")

    for cycle in range(matrix_adapter._MAX_REFRESH_ATTEMPTS + 2):
        # Successful refresh (rotation happens every ~4h in production).
        assert await a._refresh_access_token(api, reason="cycle") is True
        assert a._refresh_attempts == 0, (
            "success must reset the consecutive-failure counter"
        )
        # One transient failure (network blip) before the next success.
        api.request = fail_req
        assert await a._refresh_access_token(api, reason="blip") is False
        assert a._refresh_attempts == 1
        api.request = ok_req

    # After all cycles the chain is still alive — a lifetime cap would
    # have given up around cycle 8.
    assert await a._refresh_access_token(api, reason="still-alive") is True


@pytest.mark.asyncio
async def test_mas_path_requires_client_id(monkeypatch):
    import urllib.request as _ureq

    a = _make_adapter(client_id="")
    api = _FakeApi()
    monkeypatch.delenv("MATRIX_OIDC_CLIENT_ID", raising=False)
    monkeypatch.setattr(
        _ureq, "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")),
        raising=True,
    )
    assert await a._refresh_access_token(api, reason="test") is False
