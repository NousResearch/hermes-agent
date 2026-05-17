"""Tests for xAI Grok OAuth — tokens stored in Hermes auth store (~/.hermes/auth.json)."""

import base64
import json
import time
from pathlib import Path

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_XAI_OAUTH_BASE_URL,
    PROVIDER_REGISTRY,
    XAI_OAUTH_CLIENT_ID,
    XAI_OAUTH_REDIRECT_HOST,
    XAI_OAUTH_REDIRECT_PATH,
    XAI_OAUTH_SCOPE,
    _read_xai_oauth_tokens,
    _save_xai_oauth_tokens,
    _xai_access_token_is_expiring,
    _xai_callback_cors_origin,
    _xai_oauth_build_authorize_url,
    _xai_validate_loopback_redirect_uri,
    get_xai_oauth_auth_status,
    refresh_xai_oauth_pure,
    resolve_provider,
    resolve_xai_oauth_runtime_credentials,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_hermes_auth(
    hermes_home: Path,
    *,
    access_token: str = "access",
    refresh_token: str = "refresh",
    discovery: dict | None = None,
):
    """Write xAI OAuth tokens into the Hermes auth store at the given root."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    state = {
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
        "last_refresh": "2026-05-14T00:00:00Z",
        "auth_mode": "oauth_pkce",
    }
    if discovery is not None:
        state["discovery"] = discovery
    auth_store = {
        "version": 1,
        "active_provider": "xai-oauth",
        "providers": {"xai-oauth": state},
    }
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps(auth_store, indent=2))
    return auth_file


def _jwt_with_exp(exp_epoch: int) -> str:
    """Build a minimal JWT-shaped string with the given exp claim."""
    payload = {"exp": exp_epoch}
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
        .rstrip(b"=")
        .decode("utf-8")
    )
    return f"h.{encoded}.s"


class _StubHTTPResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StubHTTPClient:
    def __init__(self, response):
        self._response = response
        self.last_call = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        self.last_call = ("post", args, kwargs)
        return self._response


def _patch_httpx_client(monkeypatch, response):
    holder = {"client": None}

    def _factory(*args, **kwargs):
        client = _StubHTTPClient(response)
        holder["client"] = client
        return client

    monkeypatch.setattr("hermes_cli.auth.httpx.Client", _factory)
    return holder


# ---------------------------------------------------------------------------
# Constants and registry
# ---------------------------------------------------------------------------


def test_xai_oauth_provider_registered():
    assert "xai-oauth" in PROVIDER_REGISTRY
    pconfig = PROVIDER_REGISTRY["xai-oauth"]
    assert pconfig.id == "xai-oauth"
    assert pconfig.auth_type == "oauth_external"
    assert pconfig.inference_base_url == DEFAULT_XAI_OAUTH_BASE_URL


def test_resolve_provider_normalizes_xai_oauth_aliases():
    assert resolve_provider("xai-oauth") == "xai-oauth"
    assert resolve_provider("grok-oauth") == "xai-oauth"
    assert resolve_provider("x-ai-oauth") == "xai-oauth"
    assert resolve_provider("xai-grok-oauth") == "xai-oauth"


# ---------------------------------------------------------------------------
# JWT expiry detection
# ---------------------------------------------------------------------------


def test_xai_access_token_is_expiring_returns_true_for_expired_jwt():
    expired = _jwt_with_exp(int(time.time()) - 60)
    assert _xai_access_token_is_expiring(expired, 0) is True


def test_xai_access_token_is_expiring_returns_false_for_fresh_jwt():
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    assert _xai_access_token_is_expiring(fresh, 0) is False


def test_xai_access_token_is_expiring_honors_skew_window():
    near = _jwt_with_exp(int(time.time()) + 30)
    assert _xai_access_token_is_expiring(near, 60) is True
    assert _xai_access_token_is_expiring(near, 0) is False


def test_xai_access_token_is_expiring_returns_false_for_non_jwt():
    assert _xai_access_token_is_expiring("not.a.jwt.but.has.dots", 0) is False
    assert _xai_access_token_is_expiring("opaque-token-no-dots", 0) is False
    assert _xai_access_token_is_expiring("", 0) is False
    assert _xai_access_token_is_expiring(None, 0) is False  # type: ignore[arg-type]


def test_xai_access_token_is_expiring_returns_false_for_jwt_without_exp():
    payload = {"sub": "user"}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode()
    token = f"h.{encoded}.s"
    assert _xai_access_token_is_expiring(token, 0) is False


# ---------------------------------------------------------------------------
# Loopback redirect URI validation
# ---------------------------------------------------------------------------


def test_xai_validate_loopback_redirect_uri_accepts_localhost_with_port():
    host, port, path = _xai_validate_loopback_redirect_uri(
        "http://127.0.0.1:56121/callback"
    )
    assert host == XAI_OAUTH_REDIRECT_HOST
    assert port == 56121
    assert path == XAI_OAUTH_REDIRECT_PATH


def test_xai_validate_loopback_redirect_uri_rejects_https():
    with pytest.raises(AuthError) as exc:
        _xai_validate_loopback_redirect_uri("https://127.0.0.1:56121/callback")
    assert exc.value.code == "xai_redirect_invalid"


def test_xai_validate_loopback_redirect_uri_rejects_non_loopback():
    with pytest.raises(AuthError) as exc:
        _xai_validate_loopback_redirect_uri("http://example.com:56121/callback")
    assert exc.value.code == "xai_redirect_invalid"


def test_xai_validate_loopback_redirect_uri_rejects_missing_port():
    with pytest.raises(AuthError) as exc:
        _xai_validate_loopback_redirect_uri("http://127.0.0.1/callback")
    assert exc.value.code == "xai_redirect_invalid"


# ---------------------------------------------------------------------------
# Authorize URL construction
# ---------------------------------------------------------------------------


def _parse_authorize_url(url: str) -> dict:
    from urllib.parse import urlparse, parse_qs

    parsed = urlparse(url)
    return {k: v[0] for k, v in parse_qs(parsed.query).items()}


def test_xai_oauth_authorize_url_includes_plan_generic():
    """Regression: accounts.x.ai requires `plan=generic` for loopback OAuth on
    non-allowlisted clients. Must always be present on the authorize URL."""
    url = _xai_oauth_build_authorize_url(
        authorization_endpoint="https://auth.x.ai/oauth2/authorize",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_challenge="challenge-xyz",
        state="state-abc",
        nonce="nonce-def",
    )
    params = _parse_authorize_url(url)
    assert params["plan"] == "generic"


def test_xai_oauth_authorize_url_includes_referrer_hermes_agent():
    """Attribution: xAI's OAuth server can identify Hermes-originated logins
    via the referrer query param. Must always be present on the authorize URL."""
    url = _xai_oauth_build_authorize_url(
        authorization_endpoint="https://auth.x.ai/oauth2/authorize",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_challenge="challenge-xyz",
        state="state-abc",
        nonce="nonce-def",
    )
    params = _parse_authorize_url(url)
    assert params["referrer"] == "hermes-agent"


def test_xai_oauth_authorize_url_includes_pkce_and_oidc_params():
    url = _xai_oauth_build_authorize_url(
        authorization_endpoint="https://auth.x.ai/oauth2/authorize",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_challenge="challenge-xyz",
        state="state-abc",
        nonce="nonce-def",
    )
    params = _parse_authorize_url(url)
    assert params["response_type"] == "code"
    assert params["client_id"] == XAI_OAUTH_CLIENT_ID
    assert params["redirect_uri"] == "http://127.0.0.1:56121/callback"
    assert params["scope"] == XAI_OAUTH_SCOPE
    assert params["code_challenge"] == "challenge-xyz"
    assert params["code_challenge_method"] == "S256"
    assert params["state"] == "state-abc"
    assert params["nonce"] == "nonce-def"


# ---------------------------------------------------------------------------
# CORS allowlist
# ---------------------------------------------------------------------------


def test_xai_callback_cors_origin_allowlist():
    assert _xai_callback_cors_origin("https://accounts.x.ai") == "https://accounts.x.ai"
    assert _xai_callback_cors_origin("https://auth.x.ai") == "https://auth.x.ai"


def test_xai_callback_cors_origin_rejects_unknown_origin():
    assert _xai_callback_cors_origin("https://attacker.example.com") == ""
    assert _xai_callback_cors_origin(None) == ""
    assert _xai_callback_cors_origin("") == ""


# ---------------------------------------------------------------------------
# Token roundtrip + reads
# ---------------------------------------------------------------------------


def test_save_and_read_xai_oauth_tokens_roundtrip(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_xai_oauth_tokens(
        {
            "access_token": "at-1",
            "refresh_token": "rt-1",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
        discovery={"token_endpoint": "https://auth.x.ai/oauth2/token"},
        redirect_uri="http://127.0.0.1:56121/callback",
    )
    data = _read_xai_oauth_tokens()
    assert data["tokens"]["access_token"] == "at-1"
    assert data["tokens"]["refresh_token"] == "rt-1"
    assert data["redirect_uri"] == "http://127.0.0.1:56121/callback"
    assert data["discovery"]["token_endpoint"] == "https://auth.x.ai/oauth2/token"


def test_read_xai_oauth_tokens_missing(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        _read_xai_oauth_tokens()
    assert exc.value.code == "xai_auth_missing"
    assert exc.value.relogin_required is True


def test_read_xai_oauth_tokens_missing_access_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        _read_xai_oauth_tokens()
    assert exc.value.code == "xai_auth_missing_access_token"
    assert exc.value.relogin_required is True


def test_read_xai_oauth_tokens_missing_refresh_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, refresh_token="")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        _read_xai_oauth_tokens()
    assert exc.value.code == "xai_auth_missing_refresh_token"
    assert exc.value.relogin_required is True


# ---------------------------------------------------------------------------
# Runtime credential resolution
# ---------------------------------------------------------------------------


def test_resolve_xai_runtime_credentials_returns_singleton_state(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    creds = resolve_xai_oauth_runtime_credentials()
    assert creds["provider"] == "xai-oauth"
    assert creds["api_key"] == fresh
    assert creds["base_url"] == DEFAULT_XAI_OAUTH_BASE_URL
    assert creds["source"] == "hermes-auth-store"
    assert creds["auth_mode"] == "oauth_pkce"


def test_resolve_xai_runtime_credentials_refreshes_expiring_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    expiring = _jwt_with_exp(int(time.time()) - 10)
    _setup_hermes_auth(
        hermes_home,
        access_token=expiring,
        refresh_token="rt-old",
        discovery={"token_endpoint": "https://auth.x.ai/oauth2/token"},
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    new_access = _jwt_with_exp(int(time.time()) + 3600)
    called = {"count": 0}

    def _fake_refresh(tokens, **kwargs):
        called["count"] += 1
        updated = dict(tokens)
        updated["access_token"] = new_access
        updated["refresh_token"] = "rt-new"
        return updated

    monkeypatch.setattr("hermes_cli.auth._refresh_xai_oauth_tokens", _fake_refresh)

    creds = resolve_xai_oauth_runtime_credentials()
    assert called["count"] == 1
    assert creds["api_key"] == new_access


def test_resolve_xai_runtime_credentials_force_refresh(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(
        hermes_home,
        access_token=fresh,
        discovery={"token_endpoint": "https://auth.x.ai/oauth2/token"},
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    forced = _jwt_with_exp(int(time.time()) + 7200)
    called = {"count": 0}

    def _fake_refresh(tokens, **kwargs):
        called["count"] += 1
        updated = dict(tokens)
        updated["access_token"] = forced
        return updated

    monkeypatch.setattr("hermes_cli.auth._refresh_xai_oauth_tokens", _fake_refresh)

    creds = resolve_xai_oauth_runtime_credentials(force_refresh=True, refresh_if_expiring=False)
    assert called["count"] == 1
    assert creds["api_key"] == forced


def test_resolve_xai_runtime_credentials_honours_env_base_url(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_XAI_BASE_URL", "https://custom.x.ai/v1/")

    creds = resolve_xai_oauth_runtime_credentials()
    assert creds["base_url"] == "https://custom.x.ai/v1"


# ---------------------------------------------------------------------------
# Auth status surface
# ---------------------------------------------------------------------------


def test_get_xai_oauth_auth_status_logged_in_via_singleton(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    status = get_xai_oauth_auth_status()
    assert status["logged_in"] is True
    assert status["api_key"] == fresh
    assert status["auth_mode"] == "oauth_pkce"


def test_get_xai_oauth_auth_status_logged_out(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    status = get_xai_oauth_auth_status()
    assert status["logged_in"] is False
    assert "error" in status


# ---------------------------------------------------------------------------
# refresh_xai_oauth_pure error handling
# ---------------------------------------------------------------------------


def test_refresh_xai_oauth_pure_requires_refresh_token():
    with pytest.raises(AuthError) as exc:
        refresh_xai_oauth_pure("at", "")
    assert exc.value.code == "xai_auth_missing_refresh_token"
    assert exc.value.relogin_required is True


def test_refresh_xai_oauth_pure_relogin_on_400(monkeypatch):
    response = _StubHTTPResponse(400, {"error": "invalid_grant"})
    _patch_httpx_client(monkeypatch, response)
    with pytest.raises(AuthError) as exc:
        refresh_xai_oauth_pure(
            "at", "rt", token_endpoint="https://auth.x.ai/oauth2/token"
        )
    assert exc.value.code == "xai_refresh_failed"
    assert exc.value.relogin_required is True


def test_refresh_xai_oauth_pure_no_relogin_on_500(monkeypatch):
    response = _StubHTTPResponse(503, "service unavailable")
    _patch_httpx_client(monkeypatch, response)
    with pytest.raises(AuthError) as exc:
        refresh_xai_oauth_pure(
            "at", "rt", token_endpoint="https://auth.x.ai/oauth2/token"
        )
    assert exc.value.code == "xai_refresh_failed"
    assert exc.value.relogin_required is False


def test_refresh_xai_oauth_pure_returns_updated_tokens(monkeypatch):
    new_access = _jwt_with_exp(int(time.time()) + 3600)
    response = _StubHTTPResponse(
        200,
        {
            "access_token": new_access,
            "refresh_token": "rt-rotated",
            "id_token": "id-1",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    holder = _patch_httpx_client(monkeypatch, response)

    updated = refresh_xai_oauth_pure(
        "at", "rt-old", token_endpoint="https://auth.x.ai/oauth2/token"
    )
    assert updated["access_token"] == new_access
    assert updated["refresh_token"] == "rt-rotated"
    assert updated["id_token"] == "id-1"
    assert updated["token_type"] == "Bearer"
    assert updated["last_refresh"].endswith("Z")
    client = holder["client"]
    assert client is not None
    _method, _args, kwargs = client.last_call
    assert kwargs["data"]["grant_type"] == "refresh_token"
    assert kwargs["data"]["refresh_token"] == "rt-old"
    assert kwargs["data"]["client_id"] == XAI_OAUTH_CLIENT_ID


def test_refresh_xai_oauth_pure_keeps_refresh_token_when_response_omits_it(monkeypatch):
    """Some OAuth providers don't rotate refresh tokens — preserve the old one."""
    new_access = _jwt_with_exp(int(time.time()) + 3600)
    response = _StubHTTPResponse(
        200,
        {
            "access_token": new_access,
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    _patch_httpx_client(monkeypatch, response)

    updated = refresh_xai_oauth_pure(
        "at", "rt-stable", token_endpoint="https://auth.x.ai/oauth2/token"
    )
    assert updated["access_token"] == new_access
    assert updated["refresh_token"] == "rt-stable"


def test_refresh_xai_oauth_pure_rejects_response_without_access_token(monkeypatch):
    response = _StubHTTPResponse(
        200,
        {"refresh_token": "rt-new", "expires_in": 3600},
    )
    _patch_httpx_client(monkeypatch, response)
    with pytest.raises(AuthError) as exc:
        refresh_xai_oauth_pure(
            "at", "rt", token_endpoint="https://auth.x.ai/oauth2/token"
        )
    assert exc.value.code == "xai_refresh_missing_access_token"
    assert exc.value.relogin_required is True


def test_refresh_xai_oauth_pure_raises_typed_error_on_malformed_json(monkeypatch):
    """xAI returning HTTP 200 with a non-JSON body (captive portal, proxy
    error page, etc.) must surface a typed AuthError, not a raw
    ``json.JSONDecodeError`` traceback. Matches the qwen-oauth precedent
    so the upstream UX layer (``format_auth_error``) can map the failure."""
    response = _StubHTTPResponse(200, ValueError("not json"))
    response.text = "<html>captive portal</html>"
    _patch_httpx_client(monkeypatch, response)
    with pytest.raises(AuthError) as exc:
        refresh_xai_oauth_pure(
            "at", "rt", token_endpoint="https://auth.x.ai/oauth2/token"
        )
    assert exc.value.code == "xai_refresh_invalid_json"


def test_xai_oauth_discovery_raises_typed_error_on_malformed_json(monkeypatch):
    """Discovery is a cold-start, one-time fetch.  If the response is HTTP
    200 with a non-JSON body (corporate proxy / captive portal returning
    HTML), surface a typed AuthError rather than letting the
    ``json.JSONDecodeError`` escape — so the message reads as an auth
    problem instead of an internal parsing crash."""
    from hermes_cli.auth import _xai_oauth_discovery

    class _BadJSON:
        status_code = 200

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(
        "hermes_cli.auth.httpx.get",
        lambda *a, **kw: _BadJSON(),
    )
    with pytest.raises(AuthError) as exc:
        _xai_oauth_discovery()
    assert exc.value.code == "xai_discovery_invalid_json"


def test_xai_oauth_discovery_raises_typed_error_on_non_object_payload(monkeypatch):
    """A discovery body that decodes as JSON but isn't an object (e.g. a
    bare string or array) must not slip through and trigger an
    ``AttributeError`` on ``payload.get(...)`` later.  Reject loudly
    with the same incomplete-response code the missing-endpoint path uses."""
    from hermes_cli.auth import _xai_oauth_discovery

    class _StubResponse:
        status_code = 200

        def json(self):
            return ["not", "an", "object"]

    monkeypatch.setattr(
        "hermes_cli.auth.httpx.get",
        lambda *a, **kw: _StubResponse(),
    )
    with pytest.raises(AuthError) as exc:
        _xai_oauth_discovery()
    assert exc.value.code == "xai_discovery_incomplete"


# ---------------------------------------------------------------------------
# OIDC discovery endpoint origin/scheme validation (MITM hardening)
# ---------------------------------------------------------------------------


def test_refresh_xai_oauth_pure_rejects_non_https_token_endpoint(monkeypatch):
    """A poisoned auth.json (from MITM during initial discovery, or an older
    Hermes that didn't validate) must not be silently honored on the refresh
    hot path. A non-HTTPS ``token_endpoint`` would leak the refresh_token in
    cleartext on every refresh; refuse before the POST."""
    # No HTTP stub installed — refresh must fail at validation, not at POST.
    with pytest.raises(AuthError) as exc:
        refresh_xai_oauth_pure(
            "at", "rt", token_endpoint="http://auth.x.ai/oauth2/token"
        )
    assert exc.value.code == "xai_discovery_invalid"


def test_refresh_xai_oauth_pure_rejects_off_origin_token_endpoint(monkeypatch):
    """Pin the cached token_endpoint host to the xAI origin. A one-time MITM
    during discovery could persist a token_endpoint on attacker-controlled
    infrastructure — every subsequent refresh would silently leak the
    refresh_token to that attacker. Refuse off-origin endpoints loudly so
    the user can re-run discovery."""
    with pytest.raises(AuthError) as exc:
        refresh_xai_oauth_pure(
            "at", "rt", token_endpoint="https://evil.example.com/token"
        )
    assert exc.value.code == "xai_discovery_invalid"


def test_refresh_xai_oauth_pure_rejects_lookalike_suffix(monkeypatch):
    """Substring confusion: ``evil-x.ai`` ends in ``x.ai`` but is NOT a
    ``.x.ai`` subdomain. The validator must enforce the leading-dot suffix
    so attacker-registered apex lookalikes can't slip through."""
    with pytest.raises(AuthError) as exc:
        refresh_xai_oauth_pure(
            "at", "rt", token_endpoint="https://evilx.ai/token"
        )
    assert exc.value.code == "xai_discovery_invalid"


def test_refresh_xai_oauth_pure_accepts_apex_and_subdomain_endpoints(monkeypatch):
    """The validator must accept BOTH the bare xAI apex (``x.ai``) and any
    ``*.x.ai`` subdomain (e.g. ``auth.x.ai`` today, future migrations to
    ``accounts.x.ai`` etc.). Without subdomain support we'd lock the
    integration to whatever xAI happens to use today."""
    new_access = _jwt_with_exp(int(time.time()) + 3600)
    response = _StubHTTPResponse(
        200,
        {"access_token": new_access, "expires_in": 3600, "token_type": "Bearer"},
    )
    _patch_httpx_client(monkeypatch, response)
    # auth.x.ai (current production)
    updated = refresh_xai_oauth_pure(
        "at", "rt", token_endpoint="https://auth.x.ai/oauth2/token"
    )
    assert updated["access_token"] == new_access
    # hypothetical migration to accounts.x.ai
    _patch_httpx_client(monkeypatch, response)
    updated2 = refresh_xai_oauth_pure(
        "at", "rt", token_endpoint="https://accounts.x.ai/token"
    )
    assert updated2["access_token"] == new_access


def test_xai_oauth_discovery_validates_endpoints(monkeypatch):
    """The discovery response itself goes through endpoint validation, so a
    one-time MITM during initial login cannot poison ``auth.json`` with an
    attacker-controlled ``token_endpoint``. (The persistence is what makes
    this attack worth defending against — one MITM = forever credential
    leak.)"""
    from hermes_cli.auth import _xai_oauth_discovery

    class _StubGetResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def _fake_get(url, headers=None, timeout=None):
        return _StubGetResponse({
            "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
            "token_endpoint": "https://evil.example.com/token",  # poisoned
        })

    monkeypatch.setattr("hermes_cli.auth.httpx.get", _fake_get)
    with pytest.raises(AuthError) as exc:
        _xai_oauth_discovery()
    assert exc.value.code == "xai_discovery_invalid"


def test_xai_oauth_discovery_validates_authorization_endpoint(monkeypatch):
    """A poisoned ``authorization_endpoint`` is just as dangerous as a
    poisoned ``token_endpoint``: it sends the user's browser (with their
    logged-in xAI session cookies) to attacker infrastructure that can
    phish the consent screen and exchange a stolen authorization code.

    Both endpoints must be validated independently. This test pins the
    parity so nobody can later "optimise" by validating only the token
    endpoint and silently lose authorization-endpoint defense."""
    from hermes_cli.auth import _xai_oauth_discovery

    class _StubGetResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def _fake_get(url, headers=None, timeout=None):
        return _StubGetResponse({
            "authorization_endpoint": "https://evil.example.com/authorize",  # poisoned
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        })

    monkeypatch.setattr("hermes_cli.auth.httpx.get", _fake_get)
    with pytest.raises(AuthError) as exc:
        _xai_oauth_discovery()
    assert exc.value.code == "xai_discovery_invalid"


# ---------------------------------------------------------------------------
# Pool seeding from singleton
# ---------------------------------------------------------------------------


def test_credential_pool_seeds_xai_oauth_from_singleton(tmp_path, monkeypatch):
    """After `hermes model` -> xai-oauth, the singleton holds tokens.  load_pool
    must surface that as a pool entry so `hermes auth list` reflects truth and
    refreshes route through the pool consistently with codex."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=fresh, refresh_token="rt-1")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = load_pool("xai-oauth")
    assert pool.has_credentials()
    entries = pool.entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.access_token == fresh
    assert entry.refresh_token == "rt-1"
    assert entry.source == "loopback_pkce"
    assert entry.base_url == DEFAULT_XAI_OAUTH_BASE_URL


def test_credential_pool_does_not_seed_when_singleton_missing_access_token(tmp_path, monkeypatch):
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "providers": {
            "xai-oauth": {
                "tokens": {"access_token": "", "refresh_token": "rt"},
                "auth_mode": "oauth_pkce",
            }
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = load_pool("xai-oauth")
    assert not pool.has_credentials()


def test_credential_pool_seed_respects_suppression(tmp_path, monkeypatch):
    """`hermes auth remove xai-oauth <N>` for the seeded entry suppresses
    further re-seeding so the removal is stable across load_pool calls."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Suppress the source — mimic `hermes auth remove`.
    from hermes_cli.auth import suppress_credential_source

    suppress_credential_source("xai-oauth", "loopback_pkce")

    pool = load_pool("xai-oauth")
    assert not pool.has_credentials()


def test_auth_remove_xai_oauth_clears_singleton_and_sticks(tmp_path, monkeypatch):
    """End-to-end regression: ``hermes auth remove xai-oauth 1`` for a
    singleton-seeded entry must clear auth.json providers.xai-oauth AND
    suppress further re-seeding — otherwise the next ``load_pool`` call
    silently resurrects the entry from the still-present singleton, making
    the user-facing removal a no-op (the entry reappears on the next
    invocation with no warning).

    The bug pre-fix: there was no RemovalStep registered for
    (xai-oauth, loopback_pkce), so ``find_removal_step`` returned None
    and ``auth_remove_command`` fell through to the "unregistered source —
    nothing to clean up" branch. That branch is correct for ``manual``
    entries (pool-only) but wrong for singleton-seeded loopback_pkce
    entries (auth.json singleton survives the in-memory removal)."""
    from agent.credential_pool import load_pool
    from hermes_cli.auth_commands import auth_remove_command
    from types import SimpleNamespace

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=fresh, refresh_token="rt-1")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Confirm pre-state: pool sees the seeded entry, auth.json has the singleton.
    pool = load_pool("xai-oauth")
    assert pool.has_credentials()
    raw = json.loads((hermes_home / "auth.json").read_text())
    assert "xai-oauth" in raw.get("providers", {})

    # Act: the user runs `hermes auth remove xai-oauth 1`.
    auth_remove_command(SimpleNamespace(provider="xai-oauth", target="1"))

    # Post-state: auth.json singleton must be cleared so a re-seed has
    # nothing to import.
    raw_after = json.loads((hermes_home / "auth.json").read_text())
    assert "xai-oauth" not in raw_after.get("providers", {}), (
        "auth.json providers.xai-oauth must be cleared — otherwise the "
        "next load_pool() reseeds the removed entry from the surviving "
        "singleton, silently undoing the user's removal."
    )

    # And the next load must not reseed the entry from anywhere.
    pool_after = load_pool("xai-oauth")
    assert not pool_after.has_credentials(), (
        "Removal must stick across load_pool() calls — without the "
        "loopback_pkce RemovalStep, the seed function reads the singleton "
        "and rebuilds the entry on every Hermes invocation."
    )


# ---------------------------------------------------------------------------
# Pool sync-back to singleton after refresh
# ---------------------------------------------------------------------------


def test_pool_sync_back_writes_to_singleton(tmp_path, monkeypatch):
    """When the pool refreshes a singleton-seeded xAI entry, the new tokens
    must be written back to providers["xai-oauth"] so that
    resolve_xai_oauth_runtime_credentials() (which reads the singleton)
    doesn't keep using the consumed refresh token."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    expired = _jwt_with_exp(int(time.time()) - 10)
    _setup_hermes_auth(hermes_home, access_token=expired, refresh_token="rt-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    new_access = _jwt_with_exp(int(time.time()) + 3600)

    def _fake_refresh(access_token, refresh_token, **kwargs):
        assert refresh_token == "rt-old"
        return {
            "access_token": new_access,
            "refresh_token": "rt-new",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
            "last_refresh": "2026-05-15T01:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)

    pool = load_pool("xai-oauth")
    selected = pool.select()
    assert selected is not None
    assert selected.access_token == new_access
    assert selected.refresh_token == "rt-new"

    # Singleton must reflect refreshed tokens — otherwise the next process
    # to load credentials would re-seed the consumed refresh token.
    auth_path = hermes_home / "auth.json"
    raw = json.loads(auth_path.read_text())
    state = raw["providers"]["xai-oauth"]
    assert state["tokens"]["access_token"] == new_access
    assert state["tokens"]["refresh_token"] == "rt-new"
    assert state["last_refresh"] == "2026-05-15T01:00:00Z"


# ---------------------------------------------------------------------------
# Runtime provider routing
# ---------------------------------------------------------------------------


def test_runtime_provider_uses_pool_entry_for_xai_oauth(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    runtime = resolve_runtime_provider(requested="xai-oauth")
    assert runtime["provider"] == "xai-oauth"
    assert runtime["api_mode"] == "codex_responses"
    assert runtime["api_key"] == fresh
    assert runtime["base_url"] == DEFAULT_XAI_OAUTH_BASE_URL


def test_runtime_provider_default_base_url_when_pool_entry_missing_url(tmp_path, monkeypatch):
    """Edge case: a pool entry that somehow has an empty base_url should still
    surface the default xAI inference base URL instead of an empty string."""
    from agent.credential_pool import load_pool, AUTH_TYPE_OAUTH, PooledCredential
    import uuid

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    fresh = _jwt_with_exp(int(time.time()) + 3600)
    pool = load_pool("xai-oauth")
    pool.add_entry(
        PooledCredential(
            provider="xai-oauth",
            id=uuid.uuid4().hex[:6],
            label="test",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token=fresh,
            refresh_token="rt",
            base_url="",
        )
    )

    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="xai-oauth")
    assert runtime["provider"] == "xai-oauth"
    assert runtime["api_mode"] == "codex_responses"
    assert runtime["api_key"] == fresh
    assert runtime["base_url"] == DEFAULT_XAI_OAUTH_BASE_URL


# ---------------------------------------------------------------------------
# Token-expiry behavior on the pool path
# ---------------------------------------------------------------------------


def test_pool_entry_needs_refresh_when_jwt_within_skew(tmp_path, monkeypatch):
    """The pool's proactive-refresh gate must trigger when the JWT exp claim
    is within the XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS window — otherwise a
    near-expired token will hit the API and 401 unnecessarily.  Mirrors the
    Codex skew-window behavior."""
    from agent.credential_pool import load_pool, AUTH_TYPE_OAUTH, PooledCredential
    from hermes_cli.auth import XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    import uuid

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Token expires in 30s — well inside the 120s skew window.
    near_expiry = _jwt_with_exp(int(time.time()) + 30)
    pool = load_pool("xai-oauth")
    entry = PooledCredential(
        provider="xai-oauth",
        id=uuid.uuid4().hex[:6],
        label="test",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:xai_pkce",
        access_token=near_expiry,
        refresh_token="rt",
        base_url=DEFAULT_XAI_OAUTH_BASE_URL,
    )
    pool.add_entry(entry)
    assert XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS > 30
    assert pool._entry_needs_refresh(entry) is True


def test_pool_entry_no_refresh_for_fresh_jwt(tmp_path, monkeypatch):
    """A fresh JWT beyond the skew window must NOT trigger proactive refresh."""
    from agent.credential_pool import load_pool, AUTH_TYPE_OAUTH, PooledCredential
    import uuid

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    fresh = _jwt_with_exp(int(time.time()) + 3600)
    pool = load_pool("xai-oauth")
    entry = PooledCredential(
        provider="xai-oauth",
        id=uuid.uuid4().hex[:6],
        label="test",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:xai_pkce",
        access_token=fresh,
        refresh_token="rt",
        base_url=DEFAULT_XAI_OAUTH_BASE_URL,
    )
    pool.add_entry(entry)
    assert pool._entry_needs_refresh(entry) is False


def test_pool_select_proactively_refreshes_expiring_token(tmp_path, monkeypatch):
    """End-to-end: pool.select() with refresh=True on an expiring entry must
    return the refreshed token.  This is the proactive path that runs BEFORE
    the API call — separate from the 401-reactive path."""
    from agent.credential_pool import load_pool, AUTH_TYPE_OAUTH, PooledCredential
    import uuid

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    near_expiry = _jwt_with_exp(int(time.time()) + 30)
    new_access = _jwt_with_exp(int(time.time()) + 3600)

    refresh_calls = {"count": 0}

    def _fake_refresh(access_token, refresh_token, **kwargs):
        refresh_calls["count"] += 1
        assert refresh_token == "rt-old"
        return {
            "access_token": new_access,
            "refresh_token": "rt-new",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
            "last_refresh": "2026-05-15T01:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)

    pool = load_pool("xai-oauth")
    pool.add_entry(
        PooledCredential(
            provider="xai-oauth",
            id=uuid.uuid4().hex[:6],
            label="test",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token=near_expiry,
            refresh_token="rt-old",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        )
    )

    selected = pool.select()
    assert refresh_calls["count"] == 1
    assert selected is not None
    assert selected.access_token == new_access
    assert selected.refresh_token == "rt-new"


def test_pool_try_refresh_current_handles_xai_oauth(tmp_path, monkeypatch):
    """The reactive 401-recovery path uses pool.try_refresh_current().  This
    must work for xai-oauth alongside openai-codex — otherwise mid-call
    expirations get propagated as hard failures instead of being retried with
    fresh tokens."""
    from agent.credential_pool import load_pool, AUTH_TYPE_OAUTH, PooledCredential
    import uuid

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Even a "fresh-looking" token gets force-refreshed via try_refresh_current.
    # We simulate the scenario where the server rejected the token (401)
    # despite client-side expiry math saying it's still valid (e.g. clock
    # skew, server-side revocation, token bound to a session that expired).
    seemingly_fresh = _jwt_with_exp(int(time.time()) + 3600)
    new_access = _jwt_with_exp(int(time.time()) + 7200)

    def _fake_refresh(access_token, refresh_token, **kwargs):
        return {
            "access_token": new_access,
            "refresh_token": "rt-rotated",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
            "last_refresh": "2026-05-15T02:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)

    pool = load_pool("xai-oauth")
    pool.add_entry(
        PooledCredential(
            provider="xai-oauth",
            id=uuid.uuid4().hex[:6],
            label="test",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token=seemingly_fresh,
            refresh_token="rt-old",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        )
    )
    pool.select()
    refreshed = pool.try_refresh_current()
    assert refreshed is not None
    assert refreshed.access_token == new_access
    assert refreshed.refresh_token == "rt-rotated"


def test_pool_refresh_marks_entry_exhausted_on_failure(tmp_path, monkeypatch):
    """When the xAI refresh endpoint rejects the refresh_token (e.g. consumed
    by another process, revoked), the pool must surface the failure cleanly
    rather than silently retaining stale tokens.  This is critical for the
    failover path — _recover_with_credential_pool rotates to the next entry
    only if try_refresh_current returns None."""
    from agent.credential_pool import load_pool, AUTH_TYPE_OAUTH, PooledCredential
    from hermes_cli.auth import AuthError
    import uuid

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _fake_refresh_fail(*args, **kwargs):
        raise AuthError("refresh_token_reused", code="xai_refresh_failed", relogin_required=True)

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh_fail)

    pool = load_pool("xai-oauth")
    seemingly_fresh = _jwt_with_exp(int(time.time()) + 3600)
    pool.add_entry(
        PooledCredential(
            provider="xai-oauth",
            id=uuid.uuid4().hex[:6],
            label="test",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token=seemingly_fresh,
            refresh_token="rt-revoked",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        )
    )
    pool.select()
    refreshed = pool.try_refresh_current()
    # Refresh failure must return None so the caller falls through to
    # credential rotation / friendly error display.
    assert refreshed is None


def test_pool_seeded_entry_sync_back_after_refresh(tmp_path, monkeypatch):
    """When an entry seeded from the singleton (source='loopback_pkce')
    is refreshed by the pool, the new tokens must be written back so a
    fresh process load doesn't re-seed the now-consumed refresh token."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    near_expiry = _jwt_with_exp(int(time.time()) + 30)
    _setup_hermes_auth(hermes_home, access_token=near_expiry, refresh_token="rt-singleton")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    new_access = _jwt_with_exp(int(time.time()) + 3600)

    def _fake_refresh(access_token, refresh_token, **kwargs):
        assert refresh_token == "rt-singleton"
        return {
            "access_token": new_access,
            "refresh_token": "rt-rotated",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
            "last_refresh": "2026-05-15T03:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)

    pool = load_pool("xai-oauth")
    selected = pool.select()
    assert selected is not None
    assert selected.access_token == new_access

    raw = json.loads((hermes_home / "auth.json").read_text())
    tokens = raw["providers"]["xai-oauth"]["tokens"]
    assert tokens["access_token"] == new_access
    assert tokens["refresh_token"] == "rt-rotated"


def test_pool_refresh_adopts_singleton_tokens_when_consumed_elsewhere(tmp_path, monkeypatch):
    """Multi-process race: another Hermes process refreshed the singleton
    (rotating the refresh_token) while this process held a stale in-memory
    pool entry.  ``_refresh_entry`` must adopt the fresher singleton tokens
    BEFORE spending its own (now-consumed) refresh_token, otherwise the
    refresh POST would replay the consumed token and fail with
    ``refresh_token_reused``.

    Mirrors the proactive sync codex/nous already perform for the same
    reason, and is what makes the pool actually safe to share across
    profiles + Hermes processes."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    in_memory_at = _jwt_with_exp(int(time.time()) + 30)  # near-expiry
    _setup_hermes_auth(hermes_home, access_token=in_memory_at, refresh_token="rt-stale")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Load the pool once so the in-memory entry is seeded with rt-stale.
    pool = load_pool("xai-oauth")

    # Now simulate "another process refreshed the tokens" by overwriting
    # the singleton on disk WITHOUT touching this process's pool object.
    other_process_at = _jwt_with_exp(int(time.time()) + 3600)
    raw = json.loads((hermes_home / "auth.json").read_text())
    raw["providers"]["xai-oauth"]["tokens"] = {
        "access_token": other_process_at,
        "refresh_token": "rt-rotated-by-other-process",
        "id_token": "",
        "expires_in": 3600,
        "token_type": "Bearer",
    }
    (hermes_home / "auth.json").write_text(json.dumps(raw))

    refresh_calls = {"refresh_token_seen": None}
    final_at = _jwt_with_exp(int(time.time()) + 7200)

    def _fake_refresh(access_token, refresh_token, **kwargs):
        # The pool MUST have adopted the rotated token from auth.json before
        # POSTing the refresh — otherwise it would replay the stale one.
        refresh_calls["refresh_token_seen"] = refresh_token
        return {
            "access_token": final_at,
            "refresh_token": "rt-final",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
            "last_refresh": "2026-05-15T05:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)

    selected = pool.select()
    assert selected is not None
    assert refresh_calls["refresh_token_seen"] == "rt-rotated-by-other-process"
    assert selected.access_token == final_at


def test_pool_refresh_recovers_when_other_process_already_refreshed(tmp_path, monkeypatch):
    """Variant of the multi-process race where the other process refreshes
    BETWEEN our proactive sync and the HTTP POST.  Our refresh fails with a
    consumed-token error; we must re-check auth.json, find the fresh pair
    (written by the racing process), and adopt it instead of marking the
    entry exhausted."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    in_memory_at = _jwt_with_exp(int(time.time()) + 30)
    _setup_hermes_auth(hermes_home, access_token=in_memory_at, refresh_token="rt-shared")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = load_pool("xai-oauth")

    other_process_at = _jwt_with_exp(int(time.time()) + 3600)

    def _fake_refresh(access_token, refresh_token, **kwargs):
        # Simulate the racing process winning at the auth server right
        # before our POST: by the time we reach this call, auth.json
        # already holds the fresher pair, but we POSTed with rt-shared.
        raw = json.loads((hermes_home / "auth.json").read_text())
        raw["providers"]["xai-oauth"]["tokens"] = {
            "access_token": other_process_at,
            "refresh_token": "rt-rotated",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        (hermes_home / "auth.json").write_text(json.dumps(raw))
        raise AuthError(
            "refresh_token_reused",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)

    selected = pool.select()
    # Even though refresh_xai_oauth_pure raised, the post-failure
    # recovery path should adopt the fresher singleton tokens.
    assert selected is not None
    assert selected.access_token == other_process_at
    assert selected.refresh_token == "rt-rotated"


def test_pool_exhausted_xai_entry_recovers_after_singleton_refresh(tmp_path, monkeypatch):
    """When a singleton-seeded entry is parked as STATUS_EXHAUSTED and the
    user runs ``hermes model`` -> xAI Grok OAuth (or another process
    refreshes), the next ``_available_entries`` pass must adopt the fresh
    auth.json tokens instead of leaving the entry frozen until the
    cooldown elapses.  Mirrors the codex/nous self-heal pattern."""
    from agent.credential_pool import load_pool, STATUS_EXHAUSTED
    from dataclasses import replace

    hermes_home = tmp_path / "hermes"
    stale_at = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=stale_at, refresh_token="rt-stale")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = load_pool("xai-oauth")
    seeded = pool.entries()[0]
    assert seeded.source == "loopback_pkce"

    # Park the seeded entry as exhausted with a far-future cooldown so
    # without resync it would never be selectable.
    exhausted = replace(
        seeded,
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_code=401,
        last_error_reset_at=time.time() + 3600,  # 1h cooldown
    )
    pool._replace_entry(seeded, exhausted)
    pool._persist()
    assert pool.has_credentials()
    assert not pool.has_available()  # cooldown blocks everything

    # Simulate the user re-running `hermes model` -> xAI Grok OAuth: the
    # singleton now has fresh tokens.
    fresh_at = _jwt_with_exp(int(time.time()) + 7200)
    raw = json.loads((hermes_home / "auth.json").read_text())
    raw["providers"]["xai-oauth"]["tokens"] = {
        "access_token": fresh_at,
        "refresh_token": "rt-fresh",
        "id_token": "",
        "expires_in": 3600,
        "token_type": "Bearer",
    }
    (hermes_home / "auth.json").write_text(json.dumps(raw))

    # _available_entries must sync from the singleton, lifting the
    # exhausted state for the seeded entry.
    available = pool._available_entries(clear_expired=True, refresh=False)
    assert len(available) == 1
    assert available[0].access_token == fresh_at
    assert available[0].refresh_token == "rt-fresh"
    assert available[0].last_status != STATUS_EXHAUSTED


def test_pool_manual_xai_entry_not_synced_from_singleton(tmp_path, monkeypatch):
    """Sync from the singleton must apply ONLY to the singleton-seeded
    entry (source='loopback_pkce').  Manually added entries (e.g. via
    ``hermes auth add xai-oauth``) own their own refresh-token lifecycle
    and must not be silently overwritten when the user logs in via
    ``hermes model``."""
    from agent.credential_pool import load_pool, AUTH_TYPE_OAUTH, PooledCredential
    import uuid

    hermes_home = tmp_path / "hermes"
    singleton_at = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=singleton_at, refresh_token="rt-singleton")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = load_pool("xai-oauth")

    manual_at_old = _jwt_with_exp(int(time.time()) + 30)
    pool.add_entry(
        PooledCredential(
            provider="xai-oauth",
            id=uuid.uuid4().hex[:6],
            label="manual",
            auth_type=AUTH_TYPE_OAUTH,
            priority=1,
            source="manual:xai_pkce",
            access_token=manual_at_old,
            refresh_token="rt-manual",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        )
    )
    manual_entry = next(e for e in pool.entries() if e.source == "manual:xai_pkce")
    synced = pool._sync_xai_oauth_entry_from_auth_store(manual_entry)
    # Same object — no sync happened.
    assert synced is manual_entry
    assert synced.access_token == manual_at_old
    assert synced.refresh_token == "rt-manual"


def test_pool_manual_entry_does_not_sync_back_to_singleton(tmp_path, monkeypatch):
    """`hermes auth add xai-oauth` entries (source='manual:xai_pkce') are
    independent credentials and must NOT write to the singleton.  Sync-back
    is restricted to entries seeded from the singleton.  Otherwise adding a
    second pool credential would silently overwrite the user's main login."""
    from agent.credential_pool import load_pool, AUTH_TYPE_OAUTH, PooledCredential
    import uuid

    hermes_home = tmp_path / "hermes"
    # Singleton has its own tokens (separate login).
    singleton_at = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=singleton_at, refresh_token="rt-singleton")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    manual_at_old = _jwt_with_exp(int(time.time()) + 30)
    manual_at_new = _jwt_with_exp(int(time.time()) + 7200)

    def _fake_refresh(access_token, refresh_token, **kwargs):
        assert refresh_token == "rt-manual"
        return {
            "access_token": manual_at_new,
            "refresh_token": "rt-manual-new",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
            "last_refresh": "2026-05-15T04:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)

    pool = load_pool("xai-oauth")
    pool.add_entry(
        PooledCredential(
            provider="xai-oauth",
            id=uuid.uuid4().hex[:6],
            label="manual",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token=manual_at_old,
            refresh_token="rt-manual",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        )
    )
    # Refresh the manual entry — singleton must be left alone.
    manual_entries = [e for e in pool.entries() if e.source == "manual:xai_pkce"]
    assert len(manual_entries) == 1
    pool._refresh_entry(manual_entries[0], force=True)

    raw = json.loads((hermes_home / "auth.json").read_text())
    tokens = raw["providers"]["xai-oauth"]["tokens"]
    # Singleton must be untouched — manual refresh shouldn't leak across.
    assert tokens["access_token"] == singleton_at
    assert tokens["refresh_token"] == "rt-singleton"


# ---------------------------------------------------------------------------
# Auxiliary client routing
# ---------------------------------------------------------------------------


def test_auxiliary_client_routes_xai_oauth_through_responses_api(tmp_path, monkeypatch):
    """Without explicit xai-oauth handling in ``resolve_provider_client``, an
    xai-oauth main provider falls through to the generic ``oauth_external``
    arm and returns ``(None, None)`` — silently re-routing every auxiliary
    task (compression, curator, web extract, session search, ...) to
    whatever Step-2 fallback chain the user has configured (OpenRouter,
    Nous, etc.).  Users on xAI Grok OAuth would then see surprise charges
    on those side providers for side tasks they thought were running on
    their xAI subscription.

    Pin the routing contract: ``resolve_provider_client("xai-oauth", model)``
    must return a non-None client wrapping the xAI Responses API."""
    from agent.auxiliary_client import (
        CodexAuxiliaryClient,
        resolve_provider_client,
    )

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    client, model = resolve_provider_client("xai-oauth", model="grok-4")
    assert client is not None, (
        "xai-oauth must route to a Responses-API client; falling through to "
        "the generic oauth_external branch silently swaps providers for "
        "every auxiliary task."
    )
    assert isinstance(client, CodexAuxiliaryClient)
    assert model == "grok-4"
    # The wrapper preserves base_url + api_key so async wrappers and cache
    # eviction can introspect them.  Pin both to the live xAI runtime.
    assert str(client.base_url).rstrip("/") == DEFAULT_XAI_OAUTH_BASE_URL
    assert client.api_key == fresh


def test_auxiliary_client_xai_oauth_returns_none_when_unauthenticated(tmp_path, monkeypatch):
    """No xAI OAuth tokens in the auth store → ``resolve_provider_client``
    must return ``(None, None)`` so ``_resolve_auto`` falls through to the
    next provider in the chain instead of crashing or constructing a
    misconfigured client."""
    from agent.auxiliary_client import resolve_provider_client

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    client, model = resolve_provider_client("xai-oauth", model="grok-4")
    assert client is None
    assert model is None


def test_auxiliary_client_xai_oauth_requires_explicit_model(tmp_path, monkeypatch):
    """xAI's Responses API has no safe "cheap aux model" default —
    pinning one would silently rot the same way Codex's did.  Callers
    must pass an explicit model (auxiliary.<task>.model in config.yaml)."""
    from agent.auxiliary_client import resolve_provider_client

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    client, model = resolve_provider_client("xai-oauth", model=None)
    assert client is None
    assert model is None


# ---------------------------------------------------------------------------
# active_provider preservation on pool sync-back
# ---------------------------------------------------------------------------


def test_pool_sync_back_preserves_active_provider(tmp_path, monkeypatch):
    """A token-rotation sync-back is a side effect of refresh, not the user
    picking a provider.  ``_save_provider_state`` flips ``active_provider``;
    using it on the sync-back path means every xAI/Codex/Nous refresh in a
    multi-provider setup silently overrides the user's chosen active
    provider (visible to ``hermes auth status``, ``hermes setup``, and the
    ``hermes`` no-arg dispatcher).  Pin the ``set_active=False`` contract so
    no future refactor regresses to the legacy semantic."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    near_expiry = _jwt_with_exp(int(time.time()) + 30)
    _setup_hermes_auth(hermes_home, access_token=near_expiry, refresh_token="rt-xai")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Simulate a multi-provider user whose actual chosen provider is
    # OpenRouter — xai-oauth tokens exist in the singleton but are NOT
    # the active provider.
    raw = json.loads((hermes_home / "auth.json").read_text())
    raw["active_provider"] = "openrouter"
    (hermes_home / "auth.json").write_text(json.dumps(raw))

    new_access = _jwt_with_exp(int(time.time()) + 3600)

    def _fake_refresh(access_token, refresh_token, **kwargs):
        return {
            "access_token": new_access,
            "refresh_token": "rt-rotated",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
            "last_refresh": "2026-05-15T10:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)

    pool = load_pool("xai-oauth")
    selected = pool.select()
    assert selected is not None
    assert selected.access_token == new_access

    # The refresh wrote new tokens back into the singleton — the user's
    # prior ``active_provider`` choice (openrouter) MUST survive.
    raw_after = json.loads((hermes_home / "auth.json").read_text())
    assert raw_after["active_provider"] == "openrouter", (
        "pool sync-back must not flip active_provider; otherwise xAI/Codex/"
        "Nous token rotations silently take over multi-provider users' "
        "auth.json `active_provider` flag."
    )
    # Tokens were actually written so the next process won't replay the
    # consumed refresh_token (preserves the original sync-back fix).
    state = raw_after["providers"]["xai-oauth"]["tokens"]
    assert state["access_token"] == new_access
    assert state["refresh_token"] == "rt-rotated"
