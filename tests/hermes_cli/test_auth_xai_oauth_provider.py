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
    XAI_OAUTH_SCOPE,
    _read_xai_oauth_tokens,
    _save_xai_oauth_tokens,
    _xai_access_token_is_expiring,
    _xai_oauth_poll_device_token,
    _xai_oauth_request_device_code,
    _xai_validate_inference_base_url,
    format_auth_error,
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
    auth_mode: str = "oauth_pkce",
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
        "auth_mode": auth_mode,
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
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
# Device-code flow
# ---------------------------------------------------------------------------


def test_xai_oauth_request_device_code_returns_display_fields():
    response = _StubHTTPResponse(
        200,
        {
            "device_code": "device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://accounts.x.ai/oauth2/device",
            "verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH",
            "expires_in": 1800,
            "interval": 5,
        },
    )
    client = _StubHTTPClient(response)

    payload = _xai_oauth_request_device_code(client)

    assert payload["user_code"] == "ABCD-EFGH"
    method, args, kwargs = client.last_call
    assert method == "post"
    assert args[0] == "https://auth.x.ai/oauth2/device/code"
    assert kwargs["data"]["client_id"] == XAI_OAUTH_CLIENT_ID
    assert kwargs["data"]["scope"] == XAI_OAUTH_SCOPE


def test_xai_oauth_request_device_code_rejects_missing_fields():
    client = _StubHTTPClient(_StubHTTPResponse(200, {"device_code": "d"}))

    with pytest.raises(AuthError) as exc:
        _xai_oauth_request_device_code(client)

    assert exc.value.code == "device_code_invalid"


def test_xai_oauth_poll_device_token_waits_until_authorized(monkeypatch):
    class _SequenceClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                _StubHTTPResponse(
                    400,
                    {
                        "error": "authorization_pending",
                        "error_description": "User has not yet authorized",
                    },
                ),
                _StubHTTPResponse(
                    200,
                    {
                        "access_token": "xai-access",
                        "refresh_token": "xai-refresh",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    },
                ),
            ]

        def post(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.responses.pop(0)

    monkeypatch.setattr("hermes_cli.auth.time.sleep", lambda _: None)
    client = _SequenceClient()

    payload = _xai_oauth_poll_device_token(
        client,
        token_endpoint="https://auth.x.ai/oauth2/token",
        device_code="device-code",
        expires_in=30,
        poll_interval=1,
    )

    assert payload["access_token"] == "xai-access"
    assert len(client.calls) == 2
    assert client.calls[0][1]["data"]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"


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
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    creds = resolve_xai_oauth_runtime_credentials()
    assert creds["provider"] == "xai-oauth"
    assert creds["api_key"] == fresh
    assert creds["base_url"] == DEFAULT_XAI_OAUTH_BASE_URL
    assert creds["source"] == "hermes-auth-store"
    # Display/telemetry label is hardcoded to the only supported flow, even
    # though this fixture persisted a legacy ``oauth_pkce`` auth_mode.
    assert creds["auth_mode"] == "oauth_device_code"


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

    new_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_XAI_BASE_URL", "https://custom.x.ai/v1/")

    creds = resolve_xai_oauth_runtime_credentials()
    assert creds["base_url"] == "https://custom.x.ai/v1"


# ---------------------------------------------------------------------------
# Inference base-URL host guard (xai-oauth bearer leak protection)
#
# The xAI OAuth bearer is a high-value, long-lived SuperGrok credential.
# ``XAI_BASE_URL`` / ``HERMES_XAI_BASE_URL`` are a credential-leak vector
# unless the host is pinned to the xAI origin. These tests cover the
# accept/reject matrix for `_xai_validate_inference_base_url` and confirm
# the runtime resolver falls back to the default on rejection rather than
# leaking the bearer to an attacker-controlled endpoint.
# ---------------------------------------------------------------------------


def test_xai_inference_base_url_accepts_default():
    assert (
        _xai_validate_inference_base_url(
            "https://api.x.ai/v1", fallback=DEFAULT_XAI_OAUTH_BASE_URL,
        )
        == "https://api.x.ai/v1"
    )


def test_xai_inference_base_url_accepts_bare_apex():
    assert (
        _xai_validate_inference_base_url(
            "https://x.ai/v1", fallback=DEFAULT_XAI_OAUTH_BASE_URL,
        )
        == "https://x.ai/v1"
    )


def test_xai_inference_base_url_accepts_subdomain():
    assert (
        _xai_validate_inference_base_url(
            "https://custom.x.ai/v1", fallback=DEFAULT_XAI_OAUTH_BASE_URL,
        )
        == "https://custom.x.ai/v1"
    )


def test_xai_inference_base_url_strips_trailing_slash():
    assert (
        _xai_validate_inference_base_url(
            "https://api.x.ai/v1/", fallback=DEFAULT_XAI_OAUTH_BASE_URL,
        )
        == "https://api.x.ai/v1"
    )


def test_xai_inference_base_url_empty_returns_fallback():
    assert (
        _xai_validate_inference_base_url("", fallback=DEFAULT_XAI_OAUTH_BASE_URL)
        == DEFAULT_XAI_OAUTH_BASE_URL
    )
    assert (
        _xai_validate_inference_base_url("   ", fallback=DEFAULT_XAI_OAUTH_BASE_URL)
        == DEFAULT_XAI_OAUTH_BASE_URL
    )


def test_xai_inference_base_url_rejects_off_origin_host():
    # The headline attack: env var pointing at an attacker-controlled host.
    result = _xai_validate_inference_base_url(
        "https://attacker.example/v1", fallback=DEFAULT_XAI_OAUTH_BASE_URL,
    )
    assert result == DEFAULT_XAI_OAUTH_BASE_URL


def test_xai_inference_base_url_rejects_suffix_lookalike():
    # ``api.x.ai.example`` ends in ``.example``, not ``.x.ai``. urlparse picks
    # the full host as the hostname, and the suffix check uses ``.x.ai`` (with
    # leading dot) so a lookalike like ``apix.ai`` or ``api.x.ai.evil.com``
    # is rejected.
    for hostile in (
        "https://api.x.ai.evil.com/v1",
        "https://apix.ai/v1",
        "https://x.ai.evil.com/v1",
    ):
        assert (
            _xai_validate_inference_base_url(
                hostile, fallback=DEFAULT_XAI_OAUTH_BASE_URL,
            )
            == DEFAULT_XAI_OAUTH_BASE_URL
        ), hostile


def test_xai_inference_base_url_rejects_http():
    # http:// would put the bearer on the wire in cleartext.
    assert (
        _xai_validate_inference_base_url(
            "http://api.x.ai/v1", fallback=DEFAULT_XAI_OAUTH_BASE_URL,
        )
        == DEFAULT_XAI_OAUTH_BASE_URL
    )


def test_xai_inference_base_url_rejects_other_schemes():
    for hostile in (
        "ftp://api.x.ai/v1",
        "file:///etc/passwd",
        "javascript:alert(1)",
    ):
        assert (
            _xai_validate_inference_base_url(
                hostile, fallback=DEFAULT_XAI_OAUTH_BASE_URL,
            )
            == DEFAULT_XAI_OAUTH_BASE_URL
        ), hostile


def test_resolve_xai_runtime_credentials_rejects_off_origin_env_base_url(tmp_path, monkeypatch, caplog):
    # The end-to-end guarantee: if the env var points at an attacker host,
    # the resolver MUST silently fall back to the default rather than ship
    # the OAuth bearer to the attacker.
    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("XAI_BASE_URL", "https://attacker.example/v1")
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)

    with caplog.at_level("WARNING"):
        creds = resolve_xai_oauth_runtime_credentials()
    assert creds["base_url"] == DEFAULT_XAI_OAUTH_BASE_URL
    assert any(
        "attacker.example" in record.getMessage() for record in caplog.records
    ), "Expected a warning identifying the rejected override host."


# ---------------------------------------------------------------------------
# Quarantine: terminal refresh failure clears dead tokens (#28155 sibling)
# ---------------------------------------------------------------------------

_STALE_XAI_OAUTH_STATE = {
    "tokens": {
        "access_token": "dead-access-token",
        "refresh_token": "dead-refresh-token",
        "id_token": "",
        "expires_in": 3600,
        "token_type": "Bearer",
    },
    "discovery": {"token_endpoint": "https://auth.x.ai/oauth2/token"},
    "redirect_uri": "http://127.0.0.1:51827/callback",
    "last_refresh": "2000-01-01T00:00:00Z",
    "auth_mode": "oauth_pkce",
}


def _seed_xai_oauth_state(
    hermes_home: Path, state: dict, *, active_provider: str = "xai-oauth"
) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "active_provider": active_provider,
        "providers": {"xai-oauth": state},
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store, indent=2))


def test_resolve_credentials_quarantines_dead_tokens_on_terminal_refresh_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal refresh failure (relogin_required=True, code=xai_refresh_failed)
    must clear access_token/refresh_token from auth.json and write a
    last_auth_error marker so subsequent calls fail fast without a network retry.
    Mirrors the credential_pool.py quarantine for the singleton/direct resolve path.
    """
    hermes_home = tmp_path / "hermes"
    _seed_xai_oauth_state(hermes_home, dict(_STALE_XAI_OAUTH_STATE), active_provider="nous")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _terminal_refresh(tokens, **kwargs):
        raise AuthError(
            "xAI token refresh failed. Response: invalid_grant",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr("hermes_cli.auth._refresh_xai_oauth_tokens", _terminal_refresh)

    with pytest.raises(AuthError) as exc_info:
        resolve_xai_oauth_runtime_credentials(force_refresh=True)

    assert exc_info.value.code == "xai_refresh_failed"
    assert exc_info.value.relogin_required is True

    raw = json.loads((hermes_home / "auth.json").read_text())
    tokens = raw["providers"]["xai-oauth"]["tokens"]

    # Dead OAuth fields must be cleared.
    assert "access_token" not in tokens
    assert "refresh_token" not in tokens

    # Non-credential metadata must be preserved.
    assert tokens.get("token_type") == "Bearer"

    # Structured diagnostic blob must be written.
    err = raw["providers"]["xai-oauth"].get("last_auth_error")
    assert isinstance(err, dict)
    assert err["provider"] == "xai-oauth"
    assert err["code"] == "xai_refresh_failed"
    assert err["reason"] == "runtime_refresh_failure"
    assert err["relogin_required"] is True
    assert "at" in err

    # Active provider must be unchanged.
    assert raw["active_provider"] == "nous"


def test_resolve_credentials_does_not_quarantine_on_transient_refresh_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient refresh failure (relogin_required=False, e.g. 429 / 5xx) must
    NOT trigger the quarantine path — tokens stay on disk for the next attempt.
    """
    hermes_home = tmp_path / "hermes"
    _seed_xai_oauth_state(hermes_home, dict(_STALE_XAI_OAUTH_STATE))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _transient_refresh(tokens, **kwargs):
        raise AuthError(
            "xAI token refresh failed: connection error",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=False,
        )

    monkeypatch.setattr("hermes_cli.auth._refresh_xai_oauth_tokens", _transient_refresh)

    with pytest.raises(AuthError) as exc_info:
        resolve_xai_oauth_runtime_credentials(force_refresh=True)

    assert exc_info.value.relogin_required is False

    # Tokens must be untouched — no quarantine on transient errors.
    raw = json.loads((hermes_home / "auth.json").read_text())
    tokens = raw["providers"]["xai-oauth"]["tokens"]
    assert tokens["refresh_token"] == "dead-refresh-token"
    assert tokens["access_token"] == "dead-access-token"
    assert "last_auth_error" not in raw["providers"]["xai-oauth"]


# ---------------------------------------------------------------------------
# Auth status surface
# ---------------------------------------------------------------------------


def test_get_xai_oauth_auth_status_logged_in_via_singleton(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    _setup_hermes_auth(hermes_home, access_token=fresh)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    status = get_xai_oauth_auth_status()
    assert status["logged_in"] is True
    assert status["api_key"] == fresh
    # Display/telemetry label is hardcoded to the only supported flow, even
    # though this fixture persisted a legacy ``oauth_pkce`` auth_mode.
    assert status["auth_mode"] == "oauth_device_code"


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


def test_refresh_xai_oauth_pure_403_marked_tier_denied_not_relogin(monkeypatch):
    """403 from xAI's token endpoint is tier/entitlement, not stale tokens.

    Regression test for #26847 — xAI's backend has been seen to 403
    standard SuperGrok subscribers despite the in-app subscription
    being active. Re-running ``hermes model`` won't help in that
    case, so the AuthError must NOT set ``relogin_required=True``,
    and must carry the dedicated ``xai_oauth_tier_denied`` code so
    ``format_auth_error`` doesn't append the misleading re-auth hint.
    """
    response = _StubHTTPResponse(403, {"error": "permission_denied"})
    _patch_httpx_client(monkeypatch, response)
    with pytest.raises(AuthError) as exc:
        refresh_xai_oauth_pure(
            "at", "rt", token_endpoint="https://auth.x.ai/oauth2/token"
        )
    assert exc.value.code == "xai_oauth_tier_denied"
    assert exc.value.relogin_required is False
    message = str(exc.value).lower()
    assert "403" in message
    assert "xai_api_key" in message
    assert "tier" in message


def test_format_auth_error_tier_denied_does_not_suggest_relogin():
    """``xai_oauth_tier_denied`` must not append the re-authenticate hint.

    Regression for #26847: telling a tier-gated user to ``hermes model``
    is actively wrong — re-logging in won't change xAI's allowlist
    decision. The full message (with ``XAI_API_KEY`` fallback) is built
    into the error itself.
    """
    err = AuthError(
        "xAI token refresh failed with HTTP 403. Response: forbidden. "
        "This OAuth account is not authorized for xAI API access — "
        "xAI may be restricting API/OAuth use to specific SuperGrok tiers. "
        "Set ``XAI_API_KEY`` and switch to ``provider: xai``.",
        provider="xai-oauth",
        code="xai_oauth_tier_denied",
        relogin_required=False,
    )
    rendered = format_auth_error(err)
    assert "re-authenticate" not in rendered.lower()
    assert "hermes model" not in rendered.lower()
    assert "XAI_API_KEY" in rendered


def test_refresh_xai_oauth_pure_returns_updated_tokens(monkeypatch):
    new_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    new_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    new_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    refreshes route through the pool consistently with codex.

    Device code is the only supported xAI OAuth flow, so the singleton is
    always surfaced as ``device_code``."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    _setup_hermes_auth(hermes_home, access_token=fresh, refresh_token="rt-1")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = load_pool("xai-oauth")
    assert pool.has_credentials()
    entries = pool.entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.access_token == fresh
    assert entry.refresh_token == "rt-1"
    assert entry.source == "device_code"
    assert entry.base_url == DEFAULT_XAI_OAUTH_BASE_URL


def test_credential_pool_seeds_xai_oauth_device_code_source(tmp_path, monkeypatch):
    """Device-code xAI logins should show a device_code source in auth list."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    _setup_hermes_auth(
        hermes_home,
        access_token=fresh,
        refresh_token="rt-1",
        auth_mode="oauth_device_code",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = load_pool("xai-oauth")
    entry = pool.entries()[0]
    assert entry.source == "device_code"
    assert entry.access_token == fresh


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


def test_credential_pool_device_code_seed_respects_suppression(tmp_path, monkeypatch):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import suppress_credential_source

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    _setup_hermes_auth(
        hermes_home,
        access_token=fresh,
        auth_mode="oauth_device_code",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    suppress_credential_source("xai-oauth", "device_code")

    pool = load_pool("xai-oauth")
    assert not pool.has_credentials()


def test_auth_remove_xai_oauth_clears_singleton_and_sticks(tmp_path, monkeypatch):
    """End-to-end regression: ``hermes auth remove xai-oauth 1`` for a
    singleton-seeded entry must clear auth.json providers.xai-oauth AND
    suppress further re-seeding — otherwise the next ``load_pool`` call
    silently resurrects the entry from the still-present singleton, making
    the user-facing removal a no-op (the entry reappears on the next
    invocation with no warning).

    The bug pre-fix: there was no RemovalStep registered for the
    xai-oauth singleton source, so ``find_removal_step`` returned None
    and ``auth_remove_command`` fell through to the "unregistered source —
    nothing to clean up" branch. That branch is correct for ``manual``
    entries (pool-only) but wrong for singleton-seeded ``device_code``
    entries (auth.json singleton survives the in-memory removal)."""
    from agent.credential_pool import load_pool
    from hermes_cli.auth_commands import auth_remove_command
    from types import SimpleNamespace

    hermes_home = tmp_path / "hermes"
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
        "device_code RemovalStep, the seed function reads the singleton "
        "and rebuilds the entry on every Hermes invocation."
    )


def test_login_xai_oauth_relogin_clears_suppression_and_reseeds(tmp_path, monkeypatch):
    """remove -> ``hermes model`` re-login (``_login_xai_oauth``) must clear the
    ``device_code`` suppression marker so the singleton seed re-creates the
    pool entry.

    Pre-fix: ``auth_remove_command`` set ``["device_code"]`` suppression but
    only ``auth_add_command`` cleared it — the ``hermes model`` re-login path did
    not. So after remove -> re-login the seed kept skipping and ``hermes auth
    list`` showed no xAI entry even though the agent still worked via the
    singleton fallback. The fix calls ``unsuppress_credential_source`` on
    explicit interactive login success.
    """
    from types import SimpleNamespace

    from agent.credential_pool import load_pool
    from hermes_cli.auth import (
        _login_xai_oauth,
        is_source_suppressed,
        suppress_credential_source,
    )

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    # Post-remove state: singleton gone + device_code suppressed, so the
    # seed is gated off and the pool is empty.
    suppress_credential_source("xai-oauth", "device_code")
    assert is_source_suppressed("xai-oauth", "device_code") is True
    assert not load_pool("xai-oauth").has_credentials()

    new_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_device_code_login",
        lambda **kwargs: {
            "tokens": {
                "access_token": new_access,
                "refresh_token": "rt-relogin",
                "id_token": "",
                "token_type": "Bearer",
            },
            "discovery": {"token_endpoint": "https://auth.x.ai/token"},
            "redirect_uri": "",
            "base_url": DEFAULT_XAI_OAUTH_BASE_URL,
            "last_refresh": "2026-06-30T10:00:00Z",
        },
    )
    # Don't mutate a real config file during the test.
    monkeypatch.setattr(
        "hermes_cli.auth._update_config_for_provider",
        lambda *args, **kwargs: "config.toml",
    )

    _login_xai_oauth(
        SimpleNamespace(no_browser=True, timeout=3),
        None,  # pconfig is `del`-eted inside the function
        force_new_login=True,
    )

    # The explicit interactive login cleared the suppression marker...
    assert is_source_suppressed("xai-oauth", "device_code") is False
    # ...so the singleton seed re-creates the canonical pool entry.
    pool = load_pool("xai-oauth")
    assert pool.has_credentials()
    entry = next(e for e in pool.entries() if e.source == "device_code")
    assert entry.access_token == new_access


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

    new_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)

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
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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

    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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

    # Token expires in 30s — well inside the proactive refresh skew window.
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

    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    new_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)

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
    seemingly_fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    seemingly_fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    """When an entry seeded from the singleton (source='device_code')
    is refreshed by the pool, the new tokens must be written back so a
    fresh process load doesn't re-seed the now-consumed refresh token."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    near_expiry = _jwt_with_exp(int(time.time()) + 30)
    _setup_hermes_auth(hermes_home, access_token=near_expiry, refresh_token="rt-singleton")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    new_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)

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
    other_process_at = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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

    other_process_at = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)

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
    stale_at = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
    _setup_hermes_auth(hermes_home, access_token=stale_at, refresh_token="rt-stale")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = load_pool("xai-oauth")
    seeded = pool.entries()[0]
    assert seeded.source == "device_code"

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
    entry (source='device_code').  Manually added entries (e.g. via
    ``hermes auth add xai-oauth``) own their own refresh-token lifecycle
    and must not be silently overwritten when the user logs in via
    ``hermes model``."""
    from agent.credential_pool import load_pool, AUTH_TYPE_OAUTH, PooledCredential
    import uuid

    hermes_home = tmp_path / "hermes"
    singleton_at = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    singleton_at = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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
    fresh = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)
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

    new_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)

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
