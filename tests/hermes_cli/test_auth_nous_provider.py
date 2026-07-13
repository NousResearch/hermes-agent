"""Regression tests for Nous OAuth refresh and inference JWT interactions."""

import base64
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from hermes_cli.auth import AuthError, get_provider_auth_state, resolve_nous_runtime_credentials


# =============================================================================
# _resolve_verify: CA bundle path validation
# =============================================================================


class TestResolveVerifyFallback:
    """Verify _resolve_verify falls back to True when CA bundle path doesn't exist."""

    @pytest.fixture(autouse=True)
    def _pin_platform_to_linux(self, monkeypatch):
        """Pin sys.platform so the macOS certifi fallback doesn't alter the
        generic "default trust" return value asserted by these tests."""
        monkeypatch.setattr("sys.platform", "linux")

    def test_missing_ca_bundle_in_auth_state_falls_back(self):
        from hermes_cli.auth import _resolve_verify

        result = _resolve_verify(auth_state={
            "tls": {"insecure": False, "ca_bundle": "/nonexistent/ca-bundle.pem"},
        })
        assert result is True

    def test_valid_ca_bundle_in_auth_state_is_returned(self, tmp_path, monkeypatch):
        import ssl
        from hermes_cli.auth import _resolve_verify

        ca_file = tmp_path / "ca-bundle.pem"
        ca_file.write_text("fake cert")

        # Avoid loading actual PEM — just verify the return type
        mock_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        monkeypatch.setattr(ssl, "create_default_context", lambda **kw: mock_ctx)

        result = _resolve_verify(auth_state={
            "tls": {"insecure": False, "ca_bundle": str(ca_file)},
        })
        assert isinstance(result, ssl.SSLContext), (
            f"Expected ssl.SSLContext but got {type(result).__name__}: {result!r}"
        )

    def test_missing_ssl_cert_file_env_falls_back(self, monkeypatch):
        from hermes_cli.auth import _resolve_verify

        monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/ssl-cert.pem")
        monkeypatch.delenv("HERMES_CA_BUNDLE", raising=False)
        result = _resolve_verify(auth_state={"tls": {}})
        assert result is True

    def test_missing_hermes_ca_bundle_env_falls_back(self, monkeypatch):
        from hermes_cli.auth import _resolve_verify

        monkeypatch.setenv("HERMES_CA_BUNDLE", "/nonexistent/hermes-ca.pem")
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        result = _resolve_verify(auth_state={"tls": {}})
        assert result is True

    def test_insecure_takes_precedence_over_missing_ca(self):
        from hermes_cli.auth import _resolve_verify

        result = _resolve_verify(
            insecure=True,
            auth_state={"tls": {"ca_bundle": "/nonexistent/ca.pem"}},
        )
        assert result is False

    def test_string_false_in_auth_state_does_not_disable_tls_verify(self):
        import ssl
        from hermes_cli.auth import _resolve_verify

        result = _resolve_verify(auth_state={"tls": {"insecure": "false"}})
        assert result is not False
        assert result is True or isinstance(result, ssl.SSLContext)

    def test_string_true_in_auth_state_disables_tls_verify(self):
        from hermes_cli.auth import _resolve_verify

        result = _resolve_verify(auth_state={"tls": {"insecure": "true"}})
        assert result is False

    def test_no_ca_bundle_returns_true(self, monkeypatch):
        from hermes_cli.auth import _resolve_verify

        monkeypatch.delenv("HERMES_CA_BUNDLE", raising=False)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        result = _resolve_verify(auth_state={"tls": {}})
        assert result is True

    def test_explicit_ca_bundle_param_missing_falls_back(self):
        from hermes_cli.auth import _resolve_verify

        result = _resolve_verify(ca_bundle="/nonexistent/explicit-ca.pem")
        assert result is True

    def test_explicit_ca_bundle_param_valid_is_returned(self, tmp_path, monkeypatch):
        import ssl
        from hermes_cli.auth import _resolve_verify

        ca_file = tmp_path / "explicit-ca.pem"
        ca_file.write_text("fake cert")

        # Avoid loading actual PEM — just verify the return type
        mock_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        monkeypatch.setattr(ssl, "create_default_context", lambda **kw: mock_ctx)

        result = _resolve_verify(ca_bundle=str(ca_file))
        assert isinstance(result, ssl.SSLContext), (
            f"Expected ssl.SSLContext but got {type(result).__name__}: {result!r}"
        )


def _setup_nous_auth(
    hermes_home: Path,
    *,
    access_token: str = "",
    refresh_token: str = "refresh-old",
    scope: str = "inference:invoke",
    expires_at: str = "2026-02-01T00:00:00+00:00",
    expires_in: int = 0,
    agent_key: str | None = None,
    agent_key_expires_at: str | None = None,
) -> None:
    access_token = access_token or _invoke_jwt(seconds=3600, scope=scope)
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "active_provider": "nous",
        "providers": {
            "nous": {
                "portal_base_url": "https://portal.example.com",
                "inference_base_url": "https://inference.example.com/v1",
                "client_id": "hermes-cli",
                "token_type": "Bearer",
                "scope": scope,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "obtained_at": "2026-02-01T00:00:00+00:00",
                "expires_in": expires_in,
                "expires_at": expires_at,
                "agent_key": agent_key,
                "agent_key_id": None,
                "agent_key_expires_at": agent_key_expires_at,
                "agent_key_expires_in": None,
                "agent_key_reused": None,
                "agent_key_obtained_at": None,
            }
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store, indent=2))


def _jwt_with_claims(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


def _future_iso(seconds: int = 3600) -> str:
    return datetime.fromtimestamp(time.time() + seconds, tz=timezone.utc).isoformat()


def _invoke_jwt(*, seconds: int = 3600, scope: object = "inference:invoke") -> str:
    return _jwt_with_claims({
        "sub": "test-user",
        "scope": scope,
        "exp": int(time.time() + seconds),
    })


def test_resolve_nous_runtime_credentials_prefers_invoke_jwt_and_mirrors(
    tmp_path,
    monkeypatch,
):
    import hermes_cli.auth as auth_mod

    hermes_home = tmp_path / "hermes"
    token = _invoke_jwt(seconds=3600)
    _setup_nous_auth(
        hermes_home,
        access_token=token,
        scope=auth_mod.DEFAULT_NOUS_SCOPE,
        expires_at=_future_iso(3600),
        expires_in=3600,
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["api_key"] == token
    assert creds["source"] == auth_mod.NOUS_AUTH_PATH_INVOKE_JWT
    assert creds["auth_path"] == auth_mod.NOUS_AUTH_PATH_INVOKE_JWT

    payload = json.loads((hermes_home / "auth.json").read_text())
    singleton = payload["providers"]["nous"]
    assert singleton["agent_key"] == token
    assert datetime.fromisoformat(singleton["agent_key_expires_at"]).timestamp() > time.time() + 300

    pool_entries = payload["credential_pool"]["nous"]
    assert len(pool_entries) == 1
    assert pool_entries[0]["agent_key"] == token
    assert pool_entries[0]["source"] == auth_mod.NOUS_DEVICE_CODE_SOURCE


def test_resolve_nous_runtime_credentials_env_override_wins_live_not_persisted(
    tmp_path,
    monkeypatch,
    shared_store_env,
):
    """NOUS_INFERENCE_BASE_URL is a LIVE override, not a persisted one.

    The env override wins for the base_url returned to the caller this run,
    but durable auth state (auth.json, the credential pool, the shared
    store) keeps the network-validated URL from the refresh response. This
    keeps an ephemeral dev/staging override from poisoning auth.json after
    the env var is later unset.
    """
    import hermes_cli.auth as auth_mod

    hermes_home = tmp_path / "hermes"
    override_url = "https://ai.wildebeest-newton.ts.net/v1"
    network_url = "https://inference-api.nousresearch.com/v1"
    refreshed_token = _invoke_jwt(seconds=3600)
    _setup_nous_auth(
        hermes_home,
        access_token=_invoke_jwt(seconds=-60),
        refresh_token="refresh-old",
        expires_at=_future_iso(-60),
        expires_in=0,
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", override_url)

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        return {
            "access_token": refreshed_token,
            "refresh_token": "refresh-new",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "inference:invoke",
            "inference_base_url": network_url,
        }

    monkeypatch.setattr("hermes_cli.auth._refresh_access_token", _fake_refresh_access_token)

    creds = auth_mod.resolve_nous_runtime_credentials()

    # The env override wins for the LIVE returned base_url...
    assert creds["base_url"] == override_url

    # ...but it is deliberately NOT persisted: every durable store keeps the
    # network-validated URL, so the ephemeral override can't poison auth.json.
    payload = json.loads((hermes_home / "auth.json").read_text())
    assert payload["providers"]["nous"]["inference_base_url"] == network_url
    assert payload["providers"]["nous"]["inference_base_url"] != override_url
    assert payload["credential_pool"]["nous"][0]["inference_base_url"] == network_url

    shared_payload = json.loads((shared_store_env / "nous_auth.json").read_text())
    assert shared_payload["inference_base_url"] == network_url


def test_resolve_nous_runtime_credentials_invoke_jwt_is_idempotent(
    tmp_path,
    monkeypatch,
):
    import hermes_cli.auth as auth_mod

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    exp = int(time.time() + 3600)
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
    token = _jwt_with_claims({
        "sub": "test-user",
        "scope": auth_mod.DEFAULT_NOUS_SCOPE,
        "exp": exp,
    })
    original_obtained_at = "2026-04-17T22:00:10+00:00"
    auth_store = {
        "version": 1,
        "active_provider": "nous",
        "providers": {
            "nous": {
                "portal_base_url": "https://portal.nousresearch.com",
                "inference_base_url": "https://inference-api.nousresearch.com/v1",
                "client_id": "hermes-cli",
                "token_type": "Bearer",
                "scope": auth_mod.DEFAULT_NOUS_SCOPE,
                "access_token": token,
                "refresh_token": "refresh-token",
                "obtained_at": "2026-02-01T00:00:00+00:00",
                "expires_in": 123,
                "expires_at": expires_at,
                "agent_key": token,
                "agent_key_id": None,
                "agent_key_expires_at": expires_at,
                "agent_key_expires_in": 123,
                "agent_key_reused": False,
                "agent_key_obtained_at": original_obtained_at,
                "tls": {"insecure": False, "ca_bundle": None},
            },
        },
    }
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(json.dumps(auth_store, indent=2))
    before_content = auth_path.read_text()
    before_mtime = auth_path.stat().st_mtime_ns
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _unexpected_shared_write(*args, **kwargs):
        raise AssertionError("unchanged invoke JWT resolution should not sync shared store")

    sync_calls = []

    monkeypatch.setattr(auth_mod, "_write_shared_nous_state", _unexpected_shared_write)
    monkeypatch.setattr(
        auth_mod,
        "_sync_nous_pool_from_auth_store",
        lambda: sync_calls.append(True),
    )

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["api_key"] == token
    assert creds["source"] == auth_mod.NOUS_AUTH_PATH_INVOKE_JWT
    assert auth_path.read_text() == before_content
    assert auth_path.stat().st_mtime_ns == before_mtime
    assert sync_calls == []
    payload = json.loads(auth_path.read_text())
    assert (
        payload["providers"]["nous"]["agent_key_obtained_at"]
        == original_obtained_at
    )


def test_resolve_nous_runtime_credentials_trusts_invoke_jwt_exp_over_stale_metadata(
    tmp_path,
    monkeypatch,
):
    import hermes_cli.auth as auth_mod

    hermes_home = tmp_path / "hermes"
    token = _invoke_jwt(seconds=3600)
    _setup_nous_auth(
        hermes_home,
        access_token=token,
        scope=auth_mod.DEFAULT_NOUS_SCOPE,
        expires_at="2000-01-01T00:00:00+00:00",
        expires_in=0,
        agent_key=token,
        agent_key_expires_at="2000-01-01T00:00:00+00:00",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _unexpected_refresh(*args, **kwargs):
        raise AssertionError("valid invoke JWT should not be refreshed because metadata is stale")

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _unexpected_refresh)

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["api_key"] == token
    assert creds["source"] == auth_mod.NOUS_AUTH_PATH_INVOKE_JWT
    payload = json.loads((hermes_home / "auth.json").read_text())
    singleton = payload["providers"]["nous"]
    assert singleton["agent_key"] == token
    assert datetime.fromisoformat(singleton["expires_at"]).timestamp() > time.time() + 300
    assert datetime.fromisoformat(singleton["agent_key_expires_at"]).timestamp() > time.time() + 300


def test_resolve_nous_runtime_credentials_does_not_apply_agent_key_ttl_to_invoke_jwt(
    tmp_path,
    monkeypatch,
):
    import hermes_cli.auth as auth_mod

    hermes_home = tmp_path / "hermes"
    token = _invoke_jwt(seconds=900)
    _setup_nous_auth(
        hermes_home,
        access_token=token,
        scope=auth_mod.DEFAULT_NOUS_SCOPE,
        expires_at=_future_iso(900),
        expires_in=900,
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["api_key"] == token
    assert creds["source"] == auth_mod.NOUS_AUTH_PATH_INVOKE_JWT
    payload = json.loads((hermes_home / "auth.json").read_text())
    assert payload["providers"]["nous"]["agent_key"] == token
    assert payload["credential_pool"]["nous"][0]["agent_key"] == token


def test_resolve_nous_runtime_credentials_refreshes_legacy_agent_key_to_invoke_jwt(
    tmp_path,
    monkeypatch,
):
    import hermes_cli.auth as auth_mod

    hermes_home = tmp_path / "hermes"
    refreshed_token = _invoke_jwt(seconds=3600)
    _setup_nous_auth(
        hermes_home,
        access_token="legacy-access-token",
        refresh_token="refresh-old",
        scope=auth_mod.DEFAULT_NOUS_SCOPE,
        expires_at=_future_iso(3600),
        expires_in=3600,
        agent_key="legacy-opaque-session-key",
        agent_key_expires_at=_future_iso(3600),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    refresh_calls = []

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        del client, portal_base_url, client_id
        refresh_calls.append(refresh_token)
        return {
            "access_token": refreshed_token,
            "refresh_token": "refresh-new",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": auth_mod.DEFAULT_NOUS_SCOPE,
        }

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _fake_refresh_access_token)

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert refresh_calls == ["refresh-old"]
    assert creds["api_key"] == refreshed_token
    assert creds["source"] == auth_mod.NOUS_AUTH_PATH_INVOKE_JWT
    payload = json.loads((hermes_home / "auth.json").read_text())
    singleton = payload["providers"]["nous"]
    assert singleton["access_token"] == refreshed_token
    assert singleton["refresh_token"] == "refresh-new"
    assert singleton["agent_key"] == refreshed_token
    assert singleton["agent_key_id"] is None
    assert payload["credential_pool"]["nous"][0]["agent_key"] == refreshed_token


def test_resolve_nous_runtime_credentials_reauths_when_invoke_scope_missing(
    tmp_path,
    monkeypatch,
):
    import hermes_cli.auth as auth_mod

    hermes_home = tmp_path / "hermes"
    token = _jwt_with_claims({
        "sub": "test-user",
        "scope": "inference:mint_agent_key",
        "exp": int(time.time() + 3600),
    })
    _setup_nous_auth(
        hermes_home,
        access_token=token,
        refresh_token="",
        scope="inference:mint_agent_key",
        expires_at=_future_iso(3600),
        expires_in=3600,
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        auth_mod.resolve_nous_runtime_credentials()

    assert exc.value.code == "missing_inference_invoke_scope"
    assert exc.value.relogin_required is True
    payload = json.loads((hermes_home / "auth.json").read_text())
    assert payload["providers"]["nous"]["agent_key"] is None
    assert "credential_pool" not in payload or not payload["credential_pool"].get("nous")


def test_nous_device_code_login_does_not_retry_legacy_scope_when_invoke_refused(monkeypatch):
    import hermes_cli.auth as auth_mod

    scopes = []

    def _fake_request_device_code(*, client, portal_base_url, client_id, scope):
        del client, portal_base_url, client_id
        scopes.append(scope)
        request = httpx.Request("POST", "https://portal.example.com/api/oauth/device/code")
        response = httpx.Response(
            400,
            json={
                "error": "invalid_scope",
                "error_description": "unsupported inference:invoke",
            },
            request=request,
        )
        raise httpx.HTTPStatusError("invalid_scope", request=request, response=response)

    monkeypatch.setattr(auth_mod, "_request_device_code", _fake_request_device_code)

    with pytest.raises(httpx.HTTPStatusError):
        auth_mod._nous_device_code_login(
            portal_base_url="https://portal.example.com",
            inference_base_url="https://inference.example.com/v1",
            open_browser=False,
            timeout_seconds=1,
        )

    assert scopes == [auth_mod.DEFAULT_NOUS_SCOPE]


def test_removed_legacy_session_env_var_does_not_change_jwt_auth(tmp_path, monkeypatch):
    import hermes_cli.auth as auth_mod

    hermes_home = tmp_path / "hermes"
    token = _invoke_jwt(seconds=3600)
    _setup_nous_auth(
        hermes_home,
        access_token=token,
        scope=auth_mod.DEFAULT_NOUS_SCOPE,
        expires_at=_future_iso(3600),
        expires_in=3600,
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_AGENT_USE_LEGACY_SESSION_KEYS", "true")

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["api_key"] == token
    payload = json.loads((hermes_home / "auth.json").read_text())
    assert payload["providers"]["nous"]["agent_key"] == token

    requested_scopes = []
    login_token = _invoke_jwt(seconds=3600)

    def _fake_request_device_code(*, client, portal_base_url, client_id, scope):
        del client, portal_base_url, client_id
        requested_scopes.append(scope)
        return {
            "device_code": "device",
            "user_code": "user",
            "verification_uri": "https://portal.example.com/device",
            "verification_uri_complete": "https://portal.example.com/device?code=user",
            "expires_in": 600,
            "interval": 1,
        }

    def _fake_poll_for_token(**kwargs):
        del kwargs
        return {
            "access_token": login_token,
            "refresh_token": "refresh-token",
            "expires_in": 900,
            "scope": auth_mod.DEFAULT_NOUS_SCOPE,
        }

    monkeypatch.setattr(auth_mod, "_request_device_code", _fake_request_device_code)
    monkeypatch.setattr(auth_mod, "_poll_for_token", _fake_poll_for_token)

    result = auth_mod._nous_device_code_login(
        portal_base_url="https://portal.example.com",
        inference_base_url="https://inference.example.com/v1",
        open_browser=False,
        timeout_seconds=1,
    )

    assert requested_scopes == [auth_mod.DEFAULT_NOUS_SCOPE]
    assert result["agent_key"] == login_token


def test_nous_inference_auth_logs_do_not_include_secret_values(
    tmp_path,
    monkeypatch,
    caplog,
):
    import hermes_cli.auth as auth_mod

    hermes_home = tmp_path / "hermes"
    token = _invoke_jwt(seconds=3600)
    refreshed_token = _invoke_jwt(seconds=7200)
    refresh_token = "refresh-secret-token"
    _setup_nous_auth(
        hermes_home,
        access_token=token,
        refresh_token=refresh_token,
        scope=auth_mod.DEFAULT_NOUS_SCOPE,
        expires_at=_future_iso(3600),
        expires_in=3600,
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        del client, portal_base_url, client_id, refresh_token
        return {
            "access_token": refreshed_token,
            "refresh_token": "refresh-new",
            "expires_in": 7200,
            "token_type": "Bearer",
            "scope": auth_mod.DEFAULT_NOUS_SCOPE,
        }

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _fake_refresh_access_token)

    caplog.set_level(logging.INFO, logger="hermes_cli.auth")
    auth_mod.resolve_nous_runtime_credentials(
        force_refresh=True,
    )

    logged = caplog.text
    assert "using NAS invoke JWT" in logged
    assert token not in logged
    assert refreshed_token not in logged
    assert refresh_token not in logged


def test_get_nous_auth_status_checks_credential_pool(tmp_path, monkeypatch):
    """get_nous_auth_status() should find Nous credentials in the pool
    even when the auth store has no Nous provider entry — this is the
    case when login happened via the dashboard device-code flow which
    saves to the pool only.
    """
    from hermes_cli.auth import get_nous_auth_status

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Empty auth store — no Nous provider entry
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Seed the credential pool with a Nous entry
    from agent.credential_pool import PooledCredential, load_pool
    pool = load_pool("nous")
    token = _invoke_jwt(seconds=3600)
    expires_at = _future_iso(3600)
    entry = PooledCredential.from_dict("nous", {
        "access_token": token,
        "refresh_token": "test-refresh-token",
        "portal_base_url": "https://portal.example.com",
        "inference_base_url": "https://inference.example.com/v1",
        "agent_key": token,
        "agent_key_expires_at": expires_at,
        "scope": "inference:invoke",
        "label": "dashboard device_code",
        "auth_type": "oauth",
        "source": "manual:dashboard_device_code",
        "base_url": "https://inference.example.com/v1",
    })
    pool.add_entry(entry)

    status = get_nous_auth_status()
    assert status["logged_in"] is True
    assert "example.com" in str(status.get("portal_base_url", ""))


def test_get_nous_auth_status_pool_opaque_key_is_not_inference_credential(tmp_path, monkeypatch):
    from hermes_cli.auth import get_nous_auth_status, invalidate_nous_auth_status_cache

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    invalidate_nous_auth_status_cache()

    from agent.credential_pool import PooledCredential, load_pool
    pool = load_pool("nous")
    entry = PooledCredential.from_dict("nous", {
        "access_token": "",
        "agent_key": "opaque-agent-key",
        "agent_key_expires_at": "2099-01-01T00:00:00+00:00",
        "label": "manual opaque key",
        "auth_type": "api_key",
        "source": "manual",
        "base_url": "https://inference.example.com/v1",
        "inference_base_url": "https://inference.example.com/v1",
    })
    pool.add_entry(entry)

    status = get_nous_auth_status()

    assert status["logged_in"] is False
    assert status["inference_credential_present"] is False
    assert status["credential_source"] is None
    assert status.get("access_token") is None
    assert status.get("portal_base_url") is None
    assert status.get("inference_base_url") is None
    invalidate_nous_auth_status_cache()


def test_get_nous_auth_status_auth_store_fallback(tmp_path, monkeypatch):
    """get_nous_auth_status() falls back to auth store when credential
    pool is empty.
    """
    from hermes_cli.auth import get_nous_auth_status

    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(hermes_home, access_token="at-123")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        lambda **kwargs: {
            "base_url": "https://inference.example.com/v1",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "key_id": "key-1",
            "source": "cache",
        },
    )

    status = get_nous_auth_status()
    assert status["logged_in"] is True
    assert status["portal_base_url"] == "https://portal.example.com"


def test_get_nous_auth_status_prefers_runtime_auth_store_over_stale_pool(tmp_path, monkeypatch):
    from hermes_cli.auth import get_nous_auth_status
    from agent.credential_pool import PooledCredential, load_pool

    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(hermes_home, access_token="at-fresh")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = load_pool("nous")
    stale = PooledCredential.from_dict("nous", {
        "access_token": "at-stale",
        "refresh_token": "rt-stale",
        "portal_base_url": "https://portal.stale.example.com",
        "inference_base_url": "https://inference.stale.example.com/v1",
        "agent_key": "agent-stale",
        "agent_key_expires_at": "2020-01-01T00:00:00+00:00",
        "expires_at": "2020-01-01T00:00:00+00:00",
        "label": "dashboard device_code",
        "auth_type": "oauth",
        "source": "manual:dashboard_device_code",
        "base_url": "https://inference.stale.example.com/v1",
        "priority": 0,
    })
    pool.add_entry(stale)

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        lambda **kwargs: {
            "base_url": "https://inference.example.com/v1",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "key_id": "key-fresh",
            "source": "portal",
        },
    )

    status = get_nous_auth_status()
    assert status["logged_in"] is True
    assert status["portal_base_url"] == "https://portal.example.com"
    assert status["inference_base_url"] == "https://inference.example.com/v1"
    assert status["source"] == "runtime:portal"


def test_get_nous_auth_status_reports_revoked_refresh_session(tmp_path, monkeypatch):
    from hermes_cli.auth import get_nous_auth_status

    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(hermes_home, access_token="at-123")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _boom(**kwargs):
        raise AuthError("Refresh session has been revoked", provider="nous", relogin_required=True)

    monkeypatch.setattr("hermes_cli.auth.resolve_nous_runtime_credentials", _boom)

    status = get_nous_auth_status()
    assert status["logged_in"] is False
    assert status["relogin_required"] is True
    assert "revoked" in status["error"].lower()
    assert status["portal_base_url"] == "https://portal.example.com"


def test_get_nous_auth_status_empty_returns_not_logged_in(tmp_path, monkeypatch):
    """get_nous_auth_status() returns logged_in=False when both pool
    and auth store are empty.
    """
    from hermes_cli.auth import get_nous_auth_status

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    status = get_nous_auth_status()
    assert status["logged_in"] is False


def test_refresh_token_persisted_when_refreshed_jwt_lacks_invoke_scope(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(
        hermes_home,
        access_token="access-old",
        refresh_token="refresh-old",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    refresh_calls = []
    bad_jwt = _jwt_with_claims({
        "sub": "test-user",
        "scope": "profile",
        "exp": int(time.time() + 3600),
    })
    good_jwt = _invoke_jwt(seconds=3600)

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        refresh_calls.append(refresh_token)
        if len(refresh_calls) == 1:
            token = bad_jwt
        else:
            token = good_jwt
        return {
            "access_token": token,
            "refresh_token": f"refresh-{len(refresh_calls)}",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "profile" if len(refresh_calls) == 1 else "inference:invoke",
        }

    monkeypatch.setattr("hermes_cli.auth._refresh_access_token", _fake_refresh_access_token)

    with pytest.raises(AuthError) as exc:
        resolve_nous_runtime_credentials()
    assert exc.value.code == "missing_inference_invoke_scope"

    state_after_failure = get_provider_auth_state("nous")
    assert state_after_failure is not None
    assert state_after_failure["refresh_token"] == "refresh-1"
    assert state_after_failure["access_token"] == bad_jwt

    creds = resolve_nous_runtime_credentials()
    assert creds["api_key"] == good_jwt
    assert refresh_calls == ["refresh-old", "refresh-1"]


def test_refresh_token_persisted_when_refreshed_token_is_not_jwt(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(
        hermes_home,
        access_token="access-old",
        refresh_token="refresh-old",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        return {
            "access_token": "access-1",
            "refresh_token": "refresh-1",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

    monkeypatch.setattr("hermes_cli.auth._refresh_access_token", _fake_refresh_access_token)

    with pytest.raises(AuthError) as exc:
        resolve_nous_runtime_credentials()
    assert exc.value.code == "access_token_not_jwt"

    state_after_failure = get_provider_auth_state("nous")
    assert state_after_failure is not None
    assert state_after_failure["refresh_token"] == "refresh-1"
    assert state_after_failure["access_token"] == "access-1"


def test_terminal_refresh_failure_quarantines_tokens(
    tmp_path, monkeypatch, shared_store_env,
):
    """A revoked/invalid Nous refresh token must not be replayed forever."""
    from hermes_cli import auth as auth_mod

    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(
        hermes_home,
        access_token="access-old",
        refresh_token="refresh-old",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool

    assert load_pool("nous").select() is not None

    shared_state = _full_state_fixture()
    shared_state["access_token"] = "access-old"
    shared_state["refresh_token"] = "refresh-old"
    shared_state["expires_at"] = "2026-02-01T00:00:00+00:00"
    auth_mod._write_shared_nous_state(shared_state)

    refresh_calls: list[str] = []

    def _terminal_refresh_failure(*, client, portal_base_url, client_id, refresh_token):
        refresh_calls.append(refresh_token)
        raise AuthError(
            "Refresh session has been revoked",
            provider="nous",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _terminal_refresh_failure)

    with pytest.raises(AuthError, match="Refresh session has been revoked"):
        auth_mod.resolve_nous_runtime_credentials()

    state_after_failure = auth_mod.get_provider_auth_state("nous")
    assert state_after_failure is not None
    assert not state_after_failure.get("refresh_token")
    assert not state_after_failure.get("access_token")
    assert not state_after_failure.get("agent_key")
    assert state_after_failure["last_auth_error"]["code"] == "invalid_grant"
    assert auth_mod._read_shared_nous_state() is None
    payload = json.loads((hermes_home / "auth.json").read_text())
    assert payload.get("credential_pool", {}).get("nous") == []

    with pytest.raises(AuthError, match="No access token found"):
        auth_mod.resolve_nous_runtime_credentials()

    assert refresh_calls == ["refresh-old"]


def test_managed_access_token_refresh_failure_quarantines_tokens(
    tmp_path, monkeypatch, shared_store_env,
):
    from hermes_cli import auth as auth_mod

    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(hermes_home, refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool

    assert load_pool("nous").select() is not None

    refresh_calls: list[str] = []

    def _terminal_refresh_failure(*, client, portal_base_url, client_id, refresh_token):
        refresh_calls.append(refresh_token)
        raise AuthError(
            "Invalid refresh token",
            provider="nous",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _terminal_refresh_failure)

    with pytest.raises(AuthError, match="Invalid refresh token"):
        auth_mod.resolve_nous_access_token()

    state_after_failure = auth_mod.get_provider_auth_state("nous")
    assert state_after_failure is not None
    assert not state_after_failure.get("refresh_token")
    assert not state_after_failure.get("access_token")
    assert state_after_failure["last_auth_error"]["message"] == "Invalid refresh token"
    payload = json.loads((hermes_home / "auth.json").read_text())
    assert payload.get("credential_pool", {}).get("nous") == []

    with pytest.raises(AuthError, match="No access token found"):
        auth_mod.resolve_nous_access_token()

    assert refresh_calls == ["refresh-old"]


def test_unusable_access_token_refresh_uses_latest_rotated_refresh_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(
        hermes_home,
        access_token="access-old",
        refresh_token="refresh-old",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    refresh_calls = []
    good_jwt = _invoke_jwt(seconds=3600)

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        refresh_calls.append(refresh_token)
        token = "access-still-not-jwt" if len(refresh_calls) == 1 else good_jwt
        return {
            "access_token": token,
            "refresh_token": f"refresh-{len(refresh_calls)}",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "inference:invoke",
        }

    monkeypatch.setattr("hermes_cli.auth._refresh_access_token", _fake_refresh_access_token)

    with pytest.raises(AuthError) as exc:
        resolve_nous_runtime_credentials()
    assert exc.value.code == "access_token_not_jwt"
    creds = resolve_nous_runtime_credentials()
    assert creds["api_key"] == good_jwt
    assert refresh_calls == ["refresh-old", "refresh-1"]


# =============================================================================
# _login_nous: "Skip (keep current)" must preserve prior provider + model
# =============================================================================


class TestLoginNousSkipKeepsCurrent:
    """When a user runs `hermes model` → Nous Portal → Skip (keep current) after
    a successful OAuth login, the prior provider and model MUST be preserved.

    Regression: previously, _update_config_for_provider was called
    unconditionally after login, which flipped model.provider to "nous" while
    keeping the old model.default (e.g. anthropic/claude-opus-4.6 from
    OpenRouter), leaving the user with a mismatched provider/model pair.
    """

    def _setup_home_with_openrouter(self, tmp_path, monkeypatch):
        import yaml
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "model": {
                "provider": "openrouter",
                "default": "anthropic/claude-opus-4.6",
            },
        }, sort_keys=False))

        auth_path = hermes_home / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "active_provider": "openrouter",
            "providers": {"openrouter": {"api_key": "sk-or-fake"}},
        }))
        return hermes_home, config_path, auth_path

    def _patch_login_internals(self, monkeypatch, *, prompt_returns):
        """Patch OAuth + model-list + prompt so _login_nous doesn't hit network."""
        import hermes_cli.auth as auth_mod
        import hermes_cli.models as models_mod
        import hermes_cli.nous_subscription as ns

        fake_auth_state = {
            "access_token": "fake-nous-token",
            "agent_key": "fake-agent-key",
            "inference_base_url": "https://inference-api.nousresearch.com",
            "portal_base_url": "https://portal.nousresearch.com",
            "refresh_token": "fake-refresh",
            "token_expires_at": 9999999999,
        }
        monkeypatch.setattr(
            auth_mod, "_nous_device_code_login",
            lambda **kwargs: dict(fake_auth_state),
        )
        monkeypatch.setattr(
            auth_mod, "_prompt_model_selection",
            lambda *a, **kw: prompt_returns,
        )
        monkeypatch.setattr(models_mod, "get_pricing_for_provider", lambda p: {})
        free_tier_calls = []

        def _check_nous_free_tier(**kwargs):
            free_tier_calls.append(kwargs)
            return None

        monkeypatch.setattr(models_mod, "check_nous_free_tier", _check_nous_free_tier)
        monkeypatch.setattr(
            models_mod, "partition_nous_models_by_tier",
            lambda ids, p, free_tier=False: (ids, []),
        )
        monkeypatch.setattr(ns, "prompt_enable_tool_gateway", lambda cfg: None)
        return free_tier_calls

    def test_skip_keep_current_preserves_provider_and_model(self, tmp_path, monkeypatch):
        """User picks Skip → config.yaml untouched, Nous creds still saved."""
        import argparse
        import yaml
        from hermes_cli.auth import PROVIDER_REGISTRY, _login_nous

        hermes_home, config_path, auth_path = self._setup_home_with_openrouter(
            tmp_path, monkeypatch,
        )
        self._patch_login_internals(monkeypatch, prompt_returns=None)

        args = argparse.Namespace(
            portal_url=None, inference_url=None, client_id=None, scope=None,
            no_browser=True, timeout=15.0, ca_bundle=None, insecure=False,
        )
        _login_nous(args, PROVIDER_REGISTRY["nous"])

        # config.yaml model section must be unchanged
        cfg_after = yaml.safe_load(config_path.read_text())
        assert cfg_after["model"]["provider"] == "openrouter"
        assert cfg_after["model"]["default"] == "anthropic/claude-opus-4.6"
        assert "base_url" not in cfg_after["model"]

        # auth.json: active_provider restored to openrouter, but Nous creds saved
        auth_after = json.loads(auth_path.read_text())
        assert auth_after["active_provider"] == "openrouter"
        assert "nous" in auth_after["providers"]
        assert auth_after["providers"]["nous"]["access_token"] == "fake-nous-token"
        # Existing openrouter creds still intact
        assert auth_after["providers"]["openrouter"]["api_key"] == "sk-or-fake"

    def test_picking_model_switches_to_nous(self, tmp_path, monkeypatch):
        """User picks a Nous model → provider flips to nous with that model."""
        import argparse
        import yaml
        from hermes_cli.auth import PROVIDER_REGISTRY, _login_nous

        hermes_home, config_path, auth_path = self._setup_home_with_openrouter(
            tmp_path, monkeypatch,
        )
        free_tier_calls = self._patch_login_internals(
            monkeypatch, prompt_returns="xiaomi/mimo-v2-pro",
        )

        args = argparse.Namespace(
            portal_url=None, inference_url=None, client_id=None, scope=None,
            no_browser=True, timeout=15.0, ca_bundle=None, insecure=False,
        )
        _login_nous(args, PROVIDER_REGISTRY["nous"])

        cfg_after = yaml.safe_load(config_path.read_text())
        assert cfg_after["model"]["provider"] == "nous"
        assert cfg_after["model"]["default"] == "xiaomi/mimo-v2-pro"
        assert free_tier_calls == [{"force_fresh": True}]

        auth_after = json.loads(auth_path.read_text())
        assert auth_after["active_provider"] == "nous"

    def test_skip_with_no_prior_active_provider_clears_it(self, tmp_path, monkeypatch):
        """Fresh install (no prior active_provider) → Skip clears active_provider
        instead of leaving it as nous."""
        import argparse
        import yaml
        from hermes_cli.auth import PROVIDER_REGISTRY, _login_nous

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({"model": {}}, sort_keys=False))

        # No auth.json yet — simulates first-run before any OAuth
        self._patch_login_internals(monkeypatch, prompt_returns=None)

        args = argparse.Namespace(
            portal_url=None, inference_url=None, client_id=None, scope=None,
            no_browser=True, timeout=15.0, ca_bundle=None, insecure=False,
        )
        _login_nous(args, PROVIDER_REGISTRY["nous"])

        auth_path = hermes_home / "auth.json"
        auth_after = json.loads(auth_path.read_text())
        # active_provider should NOT be set to "nous" after Skip
        assert auth_after.get("active_provider") in {None, ""}
        # But Nous creds are still saved
        assert "nous" in auth_after.get("providers", {})


# =============================================================================
# persist_nous_credentials: shared helper for CLI + web dashboard login paths
# =============================================================================


def _full_state_fixture() -> dict:
    """Shape of the dict returned by _nous_device_code_login /
    refresh_nous_oauth_from_state. Used as helper input."""
    token = _invoke_jwt(seconds=3600)
    expires_at = _future_iso(3600)
    return {
        "portal_base_url": "https://portal.example.com",
        "inference_base_url": "https://inference.example.com/v1",
        "client_id": "hermes-cli",
        "scope": "inference:invoke",
        "token_type": "Bearer",
        "access_token": token,
        "refresh_token": "refresh-tok",
        "obtained_at": "2026-04-17T22:00:00+00:00",
        "expires_at": expires_at,
        "expires_in": 3600,
        "agent_key": token,
        "agent_key_id": None,
        "agent_key_expires_at": expires_at,
        "agent_key_expires_in": 3600,
        "agent_key_reused": False,
        "agent_key_obtained_at": "2026-04-17T22:00:10+00:00",
        "tls": {"insecure": False, "ca_bundle": None},
    }


def test_persist_nous_credentials_writes_both_pool_and_providers(tmp_path, monkeypatch):
    """Helper must populate BOTH credential_pool.nous AND providers.nous.

    Regression guard: before this helper existed, `hermes auth add nous`
    wrote only the pool. After the Nous agent_key's 24h TTL expired, the
    401-recovery path in run_agent.py called resolve_nous_runtime_credentials
    which reads providers.nous, found it empty, raised AuthError, and the
    agent failed with "Non-retryable client error". Both stores must stay
    in sync at write time.
    """
    from hermes_cli.auth import persist_nous_credentials, NOUS_DEVICE_CODE_SOURCE

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    state = _full_state_fixture()
    entry = persist_nous_credentials(state)

    assert entry is not None
    assert entry.provider == "nous"
    assert entry.source == NOUS_DEVICE_CODE_SOURCE

    payload = json.loads((hermes_home / "auth.json").read_text())

    # providers.nous populated with the full state (new behaviour)
    singleton = payload["providers"]["nous"]
    assert singleton["access_token"] == state["access_token"]
    assert singleton["refresh_token"] == "refresh-tok"
    assert singleton["agent_key"] == state["agent_key"]
    assert singleton["agent_key_expires_at"] == state["agent_key_expires_at"]

    # credential_pool.nous has exactly one canonical device_code entry
    pool_entries = payload["credential_pool"]["nous"]
    assert len(pool_entries) == 1, pool_entries
    pool_entry = pool_entries[0]
    assert pool_entry["source"] == NOUS_DEVICE_CODE_SOURCE
    assert pool_entry["agent_key"] == state["agent_key"]
    assert pool_entry["inference_base_url"] == "https://inference.example.com/v1"


def test_persist_nous_credentials_allows_recovery_from_401(tmp_path, monkeypatch):
    """End-to-end: after persisting via the helper, resolve_nous_runtime_credentials
    must succeed (not raise "Hermes is not logged into Nous Portal").

    This is the exact path that run_agent.py's `_try_refresh_nous_client_credentials`
    calls after a Nous 401 — before the fix it would raise AuthError because
    providers.nous was empty.
    """
    from hermes_cli.auth import (
        persist_nous_credentials,
        resolve_nous_runtime_credentials,
    )

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    persist_nous_credentials(_full_state_fixture())
    new_jwt = _invoke_jwt(seconds=3600)

    # Stub the network-touching steps so we don't actually contact the
    # portal — the point of this test is that state lookup succeeds and
    # doesn't raise "Hermes is not logged into Nous Portal".
    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        return {
            "access_token": new_jwt,
            "refresh_token": "refresh-new",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "inference:invoke",
        }

    monkeypatch.setattr("hermes_cli.auth._refresh_access_token", _fake_refresh_access_token)

    creds = resolve_nous_runtime_credentials(
        force_refresh=True,
    )
    assert creds["api_key"] == new_jwt


def test_persist_nous_credentials_idempotent_no_duplicate_pool_entries(tmp_path, monkeypatch):
    """Re-running persist must upsert — not accumulate duplicate device_code rows.

    Regression guard for the review comment on PR #11858: before normalisation,
    the helper wrote `manual:device_code` while `_seed_from_singletons` wrote
    `device_code`, so the pool grew a second duplicate entry on every
    ``load_pool()``. The helper now writes providers.nous and lets seeding
    materialise the pool entry under the canonical ``device_code`` source, so
    two persists still leave the pool with exactly one row.
    """
    from hermes_cli.auth import persist_nous_credentials, NOUS_DEVICE_CODE_SOURCE

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    first = _full_state_fixture()
    persist_nous_credentials(first)

    second = _full_state_fixture()
    second_token = _invoke_jwt(seconds=7200)
    second["access_token"] = second_token
    second["agent_key"] = second_token
    second["agent_key_expires_at"] = _future_iso(7200)
    persist_nous_credentials(second)

    payload = json.loads((hermes_home / "auth.json").read_text())

    # providers.nous reflects the latest write (singleton semantics)
    assert payload["providers"]["nous"]["access_token"] == second_token
    assert payload["providers"]["nous"]["agent_key"] == second_token

    # credential_pool.nous has exactly one entry, carrying the latest agent_key
    pool_entries = payload["credential_pool"]["nous"]
    assert len(pool_entries) == 1, pool_entries
    assert pool_entries[0]["source"] == NOUS_DEVICE_CODE_SOURCE
    assert pool_entries[0]["agent_key"] == second_token
    # And no stray `manual:device_code` / `manual:dashboard_device_code` rows
    assert not any(
        e["source"].startswith("manual:") for e in pool_entries
    )


def test_persist_nous_credentials_reloads_pool_after_singleton_write(tmp_path, monkeypatch):
    """The entry returned by the helper must come from a fresh ``load_pool`` so
    callers observe the canonical seeded state, including any legacy entries
    that ``_seed_from_singletons`` pruned or upserted.
    """
    from hermes_cli.auth import persist_nous_credentials, NOUS_DEVICE_CODE_SOURCE

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    state = _full_state_fixture()
    entry = persist_nous_credentials(state)
    assert entry is not None
    assert entry.source == NOUS_DEVICE_CODE_SOURCE
    # Label derived by _seed_from_singletons via label_from_token; we don't
    # assert its exact value, just that the helper returned a real entry.
    assert entry.access_token == state["access_token"]
    assert entry.agent_key == state["agent_key"]


def test_persist_nous_credentials_embeds_custom_label(tmp_path, monkeypatch):
    """User-supplied ``--label`` round-trips through providers.nous and the pool.

    Previously `hermes auth add nous --type oauth --label <name>` silently
    dropped the label because persist_nous_credentials() ignored it and
    _seed_from_singletons always auto-derived via label_from_token().  The
    fix stashes the label inside providers.nous so seeding prefers it.
    """
    from hermes_cli.auth import persist_nous_credentials, NOUS_DEVICE_CODE_SOURCE

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    entry = persist_nous_credentials(_full_state_fixture(), label="my-personal")
    assert entry is not None
    assert entry.source == NOUS_DEVICE_CODE_SOURCE
    assert entry.label == "my-personal"

    # providers.nous carries the label so re-seeding on the next load_pool
    # doesn't overwrite it with the auto-derived fingerprint.
    payload = json.loads((hermes_home / "auth.json").read_text())
    assert payload["providers"]["nous"]["label"] == "my-personal"


def test_persist_nous_credentials_custom_label_survives_reseed(tmp_path, monkeypatch):
    """Reopening the pool (which re-runs _seed_from_singletons) must keep the
    user-chosen label instead of clobbering it with label_from_token output.
    """
    from hermes_cli.auth import persist_nous_credentials
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    persist_nous_credentials(_full_state_fixture(), label="work-acct")

    # Second load_pool triggers _seed_from_singletons again.  Without the
    # fix, this call overwrote the label with label_from_token(access_token).
    pool = load_pool("nous")
    entries = pool.entries()
    assert len(entries) == 1
    assert entries[0].label == "work-acct"


def test_persist_nous_credentials_no_label_uses_auto_derived(tmp_path, monkeypatch):
    """When the caller doesn't pass ``label``, the auto-derived fingerprint
    is used (unchanged default behaviour — regression guard).
    """
    from hermes_cli.auth import persist_nous_credentials

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    entry = persist_nous_credentials(_full_state_fixture())
    assert entry is not None
    # label_from_token derives from the access_token; exact value depends on
    # the fingerprinter but it must not be empty and must not equal an
    # arbitrary user string we never passed.
    assert entry.label
    assert entry.label != "my-personal"

    # No "label" key embedded in providers.nous when the caller didn't supply one.
    payload = json.loads((hermes_home / "auth.json").read_text())
    assert "label" not in payload["providers"]["nous"]


def test_refresh_token_reuse_detection_surfaces_actionable_message():
    """Regression for #15099.

    When the Nous Portal server returns ``invalid_grant`` with
    ``error_description`` containing "reuse detected", Hermes must surface an
    actionable message explaining that an external process consumed the
    refresh token.  The default opaque "Refresh token reuse detected; please
    re-authenticate" string led users to report this as a Hermes persistence
    bug when the true cause is external RT consumption (monitoring scripts,
    custom self-heal hooks).
    """
    from hermes_cli.auth import _refresh_access_token

    class _FakeResponse:
        status_code = 400

        def json(self):
            return {
                "error": "invalid_grant",
                "error_description": "Refresh token reuse detected; please re-authenticate",
            }

    class _FakeClient:
        def post(self, *args, **kwargs):
            return _FakeResponse()

    with pytest.raises(AuthError) as exc_info:
        _refresh_access_token(
            client=_FakeClient(),
            portal_base_url="https://portal.nousresearch.com",
            client_id="hermes-cli",
            refresh_token="rt_consumed_elsewhere",
        )

    message = str(exc_info.value)
    assert "refresh-token reuse" in message.lower() or "refresh token reuse" in message.lower()
    # The message must mention the external-process cause and give next steps.
    assert "external process" in message.lower() or "monitoring script" in message.lower()
    assert "hermes auth add nous" in message.lower()
    # Must still be classified as invalid_grant + relogin_required.
    assert exc_info.value.code == "invalid_grant"
    assert exc_info.value.relogin_required is True


def test_refresh_token_reuse_error_code_is_terminal():
    """Nous may return refresh_token_reused as the OAuth error code itself."""
    from hermes_cli import auth as auth_mod

    class _FakeResponse:
        status_code = 400

        def json(self):
            return {
                "error": "refresh_token_reused",
                "error_description": "Refresh token reuse detected",
            }

    class _FakeClient:
        def post(self, *args, **kwargs):
            return _FakeResponse()

    with pytest.raises(AuthError) as exc_info:
        auth_mod._refresh_access_token(
            client=_FakeClient(),
            portal_base_url="https://portal.nousresearch.com",
            client_id="hermes-cli",
            refresh_token="rt_consumed_elsewhere",
        )

    assert exc_info.value.code == "refresh_token_reused"
    assert exc_info.value.relogin_required is True
    assert auth_mod._is_terminal_nous_refresh_error(exc_info.value) is True


def test_refresh_token_exchange_sends_refresh_token_header():
    """Nous refresh tokens must be sent in a header so sandbox proxies can
    substitute placeholder credentials without parsing form bodies.
    """
    from hermes_cli.auth import _refresh_access_token

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"access_token": "access-2", "refresh_token": "refresh-2"}

    class _FakeClient:
        def __init__(self):
            self.kwargs = None

        def post(self, *args, **kwargs):
            del args
            self.kwargs = kwargs
            return _FakeResponse()

    client = _FakeClient()

    payload = _refresh_access_token(
        client=client,
        portal_base_url="https://portal.nousresearch.com",
        client_id="hermes-cli",
        refresh_token="refresh-1",
    )

    assert payload["access_token"] == "access-2"
    assert payload["refresh_token"] == "refresh-2"
    assert client.kwargs is not None
    assert client.kwargs["headers"]["x-nous-refresh-token"] == "refresh-1"
    assert client.kwargs["data"] == {
        "grant_type": "refresh_token",
        "client_id": "hermes-cli",
    }


def test_refresh_non_reuse_error_keeps_original_description():
    """Non-reuse invalid_grant errors must keep their original description untouched.

    Only the "reuse detected" signature should trigger the actionable message;
    generic ``invalid_grant: Refresh session has been revoked`` (the
    downstream consequence) keeps its original text so we don't overwrite
    useful server context for unrelated failure modes.
    """
    from hermes_cli.auth import _refresh_access_token

    class _FakeResponse:
        status_code = 400

        def json(self):
            return {
                "error": "invalid_grant",
                "error_description": "Refresh session has been revoked",
            }

    class _FakeClient:
        def post(self, *args, **kwargs):
            return _FakeResponse()

    with pytest.raises(AuthError) as exc_info:
        _refresh_access_token(
            client=_FakeClient(),
            portal_base_url="https://portal.nousresearch.com",
            client_id="hermes-cli",
            refresh_token="rt_anything",
        )

    assert "Refresh session has been revoked" in str(exc_info.value)
    # Must not have been rewritten with the reuse message.
    assert "external process" not in str(exc_info.value).lower()


# =============================================================================
# Shared Nous token store — cross-profile persistence (Codex-style auto-import)
# =============================================================================


@pytest.fixture
def shared_store_env(tmp_path, monkeypatch):
    """Redirect HERMES_SHARED_AUTH_DIR to a tmp_path.

    Required for every test that exercises the shared Nous store — the
    in-auth.py seat belt refuses to touch the real user's shared store
    under pytest, so tests that forget this fixture fail loudly instead
    of corrupting real state.
    """
    shared_dir = tmp_path / "shared"
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    return shared_dir


def test_shared_store_seat_belt_refuses_real_home_under_pytest(monkeypatch):
    """Without HERMES_SHARED_AUTH_DIR override, the seat belt must trip.

    Mirrors the existing ``_auth_file_path`` seat belt: forgetting to
    redirect this store in a test must fail loudly instead of silently
    writing to the user's real ``~/.hermes/shared/`` across CI runs.
    """
    from hermes_cli.auth import _nous_shared_store_path

    monkeypatch.delenv("HERMES_SHARED_AUTH_DIR", raising=False)

    with pytest.raises(RuntimeError, match="shared Nous auth store"):
        _nous_shared_store_path()


def test_shared_store_honors_env_override(tmp_path, monkeypatch):
    """HERMES_SHARED_AUTH_DIR must redirect the path."""
    from hermes_cli.auth import _nous_shared_store_path, NOUS_SHARED_STORE_FILENAME

    custom_dir = tmp_path / "custom_shared"
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(custom_dir))

    path = _nous_shared_store_path()
    assert path == custom_dir / NOUS_SHARED_STORE_FILENAME


def test_shared_store_read_missing_returns_none(shared_store_env):
    """Missing file → ``_read_shared_nous_state()`` returns None."""
    from hermes_cli.auth import _read_shared_nous_state

    assert _read_shared_nous_state() is None


def test_shared_store_read_malformed_returns_none(shared_store_env):
    """Unreadable / non-JSON file → None, not an exception."""
    from hermes_cli.auth import _nous_shared_store_path, _read_shared_nous_state

    path = _nous_shared_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")

    assert _read_shared_nous_state() is None


def test_shared_store_read_missing_required_fields_returns_none(shared_store_env):
    """Payload without refresh_token → None (nothing worth importing)."""
    from hermes_cli.auth import _nous_shared_store_path, _read_shared_nous_state

    path = _nous_shared_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_schema": 1, "access_token": "abc"}))

    assert _read_shared_nous_state() is None


def test_shared_store_write_and_read_roundtrip(shared_store_env):
    """Write → read must preserve refresh_token + OAuth URLs."""
    from hermes_cli.auth import (
        _nous_shared_store_path,
        _read_shared_nous_state,
        _write_shared_nous_state,
    )

    state = _full_state_fixture()
    _write_shared_nous_state(state)

    path = _nous_shared_store_path()
    assert path.is_file()

    # Permissions should be 0600 where the platform supports it.
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600 or mode == 0o644  # 0o644 on platforms without chmod

    loaded = _read_shared_nous_state()
    assert loaded is not None
    assert loaded["refresh_token"] == "refresh-tok"
    assert loaded["access_token"] == state["access_token"]
    assert loaded["portal_base_url"] == "https://portal.example.com"
    assert loaded["inference_base_url"] == "https://inference.example.com/v1"
    # Volatile agent_key MUST NOT be persisted to the shared store
    # (24h TTL, profile-specific — only long-lived OAuth tokens are
    # cross-profile useful).
    assert "agent_key" not in loaded


def test_shared_store_write_skips_when_refresh_token_missing(shared_store_env):
    """Write is a no-op when refresh_token is absent (nothing to share)."""
    from hermes_cli.auth import _nous_shared_store_path, _write_shared_nous_state

    state = dict(_full_state_fixture())
    state["refresh_token"] = ""

    _write_shared_nous_state(state)

    assert not _nous_shared_store_path().is_file()


def test_persist_nous_credentials_mirrors_to_shared_store(
    tmp_path, monkeypatch, shared_store_env,
):
    """persist_nous_credentials must populate BOTH per-profile auth.json
    AND the shared store, so a future profile's `hermes auth add nous
    --type oauth` can one-tap import instead of redoing device-code.
    """
    from hermes_cli.auth import (
        _nous_shared_store_path,
        _read_shared_nous_state,
        persist_nous_credentials,
    )

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}})
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    persist_nous_credentials(_full_state_fixture())

    # Per-profile auth.json populated
    payload = json.loads((hermes_home / "auth.json").read_text())
    assert "nous" in payload.get("providers", {})

    # Shared store populated with the same refresh_token
    shared = _read_shared_nous_state()
    assert shared is not None
    assert shared["refresh_token"] == "refresh-tok"

    # Shared file path lives under the tmp override, NOT the real home
    assert str(_nous_shared_store_path()).startswith(str(shared_store_env))


def test_try_import_shared_returns_none_when_store_missing(shared_store_env):
    """No shared store → no rehydrate (fall through to device-code)."""
    from hermes_cli.auth import _try_import_shared_nous_state

    assert _try_import_shared_nous_state() is None


def test_try_import_shared_returns_none_on_refresh_failure(
    shared_store_env, monkeypatch,
):
    """If the portal rejects the stored refresh_token (revoked, expired,
    portal down), _try_import_shared_nous_state must return None so the
    login flow falls back to a fresh device-code run.
    """
    from hermes_cli import auth as auth_mod

    # Seed the shared store
    auth_mod._write_shared_nous_state(_full_state_fixture())

    # Make refresh fail
    def _boom(*_args, **_kwargs):
        raise AuthError(
            "Refresh session has been revoked",
            provider="nous",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth_mod, "refresh_nous_oauth_from_state", _boom)

    assert auth_mod._try_import_shared_nous_state() is None
    assert auth_mod._read_shared_nous_state() is None


def test_try_import_shared_persists_rotated_token_when_jwt_validation_fails(
    shared_store_env, monkeypatch,
):
    """A forced shared import refresh rotates the single-use token before validation.

    If the later inference-JWT validation fails, the shared store must still keep the
    rotated refresh token; otherwise the next import attempt replays the
    consumed token and trips refresh-token reuse.
    """
    from hermes_cli import auth as auth_mod

    shared_state = _full_state_fixture()
    shared_state["refresh_token"] = "refresh-old"
    shared_state["access_token"] = "access-old"
    auth_mod._write_shared_nous_state(shared_state)

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        assert refresh_token == "refresh-old"
        return {
            "access_token": "access-new",
            "refresh_token": "refresh-new",
            "expires_in": 900,
            "token_type": "Bearer",
        }

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _fake_refresh_access_token)

    assert auth_mod._try_import_shared_nous_state() is None

    shared_after = auth_mod._read_shared_nous_state()
    assert shared_after is not None
    assert shared_after["refresh_token"] == "refresh-new"
    assert shared_after["access_token"] == "access-new"


def test_try_import_shared_rehydrates_on_success(shared_store_env, monkeypatch):
    """Happy path: stored refresh_token is accepted, forced refresh
    returns a fresh access_token JWT, and the returned dict has
    every field persist_nous_credentials() needs.
    """
    from hermes_cli import auth as auth_mod

    auth_mod._write_shared_nous_state(_full_state_fixture())
    fresh_jwt = _invoke_jwt(seconds=7200)

    def _fake_refresh(state, **kwargs):
        # Simulate portal returning a fresh inference JWT.
        assert kwargs.get("force_refresh") is True
        return {
            **state,
            "access_token": fresh_jwt,
            "refresh_token": "fresh-refresh-tok",  # rotated
            "agent_key": fresh_jwt,
            "agent_key_expires_at": _future_iso(7200),
        }

    monkeypatch.setattr(auth_mod, "refresh_nous_oauth_from_state", _fake_refresh)

    result = auth_mod._try_import_shared_nous_state()

    assert result is not None
    assert result["access_token"] == fresh_jwt
    assert result["refresh_token"] == "fresh-refresh-tok"
    assert result["agent_key"] == fresh_jwt
    # Preserved from shared state
    assert result["portal_base_url"] == "https://portal.example.com"
    assert result["client_id"] == "hermes-cli"


def test_shared_store_survives_across_profile_switch(
    tmp_path, monkeypatch, shared_store_env,
):
    """End-to-end: profile A logs in → shared store populated → profile B
    (different HERMES_HOME) sees the same shared state and can rehydrate
    without re-running device-code.
    """
    from hermes_cli import auth as auth_mod

    # Profile A: login, which mirrors to shared store
    profile_a = tmp_path / "profile_a"
    profile_a.mkdir(parents=True, exist_ok=True)
    (profile_a / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}})
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    auth_mod.persist_nous_credentials(_full_state_fixture())

    # Profile A's auth.json has nous
    a_payload = json.loads((profile_a / "auth.json").read_text())
    assert "nous" in a_payload.get("providers", {})

    # Profile B: fresh HERMES_HOME, no auth yet, but the shared store
    # persists — _read_shared_nous_state() must still return the tokens.
    profile_b = tmp_path / "profile_b"
    profile_b.mkdir(parents=True, exist_ok=True)
    (profile_b / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}})
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_b))

    # B's own auth.json has no nous
    b_payload = json.loads((profile_b / "auth.json").read_text())
    assert "nous" not in b_payload.get("providers", {})

    # But the shared store is visible
    shared = auth_mod._read_shared_nous_state()
    assert shared is not None
    assert shared["refresh_token"] == "refresh-tok"

    # And a successful rehydrate + persist lands nous into profile B
    b_jwt = _invoke_jwt(seconds=7200)

    def _fake_refresh(state, **kwargs):
        return {
            **state,
            "access_token": b_jwt,
            "refresh_token": "b-refresh-tok",
            "agent_key": b_jwt,
            "agent_key_expires_at": _future_iso(7200),
        }

    monkeypatch.setattr(auth_mod, "refresh_nous_oauth_from_state", _fake_refresh)
    result = auth_mod._try_import_shared_nous_state()
    assert result is not None

    auth_mod.persist_nous_credentials(result)

    b_payload = json.loads((profile_b / "auth.json").read_text())
    assert "nous" in b_payload.get("providers", {})
    assert b_payload["providers"]["nous"]["refresh_token"] == "b-refresh-tok"

    # Shared store was updated with the rotated refresh_token too
    shared_after = auth_mod._read_shared_nous_state()
    assert shared_after is not None
    assert shared_after["refresh_token"] == "b-refresh-tok"


def test_runtime_refresh_uses_newer_shared_token_before_local_stale_token(
    tmp_path, monkeypatch, shared_store_env,
):
    """A sibling profile may rotate the single-use Nous refresh token.

    When this profile later wakes with an expired local token, runtime
    resolution must adopt the shared token before refreshing. Otherwise it
    can submit the stale local refresh token and trigger portal reuse
    revocation for the whole shared session.
    """
    from hermes_cli import auth as auth_mod

    profile_b = tmp_path / "profile_b"
    _setup_nous_auth(
        profile_b,
        access_token="local-expired-access",
        refresh_token="local-stale-refresh",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_b))

    shared_state = _full_state_fixture()
    shared_token = _invoke_jwt(seconds=3600)
    shared_state["access_token"] = shared_token
    shared_state["refresh_token"] = "shared-fresh-refresh"
    shared_state["expires_at"] = "2099-01-01T00:00:00+00:00"
    shared_state["scope"] = "inference:invoke"
    auth_mod._write_shared_nous_state(shared_state)

    def _refresh_should_not_happen(**_kwargs):
        raise AssertionError("stale profile-local refresh token was used")

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _refresh_should_not_happen)

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["api_key"] == shared_token

    profile_state = auth_mod.get_provider_auth_state("nous")
    assert profile_state is not None
    assert profile_state["refresh_token"] == "shared-fresh-refresh"
    assert profile_state["access_token"] == shared_token


def test_runtime_credentials_merges_shared_token_before_empty_local_access_token(
    tmp_path, monkeypatch, shared_store_env,
):
    """A profile with access_token=None must recover from the shared store.

    ``resolve_nous_access_token()`` already merges shared OAuth state before
    giving up. The runtime path must do the same so sibling profiles that
    share a valid Nous login do not dead-end on an empty local auth.json.
    """
    from hermes_cli import auth as auth_mod

    profile_b = tmp_path / "profile_b"
    _setup_nous_auth(
        profile_b,
        access_token="local-placeholder",
        refresh_token="local-stale-refresh",
        expires_at="2000-01-01T00:00:00+00:00",
        expires_in=0,
    )
    auth_path = profile_b / "auth.json"
    auth_payload = json.loads(auth_path.read_text())
    auth_payload["providers"]["nous"]["access_token"] = None
    auth_path.write_text(json.dumps(auth_payload, indent=2))
    monkeypatch.setenv("HERMES_HOME", str(profile_b))

    shared_state = _full_state_fixture()
    shared_token = _invoke_jwt(seconds=3600)
    shared_state["access_token"] = shared_token
    shared_state["refresh_token"] = "shared-fresh-refresh"
    shared_state["expires_at"] = _future_iso(3600)
    shared_state["scope"] = auth_mod.DEFAULT_NOUS_SCOPE
    shared_state["inference_base_url"] = auth_mod.DEFAULT_NOUS_INFERENCE_URL
    auth_mod._write_shared_nous_state(shared_state)

    def _refresh_should_not_happen(**_kwargs):
        raise AssertionError("empty local access token should recover from shared store")

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _refresh_should_not_happen)

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["api_key"] == shared_token
    assert creds["base_url"] == auth_mod.DEFAULT_NOUS_INFERENCE_URL

    profile_state = auth_mod.get_provider_auth_state("nous")
    assert profile_state is not None
    assert profile_state["access_token"] == shared_token
    assert profile_state["refresh_token"] == "shared-fresh-refresh"
    assert profile_state["agent_key"] == shared_token


def test_runtime_shared_recovery_recomputes_routing_before_force_refresh(
    tmp_path, monkeypatch, shared_store_env,
):
    """A shared refresh token must use its own routing metadata."""
    from hermes_cli import auth as auth_mod

    profile_b = tmp_path / "profile_b"
    _setup_nous_auth(
        profile_b,
        access_token="local-placeholder",
        refresh_token="local-stale-refresh",
        expires_at="2000-01-01T00:00:00+00:00",
        expires_in=0,
    )
    auth_path = profile_b / "auth.json"
    auth_payload = json.loads(auth_path.read_text())
    local_state = auth_payload["providers"]["nous"]
    local_state["access_token"] = None
    local_state["portal_base_url"] = "http://127.0.0.1:8001"
    local_state["client_id"] = "local-client"
    auth_path.write_text(json.dumps(auth_payload, indent=2))
    monkeypatch.setenv("HERMES_HOME", str(profile_b))

    shared_state = _full_state_fixture()
    shared_state["access_token"] = _invoke_jwt(seconds=3600)
    shared_state["refresh_token"] = "shared-refresh"
    shared_state["expires_at"] = _future_iso(3600)
    shared_state["scope"] = auth_mod.DEFAULT_NOUS_SCOPE
    shared_state["portal_base_url"] = "http://localhost:8002"
    shared_state["client_id"] = "shared-client"
    shared_state["inference_base_url"] = auth_mod.DEFAULT_NOUS_INFERENCE_URL
    auth_mod._write_shared_nous_state(shared_state)

    captured = {}
    refreshed_token = _invoke_jwt(seconds=7200)

    def _refresh(**kwargs):
        captured.update(kwargs)
        return {
            "access_token": refreshed_token,
            "refresh_token": "rotated-shared-refresh",
            "expires_in": 7200,
            "scope": auth_mod.DEFAULT_NOUS_SCOPE,
            "inference_base_url": auth_mod.DEFAULT_NOUS_INFERENCE_URL,
        }

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _refresh)

    creds = auth_mod.resolve_nous_runtime_credentials(force_refresh=True)

    assert captured["portal_base_url"] == "http://localhost:8002"
    assert captured["client_id"] == "shared-client"
    assert captured["refresh_token"] == "shared-refresh"
    assert creds["api_key"] == refreshed_token

    profile_state = auth_mod.get_provider_auth_state("nous")
    assert profile_state is not None
    assert profile_state["portal_base_url"] == "http://localhost:8002"
    assert profile_state["client_id"] == "shared-client"
    assert profile_state["refresh_token"] == "rotated-shared-refresh"


def test_runtime_unusable_local_token_recomputes_shared_routing(
    tmp_path, monkeypatch, shared_store_env,
):
    """The unusable-token merge branch must also adopt shared routing."""
    from hermes_cli import auth as auth_mod

    profile_b = tmp_path / "profile_b"
    _setup_nous_auth(
        profile_b,
        access_token="local-not-an-invoke-jwt",
        refresh_token="local-stale-refresh",
        expires_at="2000-01-01T00:00:00+00:00",
        expires_in=0,
    )
    auth_path = profile_b / "auth.json"
    auth_payload = json.loads(auth_path.read_text())
    local_state = auth_payload["providers"]["nous"]
    local_state["portal_base_url"] = "http://127.0.0.1:8001"
    local_state["client_id"] = "local-client"
    auth_path.write_text(json.dumps(auth_payload, indent=2))
    monkeypatch.setenv("HERMES_HOME", str(profile_b))

    shared_state = _full_state_fixture()
    shared_state["access_token"] = "shared-not-an-invoke-jwt"
    shared_state["refresh_token"] = "shared-refresh"
    shared_state["expires_at"] = _future_iso(3600)
    shared_state["portal_base_url"] = "http://localhost:8002"
    shared_state["client_id"] = "shared-client"
    auth_mod._write_shared_nous_state(shared_state)

    captured = {}
    refreshed_token = _invoke_jwt(seconds=7200)

    def _refresh(**kwargs):
        captured.update(kwargs)
        return {
            "access_token": refreshed_token,
            "refresh_token": "rotated-refresh",
            "expires_in": 7200,
            "scope": auth_mod.DEFAULT_NOUS_SCOPE,
        }

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _refresh)

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert captured["portal_base_url"] == "http://localhost:8002"
    assert captured["client_id"] == "shared-client"
    assert captured["refresh_token"] == "shared-refresh"
    assert creds["api_key"] == refreshed_token


def test_runtime_refresh_persists_routing_before_jwt_validation_failure(
    tmp_path, monkeypatch, shared_store_env,
):
    """Rotated tokens and routing survive a later runtime JWT rejection."""
    from hermes_cli import auth as auth_mod

    profile_b = tmp_path / "profile_b"
    _setup_nous_auth(
        profile_b,
        access_token="local-unusable",
        refresh_token="local-refresh",
        expires_at="2000-01-01T00:00:00+00:00",
        expires_in=0,
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_b))

    rotated_refresh = "rotated-refresh"
    refreshed_url = auth_mod.DEFAULT_NOUS_INFERENCE_URL
    monkeypatch.setattr(
        auth_mod,
        "_refresh_access_token",
        lambda **_kwargs: {
            "access_token": "refreshed-but-not-jwt",
            "refresh_token": rotated_refresh,
            "expires_in": 3600,
            "scope": auth_mod.DEFAULT_NOUS_SCOPE,
            "inference_base_url": refreshed_url,
        },
    )

    with pytest.raises(AuthError):
        auth_mod.resolve_nous_runtime_credentials()

    profile_state = auth_mod.get_provider_auth_state("nous")
    assert profile_state is not None
    assert profile_state["refresh_token"] == rotated_refresh
    assert profile_state["inference_base_url"] == refreshed_url

    shared_state = auth_mod._read_shared_nous_state()
    assert shared_state is not None
    assert shared_state["refresh_token"] == rotated_refresh
    assert shared_state["inference_base_url"] == refreshed_url


def test_runtime_shared_recovery_honors_inference_env_override(
    tmp_path, monkeypatch, shared_store_env,
):
    """Shared state is persisted, but the operator inference override wins."""
    from hermes_cli import auth as auth_mod

    profile_b = tmp_path / "profile_b"
    _setup_nous_auth(
        profile_b,
        access_token="local-placeholder",
        refresh_token="local-stale-refresh",
        expires_at="2000-01-01T00:00:00+00:00",
        expires_in=0,
    )
    auth_path = profile_b / "auth.json"
    auth_payload = json.loads(auth_path.read_text())
    auth_payload["providers"]["nous"]["access_token"] = None
    auth_path.write_text(json.dumps(auth_payload, indent=2))
    monkeypatch.setenv("HERMES_HOME", str(profile_b))
    override_url = "https://operator.example/v1"
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", override_url)

    shared_state = _full_state_fixture()
    shared_token = _invoke_jwt(seconds=3600)
    shared_state["access_token"] = shared_token
    shared_state["refresh_token"] = "shared-refresh"
    shared_state["expires_at"] = _future_iso(3600)
    shared_state["scope"] = auth_mod.DEFAULT_NOUS_SCOPE
    shared_state["inference_base_url"] = auth_mod.DEFAULT_NOUS_INFERENCE_URL
    auth_mod._write_shared_nous_state(shared_state)

    monkeypatch.setattr(
        auth_mod,
        "_refresh_access_token",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected refresh")),
    )

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["base_url"] == override_url
    profile_state = auth_mod.get_provider_auth_state("nous")
    assert profile_state is not None
    assert profile_state["inference_base_url"] == auth_mod.DEFAULT_NOUS_INFERENCE_URL


def test_managed_gateway_access_token_uses_newer_shared_token(
    tmp_path, monkeypatch, shared_store_env,
):
    """Managed-tool token reads share the same stale-refresh-token hazard."""
    from hermes_cli import auth as auth_mod

    profile_b = tmp_path / "profile_b"
    _setup_nous_auth(
        profile_b,
        access_token="local-expired-access",
        refresh_token="local-stale-refresh",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_b))

    shared_state = _full_state_fixture()
    shared_state["access_token"] = "shared-fresh-access"
    shared_state["refresh_token"] = "shared-fresh-refresh"
    shared_state["expires_at"] = "2099-01-01T00:00:00+00:00"
    auth_mod._write_shared_nous_state(shared_state)

    def _refresh_should_not_happen(**_kwargs):
        raise AssertionError("stale profile-local refresh token was used")

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _refresh_should_not_happen)

    assert auth_mod.resolve_nous_access_token() == "shared-fresh-access"

    profile_state = auth_mod.get_provider_auth_state("nous")
    assert profile_state is not None
    assert profile_state["refresh_token"] == "shared-fresh-refresh"

class TestStalePortalBaseUrlMigration:
    """_migrate_stale_nous_portal_url auto-corrects stale portal_base_url on load."""

    def test_migrates_stale_portal_url_on_load(self, tmp_path, monkeypatch):
        from hermes_cli.auth import _load_auth_store, DEFAULT_NOUS_PORTAL_URL

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "version": 1,
            "active_provider": "nous",
            "providers": {
                "nous": {
                    "portal_base_url": "https://api.nousresearch.com",
                    "access_token": "test-token",
                    "refresh_token": "test-refresh",
                }
            },
        }))

        store = _load_auth_store(auth_file)
        nous = store["providers"]["nous"]
        assert nous["portal_base_url"] == DEFAULT_NOUS_PORTAL_URL

    def test_preserves_correct_portal_url(self, tmp_path, monkeypatch):
        from hermes_cli.auth import _load_auth_store, DEFAULT_NOUS_PORTAL_URL

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "version": 1,
            "active_provider": "nous",
            "providers": {
                "nous": {
                    "portal_base_url": DEFAULT_NOUS_PORTAL_URL,
                    "access_token": "test-token",
                    "refresh_token": "test-refresh",
                }
            },
        }))

        store = _load_auth_store(auth_file)
        nous = store["providers"]["nous"]
        assert nous["portal_base_url"] == DEFAULT_NOUS_PORTAL_URL

    def test_ignores_other_providers(self, tmp_path, monkeypatch):
        from hermes_cli.auth import _load_auth_store, DEFAULT_NOUS_PORTAL_URL

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {},
        }))

        store = _load_auth_store(auth_file)
        assert "nous" not in store.get("providers", {})

    def test_noop_when_nous_state_not_dict(self, tmp_path, monkeypatch):
        from hermes_cli.auth import _load_auth_store

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "version": 1,
            "active_provider": "nous",
            "providers": {"nous": None},
        }))

        store = _load_auth_store(auth_file)
        assert store["providers"]["nous"] is None

    def test_runtime_fallback_for_invalid_portal_url(self, tmp_path, monkeypatch):
        from hermes_cli import auth as auth_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _setup_nous_auth(
            tmp_path,
            access_token="expired-access",
            refresh_token="valid-refresh",
            expires_at="2025-01-01T00:00:00+00:00",
        )
        auth_file = tmp_path / "auth.json"
        store = json.loads(auth_file.read_text())
        store["providers"]["nous"]["portal_base_url"] = "https://api.nousresearch.com"
        auth_file.write_text(json.dumps(store, indent=2))

        refresh_calls = []

        def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
            del client, client_id, refresh_token
            refresh_calls.append(portal_base_url)
            return {
                "access_token": "refreshed-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

        monkeypatch.setattr(auth_mod, "_refresh_access_token", _fake_refresh_access_token)

        token = auth_mod.resolve_nous_access_token()
        assert token == "refreshed-access"
        assert len(refresh_calls) == 1
        assert refresh_calls[0] == auth_mod.DEFAULT_NOUS_PORTAL_URL

    def test_runtime_accepts_localhost(self, tmp_path, monkeypatch):
        from hermes_cli import auth as auth_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _setup_nous_auth(
            tmp_path,
            access_token="expired-access",
            refresh_token="valid-refresh",
            expires_at="2025-01-01T00:00:00+00:00",
        )
        auth_file = tmp_path / "auth.json"
        store = json.loads(auth_file.read_text())
        store["providers"]["nous"]["portal_base_url"] = "http://localhost:8080/"
        auth_file.write_text(json.dumps(store, indent=2))

        refresh_calls = []

        def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
            del client, client_id, refresh_token
            refresh_calls.append(portal_base_url)
            return {
                "access_token": "refreshed-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

        monkeypatch.setattr(auth_mod, "_refresh_access_token", _fake_refresh_access_token)

        token = auth_mod.resolve_nous_access_token()
        assert token == "refreshed-access"
        assert len(refresh_calls) == 1
        assert "localhost" in refresh_calls[0]

    def test_runtime_credentials_fallback_for_invalid_portal_url(self, tmp_path, monkeypatch):
        """resolve_nous_runtime_credentials also rejects an off-allowlist portal host.

        The refresh token is POSTed to portal_base_url on refresh; a poisoned
        value must never receive the bearer. This mirrors the guard on
        resolve_nous_access_token so the whole class is covered, not just the
        managed-gateway path.
        """
        from hermes_cli import auth as auth_mod

        hermes_home = tmp_path / "hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _setup_nous_auth(
            hermes_home,
            access_token=_invoke_jwt(seconds=-60),
            refresh_token="valid-refresh",
            expires_at=_future_iso(-60),
            expires_in=0,
        )
        auth_file = hermes_home / "auth.json"
        store = json.loads(auth_file.read_text())
        store["providers"]["nous"]["portal_base_url"] = "https://evil.example.com"
        auth_file.write_text(json.dumps(store, indent=2))

        refresh_calls = []

        def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
            del client, client_id, refresh_token
            refresh_calls.append(portal_base_url)
            return {
                "access_token": _invoke_jwt(seconds=3600),
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": "inference:invoke",
                "inference_base_url": "https://inference-api.nousresearch.com/v1",
            }

        monkeypatch.setattr(auth_mod, "_refresh_access_token", _fake_refresh_access_token)

        auth_mod.resolve_nous_runtime_credentials()
        assert len(refresh_calls) == 1
        assert refresh_calls[0] == auth_mod.DEFAULT_NOUS_PORTAL_URL

    def test_runtime_credentials_rejects_http_for_production_portal(
        self, tmp_path, monkeypatch,
    ):
        """An allowlisted production host is still unsafe over plain HTTP."""
        from hermes_cli import auth as auth_mod

        hermes_home = tmp_path / "hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _setup_nous_auth(
            hermes_home,
            access_token=_invoke_jwt(seconds=-60),
            refresh_token="valid-refresh",
            expires_at=_future_iso(-60),
            expires_in=0,
        )
        auth_file = hermes_home / "auth.json"
        store = json.loads(auth_file.read_text())
        store["providers"]["nous"]["portal_base_url"] = (
            "http://portal.nousresearch.com"
        )
        auth_file.write_text(json.dumps(store, indent=2))

        refresh_calls = []

        def _fake_refresh_access_token(
            *, client, portal_base_url, client_id, refresh_token,
        ):
            del client, client_id, refresh_token
            refresh_calls.append(portal_base_url)
            return {
                "access_token": _invoke_jwt(seconds=3600),
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "scope": "inference:invoke",
            }

        monkeypatch.setattr(
            auth_mod, "_refresh_access_token", _fake_refresh_access_token
        )

        auth_mod.resolve_nous_runtime_credentials()
        assert refresh_calls == [auth_mod.DEFAULT_NOUS_PORTAL_URL]
