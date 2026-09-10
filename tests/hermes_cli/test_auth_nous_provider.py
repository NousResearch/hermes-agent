"""Regression tests for Nous OAuth refresh and inference JWT interactions."""

import base64
import json
import logging
import sys
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
    """Verify _resolve_verify falls back to default trust when the CA bundle
    path doesn't exist."""

    def test_missing_ca_bundle_in_auth_state_falls_back(self):
        import ssl
        from hermes_cli.auth import _resolve_verify

        result = _resolve_verify(auth_state={
            "tls": {"insecure": False, "ca_bundle": "/nonexistent/ca-bundle.pem"},
        })
        # The subject is "falls back to _default_verify()", not the literal
        # True. Deriving the expectation from the real host keeps the
        # regression covered on the macOS lane too, where _default_verify
        # pins certifi's bundle and returns a context instead.
        if sys.platform == "darwin":
            assert isinstance(result, ssl.SSLContext)
        else:
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


def test_resolve_nous_runtime_credentials_invoke_jwt_is_idempotent(
    tmp_path,
    monkeypatch,
):
    import hermes_cli.auth as auth_mod
    import hermes_cli.auth_nous as auth_nous

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
    monkeypatch.setattr(auth_nous, "_write_shared_nous_state", _unexpected_shared_write)
    monkeypatch.setattr(
        auth_mod,
        "_sync_nous_pool_from_auth_store",
        lambda: sync_calls.append(True),
    )
    monkeypatch.setattr(
        auth_nous,
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
    import hermes_cli.auth_nous as auth_nous

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
    monkeypatch.setattr(auth_nous, "_refresh_access_token", _fake_refresh_access_token)

    caplog.set_level(logging.DEBUG, logger="hermes_cli.auth")
    auth_mod.resolve_nous_runtime_credentials(
        force_refresh=True,
    )

    logged = caplog.text
    assert "using NAS invoke JWT" in logged
    assert not any(
        record.levelno >= logging.INFO
        and "using NAS invoke JWT" in record.getMessage()
        for record in caplog.records
    )
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
        import hermes_cli.auth_nous as auth_nous
        import hermes_cli.models as models_mod
        from hermes_cli import models_pricing
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
            auth_nous, "_nous_device_code_login",
            lambda **kwargs: dict(fake_auth_state),
        )
        monkeypatch.setattr(
            auth_mod, "_prompt_model_selection",
            lambda *a, **kw: prompt_returns,
        )
        monkeypatch.setattr(models_pricing, "get_pricing_for_provider", lambda p: {})
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




def test_try_import_shared_rehydrates_on_success(shared_store_env, monkeypatch):
    """Happy path: stored refresh_token is accepted, forced refresh
    returns a fresh access_token JWT, and the returned dict has
    every field persist_nous_credentials() needs.
    """
    from hermes_cli import auth as auth_mod
    import hermes_cli.auth_nous as auth_nous

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
    monkeypatch.setattr(auth_nous, "refresh_nous_oauth_from_state", _fake_refresh)

    result = auth_mod._try_import_shared_nous_state()

    assert result is not None
    assert result["access_token"] == fresh_jwt
    assert result["refresh_token"] == "fresh-refresh-tok"
    assert result["agent_key"] == fresh_jwt
    # Preserved from shared state
    assert result["portal_base_url"] == "https://portal.example.com"
    assert result["client_id"] == "hermes-cli"










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





    def test_runtime_credentials_rejects_http_for_production_portal(
        self, tmp_path, monkeypatch,
    ):
        """An allowlisted production host is still unsafe over plain HTTP."""
        from hermes_cli import auth as auth_mod
        import hermes_cli.auth_nous as auth_nous

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
        monkeypatch.setattr(
            auth_nous, "_refresh_access_token", _fake_refresh_access_token
        )

        auth_mod.resolve_nous_runtime_credentials()
        assert refresh_calls == [auth_mod.DEFAULT_NOUS_PORTAL_URL]


# =============================================================================
# Device-auth timeout guidance (#20605 kernel from PR #75290)
# =============================================================================


class TestNousDeviceAuthTimeoutMessage:
    def test_timeout_message_mentions_captcha_login_and_retry(self):
        from hermes_cli.auth import _nous_device_auth_timeout_message

        msg = _nous_device_auth_timeout_message("https://portal.nousresearch.com")
        assert "CAPTCHA" in msg
        assert "hermes portal" in msg
        assert "https://portal.nousresearch.com/login" in msg
        # Must NOT point at the nonexistent /device page (live Portal 404s it).
        assert "/device" not in msg

    def test_timeout_message_falls_back_to_default_portal(self):
        from hermes_cli.auth import (
            DEFAULT_NOUS_PORTAL_URL,
            _nous_device_auth_timeout_message,
        )

        msg = _nous_device_auth_timeout_message("")
        assert f"{DEFAULT_NOUS_PORTAL_URL.rstrip('/')}/login" in msg


def test_poll_for_token_timeout_raises_actionable_message():
    """The poll deadline must raise the CAPTCHA-aware guidance at the SOURCE,
    so both the CLI login and the dashboard poller (web_server_oauth._nous_poller,
    which surfaces str(e) to the UI) inherit it."""
    import httpx
    import pytest

    import hermes_cli.auth as auth_mod

    class _PendingClient:
        def post(self, url, data=None):
            request = httpx.Request("POST", url)
            return httpx.Response(
                400,
                json={"error": "authorization_pending"},
                request=request,
            )

    from typing import cast

    with pytest.raises(TimeoutError) as excinfo:
        auth_mod._poll_for_token(
            client=cast(httpx.Client, _PendingClient()),
            portal_base_url="https://portal.nousresearch.com",
            client_id="hermes-cli",
            device_code="device",
            expires_in=1,
            poll_interval=1,
        )

    msg = str(excinfo.value)
    assert "CAPTCHA" in msg
    assert "hermes portal" in msg
    assert "https://portal.nousresearch.com/login" in msg


def test_nous_device_code_login_timeout_raises_actionable_message(monkeypatch):
    """Poll timeout must surface the CAPTCHA-aware guidance through the CLI
    login flow (propagates unchanged from _poll_for_token)."""
    import pytest

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "_request_device_code",
        lambda **kwargs: {
            "device_code": "device",
            "user_code": "SMCL-97YT",
            "verification_uri": "https://portal.nousresearch.com/manage-subscription",
            "verification_uri_complete": (
                "https://portal.nousresearch.com/manage-subscription"
                "?user_code=SMCL-97YT"
            ),
            "expires_in": 600,
            "interval": 1,
        },
    )

    def _timeout(**kwargs):
        raise TimeoutError(
            auth_mod._nous_device_auth_timeout_message(
                kwargs.get("portal_base_url", "")
            )
        )

    monkeypatch.setattr(auth_mod, "_poll_for_token", _timeout)
    monkeypatch.setattr(auth_mod.webbrowser, "open", lambda url: True)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    with pytest.raises(TimeoutError) as excinfo:
        auth_mod._nous_device_code_login(
            portal_base_url="https://portal.nousresearch.com",
            inference_base_url="https://inference.example.com/v1",
            open_browser=False,
            timeout_seconds=1,
        )

    msg = str(excinfo.value)
    assert "CAPTCHA" in msg
    assert "hermes portal" in msg
    assert "https://portal.nousresearch.com/login" in msg
