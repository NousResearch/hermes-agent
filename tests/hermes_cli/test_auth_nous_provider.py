"""Regression tests for Nous OAuth refresh + agent-key mint interactions."""

import json
import os
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

    def test_missing_ca_bundle_in_auth_state_falls_back(self):
        from hermes_cli.auth import _resolve_verify

        result = _resolve_verify(auth_state={
            "tls": {"insecure": False, "ca_bundle": "/nonexistent/ca-bundle.pem"},
        })
        assert result is True

    def test_valid_ca_bundle_in_auth_state_is_returned(self, tmp_path):
        from hermes_cli.auth import _resolve_verify

        ca_file = tmp_path / "ca-bundle.pem"
        ca_file.write_text("fake cert")
        result = _resolve_verify(auth_state={
            "tls": {"insecure": False, "ca_bundle": str(ca_file)},
        })
        assert result == str(ca_file)

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

    def test_explicit_ca_bundle_param_valid_is_returned(self, tmp_path):
        from hermes_cli.auth import _resolve_verify

        ca_file = tmp_path / "explicit-ca.pem"
        ca_file.write_text("fake cert")
        result = _resolve_verify(ca_bundle=str(ca_file))
        assert result == str(ca_file)


def _setup_nous_auth(
    hermes_home: Path,
    *,
    access_token: str = "access-old",
    refresh_token: str = "refresh-old",
) -> None:
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
                "scope": "inference:mint_agent_key",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "obtained_at": "2026-02-01T00:00:00+00:00",
                "expires_in": 0,
                "expires_at": "2026-02-01T00:00:00+00:00",
                "agent_key": None,
                "agent_key_id": None,
                "agent_key_expires_at": None,
                "agent_key_expires_in": None,
                "agent_key_reused": None,
                "agent_key_obtained_at": None,
            }
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store, indent=2))


def _mint_payload(api_key: str = "agent-key") -> dict:
    return {
        "api_key": api_key,
        "key_id": "key-id-1",
        "expires_at": datetime.now(timezone.utc).isoformat(),
        "expires_in": 1800,
        "reused": False,
    }


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
    entry = PooledCredential.from_dict("nous", {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "portal_base_url": "https://portal.example.com",
        "inference_base_url": "https://inference.example.com/v1",
        "agent_key": "test-agent-key",
        "agent_key_expires_at": "2099-01-01T00:00:00+00:00",
        "label": "dashboard device_code",
        "auth_type": "oauth",
        "source": "manual:dashboard_device_code",
        "base_url": "https://inference.example.com/v1",
    })
    pool.add_entry(entry)

    status = get_nous_auth_status()
    assert status["logged_in"] is True
    assert "example.com" in str(status.get("portal_base_url", ""))


def test_get_nous_auth_status_auth_store_fallback(tmp_path, monkeypatch):
    """get_nous_auth_status() falls back to auth store when credential
    pool is empty.
    """
    from hermes_cli.auth import get_nous_auth_status

    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(hermes_home, access_token="at-123")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    status = get_nous_auth_status()
    assert status["logged_in"] is True
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


def test_refresh_token_persisted_when_mint_returns_insufficient_credits(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(hermes_home, refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    refresh_calls = []
    mint_calls = {"count": 0}

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        refresh_calls.append(refresh_token)
        idx = len(refresh_calls)
        return {
            "access_token": f"access-{idx}",
            "refresh_token": f"refresh-{idx}",
            "expires_in": 0,
            "token_type": "Bearer",
        }

    def _fake_mint_agent_key(*, client, portal_base_url, access_token, min_ttl_seconds):
        mint_calls["count"] += 1
        if mint_calls["count"] == 1:
            raise AuthError("credits exhausted", provider="nous", code="insufficient_credits")
        return _mint_payload(api_key="agent-key-2")

    monkeypatch.setattr("hermes_cli.auth._refresh_access_token", _fake_refresh_access_token)
    monkeypatch.setattr("hermes_cli.auth._mint_agent_key", _fake_mint_agent_key)

    with pytest.raises(AuthError) as exc:
        resolve_nous_runtime_credentials(min_key_ttl_seconds=300)
    assert exc.value.code == "insufficient_credits"

    state_after_failure = get_provider_auth_state("nous")
    assert state_after_failure is not None
    assert state_after_failure["refresh_token"] == "refresh-1"
    assert state_after_failure["access_token"] == "access-1"

    creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=300)
    assert creds["api_key"] == "agent-key-2"
    assert refresh_calls == ["refresh-old", "refresh-1"]


def test_refresh_token_persisted_when_mint_times_out(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(hermes_home, refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        return {
            "access_token": "access-1",
            "refresh_token": "refresh-1",
            "expires_in": 0,
            "token_type": "Bearer",
        }

    def _fake_mint_agent_key(*, client, portal_base_url, access_token, min_ttl_seconds):
        raise httpx.ReadTimeout("mint timeout")

    monkeypatch.setattr("hermes_cli.auth._refresh_access_token", _fake_refresh_access_token)
    monkeypatch.setattr("hermes_cli.auth._mint_agent_key", _fake_mint_agent_key)

    with pytest.raises(httpx.ReadTimeout):
        resolve_nous_runtime_credentials(min_key_ttl_seconds=300)

    state_after_failure = get_provider_auth_state("nous")
    assert state_after_failure is not None
    assert state_after_failure["refresh_token"] == "refresh-1"
    assert state_after_failure["access_token"] == "access-1"


def test_mint_retry_uses_latest_rotated_refresh_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_nous_auth(hermes_home, refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    refresh_calls = []
    mint_calls = {"count": 0}

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        refresh_calls.append(refresh_token)
        idx = len(refresh_calls)
        return {
            "access_token": f"access-{idx}",
            "refresh_token": f"refresh-{idx}",
            "expires_in": 0,
            "token_type": "Bearer",
        }

    def _fake_mint_agent_key(*, client, portal_base_url, access_token, min_ttl_seconds):
        mint_calls["count"] += 1
        if mint_calls["count"] == 1:
            raise AuthError("stale access token", provider="nous", code="invalid_token")
        return _mint_payload(api_key="agent-key")

    monkeypatch.setattr("hermes_cli.auth._refresh_access_token", _fake_refresh_access_token)
    monkeypatch.setattr("hermes_cli.auth._mint_agent_key", _fake_mint_agent_key)

    creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=300)
    assert creds["api_key"] == "agent-key"
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
        monkeypatch.setattr(models_mod, "filter_nous_free_models", lambda ids, p: ids)
        monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda: None)
        monkeypatch.setattr(
            models_mod, "partition_nous_models_by_tier",
            lambda ids, p, free_tier=False: (ids, []),
        )
        monkeypatch.setattr(ns, "prompt_enable_tool_gateway", lambda cfg: None)

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
        self._patch_login_internals(
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
        assert auth_after.get("active_provider") in (None, "")
        # But Nous creds are still saved
        assert "nous" in auth_after.get("providers", {})


