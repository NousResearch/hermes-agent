"""Regression tests for Nous OAuth refresh + agent-key mint interactions."""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from hermes_cli.auth import AuthError, get_provider_auth_state, resolve_nous_runtime_credentials


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

