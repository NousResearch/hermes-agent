"""Tests for native Cline/ClinePass OAuth support."""

import json
import time
from pathlib import Path

from hermes_cli.auth import (
    DEFAULT_CLINE_INFERENCE_BASE_URL,
    _save_cline_tokens,
    format_cline_api_key,
    resolve_cline_runtime_credentials,
    resolve_provider,
)
from hermes_cli.runtime_provider import resolve_runtime_provider


def _clear_cline_env(monkeypatch):
    for key in (
        "CLINE_API_KEY",
        "CLINEPASS_TOKEN",
        "CLINE_ACCESS_TOKEN",
        "CLINE_BASE_URL",
        "CLINE_API_BASE_URL",
        "HERMES_CLINE_API_BASE_URL",
        "HERMES_CLINE_INFERENCE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_format_cline_api_key_accepts_raw_workos_and_bearer():
    assert format_cline_api_key("abc.def.ghi") == "workos:abc.def.ghi"
    assert format_cline_api_key("workos:abc.def.ghi") == "workos:abc.def.ghi"
    assert format_cline_api_key("Bearer workos:abc.def.ghi") == "workos:abc.def.ghi"


def test_resolve_cline_pass_uses_shared_cline_auth_store(tmp_path, monkeypatch):
    _clear_cline_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_cline_tokens(
        {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_at_ms": int((time.time() + 3600) * 1000),
            "email": "user@example.com",
        },
        provider_id="cline-pass",
    )

    creds = resolve_cline_runtime_credentials("cline-pass")
    assert creds["provider"] == "cline-pass"
    assert creds["api_key"] == "workos:access-token"
    assert creds["base_url"] == DEFAULT_CLINE_INFERENCE_BASE_URL

    auth_store = json.loads((hermes_home / "auth.json").read_text())
    assert "cline" in auth_store["providers"]
    assert "cline-pass" not in auth_store["providers"]


def test_resolve_cline_runtime_refreshes_expired_token(tmp_path, monkeypatch):
    _clear_cline_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_cline_tokens(
        {
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "expires_at_ms": int((time.time() - 60) * 1000),
        },
        provider_id="cline",
    )

    called = {"count": 0}

    def _fake_refresh(state, timeout_seconds, *, provider_id="cline"):
        called["count"] += 1
        updated = dict(state)
        updated.update(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_at_ms": int((time.time() + 3600) * 1000),
                "last_refresh": "2026-01-01T00:00:00Z",
            }
        )
        return updated

    monkeypatch.setattr("hermes_cli.auth._refresh_cline_auth_tokens", _fake_refresh)

    creds = resolve_cline_runtime_credentials("cline")
    assert called["count"] == 1
    assert creds["api_key"] == "workos:new-access"


def test_runtime_provider_uses_cline_env_token_from_pool(tmp_path, monkeypatch):
    _clear_cline_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CLINE_API_KEY", "raw-browser-token")

    runtime = resolve_runtime_provider(requested="cline")
    assert runtime["provider"] == "cline"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["api_key"] == "workos:raw-browser-token"
    assert runtime["base_url"] == DEFAULT_CLINE_INFERENCE_BASE_URL


def test_auto_provider_detects_cline_env_tokens(tmp_path, monkeypatch):
    _clear_cline_env(monkeypatch)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    monkeypatch.setenv("CLINE_API_KEY", "raw-browser-token")
    assert resolve_provider("auto") == "cline"

    monkeypatch.delenv("CLINE_API_KEY", raising=False)
    monkeypatch.setenv("CLINEPASS_TOKEN", "raw-browser-token")
    assert resolve_provider("auto") == "cline-pass"
