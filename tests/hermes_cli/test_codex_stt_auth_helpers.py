"""Tests for public Codex auth helpers used by speech-to-text UIs."""

import base64
import json
import time
from unittest.mock import Mock, patch

from hermes_cli.auth import (
    codex_account_id_from_access_token,
    has_codex_runtime_credentials,
    login_openai_codex_credentials_only,
)


def _jwt(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{_part({'alg': 'none'})}.{_part(claims)}.sig"


def test_codex_account_id_is_read_from_access_token_claims():
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": " account-123 "
            }
        }
    )

    assert codex_account_id_from_access_token(token) == "account-123"


def test_codex_credential_probe_uses_read_only_pool_availability():
    pool = Mock()
    pool.has_available.return_value = True
    with patch("agent.credential_pool.load_pool", return_value=pool):
        assert has_codex_runtime_credentials() is True

    pool.has_available.assert_called_once_with()
    pool.select.assert_not_called()


def test_codex_credential_probe_reports_exhausted_pool_unavailable():
    pool = Mock()
    pool.has_available.return_value = False
    with patch("agent.credential_pool.load_pool", return_value=pool):
        assert has_codex_runtime_credentials() is False


def test_codex_credentials_only_login_does_not_activate_model_provider():
    credentials = {
        "tokens": {
            "access_token": "new-token",
            "refresh_token": "refresh-token",
        },
        "last_refresh": "now",
    }
    with (
        patch(
            "hermes_cli.auth.has_codex_runtime_credentials",
            side_effect=[False, True],
        ),
        patch(
            "hermes_cli.auth._codex_device_code_login",
            return_value=credentials,
        ),
        patch("hermes_cli.auth._save_codex_device_login_tokens") as save_tokens,
        patch("hermes_cli.auth._update_config_for_provider") as update_model,
    ):
        assert login_openai_codex_credentials_only() is True

    save_tokens.assert_called_once_with(
        credentials["tokens"], "now", set_active=False
    )
    update_model.assert_not_called()


def test_codex_credentials_only_relogin_restores_suppressed_device_source(
    tmp_path, monkeypatch
):
    token = _jwt(
        {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "account-123"
            },
        }
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "active_provider": "anthropic",
                "providers": {"anthropic": {"api_key": "anthropic-key"}},
                "suppressed_sources": {"openai-codex": ["device_code"]},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  provider: anthropic\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.auth._codex_device_code_login",
        lambda: {
            "tokens": {"access_token": token, "refresh_token": "refresh-token"},
            "last_refresh": "now",
        },
    )

    assert login_openai_codex_credentials_only() is True

    saved = json.loads(auth_path.read_text(encoding="utf-8"))
    assert saved["active_provider"] == "anthropic"
    assert "device_code" not in saved.get("suppressed_sources", {}).get(
        "openai-codex", []
    )
    assert config_path.read_text(encoding="utf-8") == "model:\n  provider: anthropic\n"
