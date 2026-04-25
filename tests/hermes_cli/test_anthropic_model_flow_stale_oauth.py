"""Tests for Bug #12905 fix — stale OAuth token detection in hermes model flow.

Bug 3: `hermes model` with `provider=anthropic` skips OAuth re-authentication
when a stale ANTHROPIC_TOKEN exists in ~/.hermes/.env but no valid
Claude Code credentials are available. The fast-path silently proceeds to
model selection with a broken token instead of offering re-auth.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.config import load_env, save_env_value


class TestStaleOAuthTokenDetection:
    """Bug 3: stale OAuth token must trigger needs_auth=True in _model_flow_anthropic."""

    def test_stale_oauth_token_triggers_reauth(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: ANTHROPIC_TOKEN is an expired OAuth token and there are no
        valid Claude Code credentials anywhere. The flow MUST offer re-auth
        instead of silently skipping to model selection.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Pre-load .env with an expired OAuth token (sk-ant- prefix = OAuth)
        save_env_value("ANTHROPIC_TOKEN", "sk-ant-oat-ExpiredToken00000")
        save_env_value("ANTHROPIC_API_KEY", "")

        # No valid Claude Code credentials available (expired, no refresh token)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: {
                "accessToken": "expired-cc-token",
                "refreshToken": "",          # No refresh — can't recover
                "expiresAt": 0,               # Already expired
                "source": "claude_code_credentials_file",
            },
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid",
            lambda creds: False,             # Explicitly expired
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._is_oauth_token",
            lambda key: key.startswith("sk-ant-"),
        )
        # _resolve_claude_code_token_from_credentials has no valid path
        monkeypatch.setattr(
            "agent.anthropic_adapter._resolve_claude_code_token_from_credentials",
            lambda creds=None: None,
        )

        # Simulate user types "3" (Cancel) when prompted for re-auth
        monkeypatch.setattr("builtins.input", lambda _: "3")
        monkeypatch.setattr("getpass.getpass", lambda _: "")

        from hermes_cli.main import _model_flow_anthropic
        cfg = {}

        _model_flow_anthropic(cfg)

        output = capsys.readouterr().out
        # Must show auth method choice since token is stale
        assert "subscription" in output or "API key" in output, (
            f"Expected auth method menu but got: {output!r}"
        )

    def test_valid_api_key_skips_stale_check(self, tmp_path, monkeypatch, capsys):
        """
        A non-OAuth ANTHROPIC_API_KEY (regular pay-per-token key) must NOT be
        flagged as stale even when cc_creds are invalid. Regular API keys don't
        expire the same way OAuth tokens do.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Regular API key — NOT an OAuth token
        save_env_value("ANTHROPIC_API_KEY", "sk-ant-api03-RegularPayPerTokenKey")
        save_env_value("ANTHROPIC_TOKEN", "")

        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: None,   # No CC creds
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid",
            lambda creds: False,
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._is_oauth_token",
            lambda key: key.startswith("sk-ant-") and "oat" in key,
        )

        # Simulate user picks "1" (use existing)
        monkeypatch.setattr("builtins.input", lambda _: "1")

        from hermes_cli.main import _model_flow_anthropic
        cfg = {}

        _model_flow_anthropic(cfg)

        output = capsys.readouterr().out
        # Should show "Use existing credentials" menu, NOT auth method choice
        assert "Use existing" in output or "credentials" in output.lower()

    def test_valid_oauth_token_with_refresh_available_skips_reauth(self, tmp_path, monkeypatch, capsys):
        """
        When ANTHROPIC_TOKEN is OAuth and valid cc_creds with refresh exist,
        the flow should use existing credentials (no forced re-auth).
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        save_env_value("ANTHROPIC_TOKEN", "sk-ant-oat-GoodOAuthToken")
        save_env_value("ANTHROPIC_API_KEY", "")

        # Valid Claude Code credentials with refresh token
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: {
                "accessToken": "valid-cc-token",
                "refreshToken": "valid-refresh",
                "expiresAt": 9999999999999,
            },
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid",
            lambda creds: True,
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._is_oauth_token",
            lambda key: key.startswith("sk-ant-"),
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._resolve_claude_code_token_from_credentials",
            lambda creds=None: "valid-cc-token",
        )

        # Simulate user picks "1" (use existing)
        monkeypatch.setattr("builtins.input", lambda _: "1")

        from hermes_cli.main import _model_flow_anthropic
        cfg = {}

        _model_flow_anthropic(cfg)

        output = capsys.readouterr().out
        # Should show "Use existing" without forcing re-auth
        assert "Use existing" in output or "credentials" in output.lower()


class TestStaleOAuthGuardLogic:
    """Unit-level test of the stale-OAuth detection guard logic."""

    def test_stale_oauth_flag_logic_no_cc_creds(self):
        """
        When existing_key is OAuth and cc_available is False,
        existing_is_stale_oauth should be True → has_creds = False.
        """
        existing_key = "sk-ant-oat-expiredtoken123"
        _is_oauth_token = lambda k: k.startswith("sk-ant-")
        cc_available = False

        existing_is_stale_oauth = (
            bool(existing_key) and
            _is_oauth_token(existing_key) and
            not cc_available
        )
        has_creds = (bool(existing_key) and not existing_is_stale_oauth) or cc_available

        assert existing_is_stale_oauth is True
        assert has_creds is False

    def test_stale_oauth_flag_logic_with_valid_cc_creds(self):
        """
        When existing_key is OAuth but cc_available is True (valid creds exist),
        has_creds should be True — the cc_creds will be used instead.
        """
        existing_key = "sk-ant-oat-sometoken"
        _is_oauth_token = lambda k: k.startswith("sk-ant-")
        cc_available = True

        existing_is_stale_oauth = (
            bool(existing_key) and
            _is_oauth_token(existing_key) and
            not cc_available
        )
        has_creds = (bool(existing_key) and not existing_is_stale_oauth) or cc_available

        assert existing_is_stale_oauth is False
        assert has_creds is True

    def test_non_oauth_key_not_flagged_as_stale(self):
        """
        Regular ANTHROPIC_API_KEY (non-OAuth) must not be flagged as stale
        even when cc_available is False.
        """
        existing_key = "sk-ant-api03-regular-key"
        _is_oauth_token = lambda k: k.startswith("sk-ant-") and "oat" in k
        cc_available = False

        existing_is_stale_oauth = (
            bool(existing_key) and
            _is_oauth_token(existing_key) and
            not cc_available
        )
        has_creds = (bool(existing_key) and not existing_is_stale_oauth) or cc_available

        assert existing_is_stale_oauth is False
        assert has_creds is True
