"""Tests for Claude Code credentials opt-out (issue #103978).

When HERMES_DISABLE_CLAUDE_CODE_CREDENTIALS is truthy, resolve_anthropic_token()
must skip auto-discovery of Claude Code's OAuth credentials (Keychain + ~/.claude/.credentials.json),
falling back to the credential pool instead.
"""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from agent.anthropic_credentials import resolve_anthropic_token


class TestClaudeCodeCredentialsOptOut:
    """Issue #103978: opt-out must skip Claude Code credential discovery."""

    def _fake_claude_creds(self, tmp_path):
        """Write fake Claude Code credentials to tmp_path/.claude/.credentials.json."""
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir(parents=True, exist_ok=True)
        cred_file = cred_dir / ".credentials.json"
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "claude-code-token",
                "refreshToken": "claude-code-refresh",
                "expiresAt": 9_999_999_999_999,  # far future
            }
        }))
        return tmp_path

    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "Yes", "YES"])
    def test_opt_out_skips_claude_code_discovery(self, tmp_path, monkeypatch, value):
        """With opt-out enabled, Claude Code credentials must not be read."""
        home = self._fake_claude_creds(tmp_path)
        monkeypatch.setattr("agent.anthropic_credentials.Path.home", lambda: home)
        monkeypatch.setenv("HERMES_DISABLE_CLAUDE_CODE_CREDENTIALS", value)

        # Mock the pool token to verify it's used instead of Claude Code creds
        with patch("agent.anthropic_credentials._resolve_anthropic_pool_token", return_value="pool-token") as mock_pool, \
             patch("agent.anthropic_credentials._resolve_claude_code_token_from_credentials") as mock_cc:
            result = resolve_anthropic_token()
            assert result == "pool-token"
            # Claude Code discovery must NOT have been called
            mock_cc.assert_not_called()
            mock_pool.assert_called_once()

    def test_opt_out_disabled_reads_claude_code(self, tmp_path, monkeypatch):
        """Without opt-out, Claude Code credentials ARE read (default behavior preserved)."""
        home = self._fake_claude_creds(tmp_path)
        monkeypatch.setattr("agent.anthropic_credentials.Path.home", lambda: home)
        # Ensure the env var is not set
        monkeypatch.delenv("HERMES_DISABLE_CLAUDE_CODE_CREDENTIALS", raising=False)

        with patch("agent.anthropic_credentials._resolve_anthropic_pool_token", return_value=None) as mock_pool:
            result = resolve_anthropic_token()
            # Should get the Claude Code token since opt-out is not set
            assert result == "claude-code-token"

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "anything-else"])
    def test_opt_out_only_accepts_truthy_values(self, tmp_path, monkeypatch, value):
        """Falsy/empty values must NOT trigger opt-out."""
        home = self._fake_claude_creds(tmp_path)
        monkeypatch.setattr("agent.anthropic_credentials.Path.home", lambda: home)
        monkeypatch.setenv("HERMES_DISABLE_CLAUDE_CODE_CREDENTIALS", value)

        with patch("agent.anthropic_credentials._resolve_anthropic_pool_token", return_value=None):
            result = resolve_anthropic_token()
            # Should get Claude Code token since opt-out value is not truthy
            assert result == "claude-code-token"

    def test_env_var_takes_precedence_over_later_claude_code(self, tmp_path, monkeypatch):
        """ANTHROPIC_TOKEN env var always wins, regardless of opt-out state."""
        home = self._fake_claude_creds(tmp_path)
        monkeypatch.setattr("agent.anthropic_credentials.Path.home", lambda: home)
        monkeypatch.setenv("HERMES_DISABLE_CLAUDE_CODE_CREDENTIALS", "1")
        monkeypatch.setenv("ANTHROPIC_TOKEN", "explicit-token")

        result = resolve_anthropic_token()
        assert result == "explicit-token"
