"""resolve_anthropic_token honours suppressed_sources for the claude_code source.

Operators can suppress a credential source per provider via the
``suppressed_sources`` key in auth.json (managed by ``hermes auth`` and
read through ``hermes_cli.auth.is_source_suppressed``). Before this fix,
``resolve_anthropic_token`` unconditionally read the local Claude Code
credentials (keychain / ~/.claude files) at priority 3, shadowing the
hermes-managed ``credential_pool`` entry at priority 4 — and worse, the
refresh path could consume Claude Code's single-use rotating refresh
token, invalidating the user's separate Claude Code login.

These tests pin the new contract: when ``claude_code`` is suppressed for
``anthropic``, the Claude Code credential sources are never read; when it
is not suppressed, the historical priority order is unchanged.
"""

import json
from unittest.mock import patch

import pytest

from agent.anthropic_adapter import resolve_anthropic_token

CC_CREDS = {
    "claudeAiOauth": {"accessToken": "cc-token", "expiresAt": 9999999999999}
}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Clear token env vars and point the root-auth fallback at an empty home.

    Without this, a developer's real environment (ANTHROPIC_TOKEN etc.) or
    their real ~/.hermes/auth.json suppression marker would leak into the
    resolution chain under test.
    """
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    with patch("agent.anthropic_adapter.Path.home", return_value=fake_home):
        yield fake_home


class TestClaudeCodeSuppression:
    def test_suppressed_source_is_never_read_and_pool_wins(self):
        """With claude_code suppressed, neither the credential read nor the
        token resolution for Claude Code may run — the credential_pool entry
        becomes the effective source."""
        with patch("hermes_cli.auth.is_source_suppressed", return_value=True), \
             patch("agent.anthropic_adapter.read_claude_code_credentials") as read_cc, \
             patch("agent.anthropic_adapter._resolve_claude_code_token_from_credentials") as resolve_cc, \
             patch("agent.anthropic_adapter._resolve_anthropic_pool_token", return_value="pool-token"):
            assert resolve_anthropic_token() == "pool-token"
        read_cc.assert_not_called()
        resolve_cc.assert_not_called()

    def test_unsuppressed_keeps_claude_code_priority(self):
        """Regression guard: without suppression, the Claude Code credential
        file still outranks the credential_pool entry (historical order)."""
        with patch("hermes_cli.auth.is_source_suppressed", return_value=False), \
             patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=CC_CREDS), \
             patch("agent.anthropic_adapter._resolve_claude_code_token_from_credentials", return_value="cc-token"), \
             patch("agent.anthropic_adapter._resolve_anthropic_pool_token", return_value="pool-token"):
            assert resolve_anthropic_token() == "cc-token"

    def test_root_auth_json_marker_suppresses_for_profiles_without_own_auth(self, _isolated_env):
        """Profiles without their own auth.json inherit the root suppression
        marker (the fallback read of ~/.hermes/auth.json)."""
        auth_dir = _isolated_env / ".hermes"
        auth_dir.mkdir()
        (auth_dir / "auth.json").write_text(
            json.dumps({"suppressed_sources": {"anthropic": ["claude_code"]}})
        )
        with patch("hermes_cli.auth.is_source_suppressed", return_value=False), \
             patch("agent.anthropic_adapter.read_claude_code_credentials") as read_cc, \
             patch("agent.anthropic_adapter._resolve_anthropic_pool_token", return_value="pool-token"):
            assert resolve_anthropic_token() == "pool-token"
        read_cc.assert_not_called()

    def test_suppressed_with_no_pool_falls_back_to_api_key(self, monkeypatch):
        """Suppression removes one source; the rest of the chain is intact
        (ANTHROPIC_API_KEY remains the final fallback)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")
        with patch("hermes_cli.auth.is_source_suppressed", return_value=True), \
             patch("agent.anthropic_adapter.read_claude_code_credentials") as read_cc, \
             patch("agent.anthropic_adapter._resolve_anthropic_pool_token", return_value=None):
            assert resolve_anthropic_token() == "sk-ant-fallback"
        read_cc.assert_not_called()

    def test_env_token_still_wins_when_suppressed(self, monkeypatch):
        """ANTHROPIC_TOKEN (priority 1) is unaffected by suppression — and
        with the Claude Code credentials unread, the refreshable-credential
        preference can no longer swap the env token for a Claude Code one."""
        monkeypatch.setenv("ANTHROPIC_TOKEN", "env-token")
        with patch("hermes_cli.auth.is_source_suppressed", return_value=True), \
             patch("agent.anthropic_adapter.read_claude_code_credentials") as read_cc, \
             patch("agent.anthropic_adapter._resolve_anthropic_pool_token", return_value="pool-token"):
            assert resolve_anthropic_token() == "env-token"
        read_cc.assert_not_called()

    def test_suppression_helper_failure_fails_open(self):
        """If the suppression lookup itself blows up, behaviour must fall
        back to today's (Claude Code readable) rather than breaking auth."""
        with patch("hermes_cli.auth.is_source_suppressed", side_effect=RuntimeError("boom")), \
             patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=CC_CREDS), \
             patch("agent.anthropic_adapter._resolve_claude_code_token_from_credentials", return_value="cc-token"), \
             patch("agent.anthropic_adapter._resolve_anthropic_pool_token", return_value="pool-token"):
            assert resolve_anthropic_token() == "cc-token"
