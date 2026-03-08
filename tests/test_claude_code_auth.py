"""Tests for Claude Code OAuth credential discovery and fallback provider.

Tests the auto-detection of Claude Code's ~/.claude/.credentials.json
for the Anthropic provider, and the Anthropic -> OpenRouter fallback chain.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_CREDENTIALS = {
    "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-test-token-value",
        "refreshToken": "sk-ant-rt01-test-refresh",
        "expiresAt": int((time.time() + 86400) * 1000),  # 24h from now
        "scopes": ["user:inference", "user:profile"],
        "subscriptionType": "pro",
        "rateLimitTier": "default_claude_ai",
    }
}

EXPIRED_CREDENTIALS = {
    "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-expired-token",
        "refreshToken": "sk-ant-rt01-expired-refresh",
        "expiresAt": int((time.time() - 3600) * 1000),  # 1h ago
        "scopes": ["user:inference"],
        "subscriptionType": "pro",
    }
}


@pytest.fixture
def mock_creds_file(tmp_path):
    """Create a temporary credentials file."""
    creds_path = tmp_path / ".claude" / ".credentials.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(data):
        creds_path.write_text(json.dumps(data))
        return creds_path

    return _write


# ---------------------------------------------------------------------------
# _read_claude_code_credentials
# ---------------------------------------------------------------------------

class TestReadClaudeCodeCredentials:
    def test_valid_credentials(self, mock_creds_file):
        creds_path = mock_creds_file(VALID_CREDENTIALS)
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path):
            from hermes_cli.auth import _read_claude_code_credentials
            result = _read_claude_code_credentials()
            assert result is not None
            assert result["accessToken"] == "sk-ant-oat01-test-token-value"
            assert result["subscriptionType"] == "pro"

    def test_missing_file(self, tmp_path):
        missing = tmp_path / ".claude" / ".credentials.json"
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", missing):
            from hermes_cli.auth import _read_claude_code_credentials
            assert _read_claude_code_credentials() is None

    def test_malformed_json(self, mock_creds_file, tmp_path):
        creds_path = tmp_path / ".claude" / ".credentials.json"
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        creds_path.write_text("not json{{{")
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path):
            from hermes_cli.auth import _read_claude_code_credentials
            assert _read_claude_code_credentials() is None

    def test_missing_oauth_key(self, mock_creds_file):
        creds_path = mock_creds_file({"someOtherKey": {}})
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path):
            from hermes_cli.auth import _read_claude_code_credentials
            assert _read_claude_code_credentials() is None

    def test_empty_access_token(self, mock_creds_file):
        data = {"claudeAiOauth": {"accessToken": "", "expiresAt": 9999999999999}}
        creds_path = mock_creds_file(data)
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path):
            from hermes_cli.auth import _read_claude_code_credentials
            assert _read_claude_code_credentials() is None

    def test_oauth_not_dict(self, mock_creds_file):
        data = {"claudeAiOauth": "not-a-dict"}
        creds_path = mock_creds_file(data)
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path):
            from hermes_cli.auth import _read_claude_code_credentials
            assert _read_claude_code_credentials() is None


# ---------------------------------------------------------------------------
# resolve_anthropic_claude_code_credentials
# ---------------------------------------------------------------------------

class TestResolveClaudeCodeCredentials:
    def test_valid_returns_creds(self, mock_creds_file):
        creds_path = mock_creds_file(VALID_CREDENTIALS)
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path):
            from hermes_cli.auth import resolve_anthropic_claude_code_credentials
            result = resolve_anthropic_claude_code_credentials()
            assert result["provider"] == "anthropic"
            assert result["api_mode"] == "anthropic_messages"
            assert result["api_key"] == "sk-ant-oat01-test-token-value"
            assert result["source"] == "claude-code"
            assert result["subscription_type"] == "pro"
            assert result["base_url"] == "https://api.anthropic.com"

    def test_expired_raises(self, mock_creds_file):
        creds_path = mock_creds_file(EXPIRED_CREDENTIALS)
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path):
            from hermes_cli.auth import resolve_anthropic_claude_code_credentials, AuthError
            with pytest.raises(AuthError, match="expired"):
                resolve_anthropic_claude_code_credentials()

    def test_missing_raises(self, tmp_path):
        missing = tmp_path / ".claude" / ".credentials.json"
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", missing):
            from hermes_cli.auth import resolve_anthropic_claude_code_credentials, AuthError
            with pytest.raises(AuthError, match="No Claude Code credentials"):
                resolve_anthropic_claude_code_credentials()

    def test_custom_base_url_from_env(self, mock_creds_file):
        creds_path = mock_creds_file(VALID_CREDENTIALS)
        with (
            patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path),
            patch.dict("os.environ", {"ANTHROPIC_BASE_URL": "https://custom.example.com/v1/"}),
        ):
            from hermes_cli.auth import resolve_anthropic_claude_code_credentials
            result = resolve_anthropic_claude_code_credentials()
            assert result["base_url"] == "https://custom.example.com/v1"


# ---------------------------------------------------------------------------
# get_claude_code_auth_status
# ---------------------------------------------------------------------------

class TestClaudeCodeAuthStatus:
    def test_logged_in(self, mock_creds_file):
        creds_path = mock_creds_file(VALID_CREDENTIALS)
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path):
            from hermes_cli.auth import get_claude_code_auth_status
            status = get_claude_code_auth_status()
            assert status["logged_in"] is True
            assert status["source"] == "claude-code"
            assert status["subscription_type"] == "pro"
            assert status["is_expired"] is False

    def test_expired(self, mock_creds_file):
        creds_path = mock_creds_file(EXPIRED_CREDENTIALS)
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path):
            from hermes_cli.auth import get_claude_code_auth_status
            status = get_claude_code_auth_status()
            assert status["logged_in"] is False
            assert status["is_expired"] is True

    def test_no_credentials(self, tmp_path):
        missing = tmp_path / ".claude" / ".credentials.json"
        with patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", missing):
            from hermes_cli.auth import get_claude_code_auth_status
            status = get_claude_code_auth_status()
            assert status["logged_in"] is False


# ---------------------------------------------------------------------------
# resolve_provider auto-detection
# ---------------------------------------------------------------------------

class TestResolveProviderClaudeCode:
    def test_auto_detects_claude_code(self, mock_creds_file):
        """When Claude Code creds are present and valid, auto-resolve to anthropic."""
        creds_path = mock_creds_file(VALID_CREDENTIALS)
        with (
            patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path),
            patch("hermes_cli.auth._load_auth_store", return_value={}),
            patch.dict("os.environ", {}, clear=False),
        ):
            # Remove any env vars that would cause earlier resolution
            import os
            env_backup = {}
            for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY"):
                if k in os.environ:
                    env_backup[k] = os.environ.pop(k)
            try:
                from hermes_cli.auth import resolve_provider
                result = resolve_provider()
                assert result == "anthropic"
            finally:
                os.environ.update(env_backup)

    def test_expired_creds_not_detected(self, mock_creds_file):
        """Expired Claude Code creds should not auto-resolve."""
        creds_path = mock_creds_file(EXPIRED_CREDENTIALS)
        with (
            patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path),
            patch("hermes_cli.auth._load_auth_store", return_value={}),
        ):
            import os
            env_backup = {}
            for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY"):
                if k in os.environ:
                    env_backup[k] = os.environ.pop(k)
            try:
                from hermes_cli.auth import resolve_provider
                result = resolve_provider()
                # Should fall through to openrouter (default)
                assert result == "openrouter"
            finally:
                os.environ.update(env_backup)

    def test_claude_alias(self):
        """'claude' and 'claude-code' should resolve to 'anthropic'."""
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("claude") == "anthropic"
        assert resolve_provider("claude-code") == "anthropic"


# ---------------------------------------------------------------------------
# runtime_provider resolution
# ---------------------------------------------------------------------------

class TestRuntimeProviderClaudeCode:
    def test_claude_code_creds_used(self, mock_creds_file):
        """runtime_provider should use Claude Code creds when available."""
        creds_path = mock_creds_file(VALID_CREDENTIALS)
        with (
            patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path),
            patch("hermes_cli.runtime_provider.resolve_provider", return_value="anthropic"),
        ):
            from hermes_cli.runtime_provider import resolve_runtime_provider
            result = resolve_runtime_provider(requested="anthropic")
            assert result["provider"] == "anthropic"
            assert result["api_mode"] == "anthropic_messages"
            assert result["source"] == "claude-code"
            assert result["api_key"] == "sk-ant-oat01-test-token-value"

    def test_falls_back_to_env_var(self, tmp_path):
        """When no Claude Code creds, fall back to env var."""
        missing = tmp_path / ".claude" / ".credentials.json"
        with (
            patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", missing),
            patch("hermes_cli.runtime_provider.resolve_provider", return_value="anthropic"),
            patch.dict("os.environ", {"ANTHROPIC_TOKEN": "sk-ant-api-test-key"}),
        ):
            from hermes_cli.runtime_provider import resolve_runtime_provider
            result = resolve_runtime_provider(requested="anthropic")
            assert result["provider"] == "anthropic"
            assert result["api_mode"] == "anthropic_messages"
            assert result["source"] == "ANTHROPIC_TOKEN"


# ---------------------------------------------------------------------------
# get_auth_status dispatcher
# ---------------------------------------------------------------------------

class TestAuthStatusDispatcher:
    def test_anthropic_prefers_claude_code(self, mock_creds_file):
        """get_auth_status('anthropic') should return Claude Code status if available."""
        creds_path = mock_creds_file(VALID_CREDENTIALS)
        with (
            patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", creds_path),
            patch("hermes_cli.auth.get_active_provider", return_value="anthropic"),
        ):
            from hermes_cli.auth import get_auth_status
            status = get_auth_status("anthropic")
            assert status["logged_in"] is True
            assert status["source"] == "claude-code"

    def test_anthropic_falls_back_to_env(self, tmp_path):
        """get_auth_status('anthropic') falls back to api_key status."""
        missing = tmp_path / ".claude" / ".credentials.json"
        with (
            patch("hermes_cli.auth.DEFAULT_CLAUDE_CREDENTIALS_PATH", missing),
            patch("hermes_cli.auth.get_active_provider", return_value="anthropic"),
            patch.dict("os.environ", {"ANTHROPIC_TOKEN": "sk-test"}),
        ):
            from hermes_cli.auth import get_auth_status
            status = get_auth_status("anthropic")
            assert status.get("configured") is True
            assert status.get("provider") == "anthropic"


# ---------------------------------------------------------------------------
# Fallback provider: Anthropic -> OpenRouter
# ---------------------------------------------------------------------------

class TestFallbackProvider:
    """Test the provider fallback chain mechanism in AIAgent."""

    def _make_agent(self, **extra_env):
        """Create an AIAgent in anthropic_messages mode with mocked clients."""
        import sys

        env = {
            "ANTHROPIC_TOKEN": "sk-ant-oat01-test-token",
            "OPENROUTER_API_KEY": "sk-or-test-key",
        }
        env.update(extra_env)

        # Mock the anthropic SDK if not installed
        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = MagicMock()
        needs_mock = "anthropic" not in sys.modules
        if needs_mock:
            sys.modules["anthropic"] = mock_anthropic_mod

        # Also mock httpx.Timeout if needed
        mock_httpx = MagicMock()
        needs_httpx_mock = "httpx" not in sys.modules
        if needs_httpx_mock:
            sys.modules["httpx"] = mock_httpx

        try:
            # Force re-import of adapter to pick up mocked anthropic
            if "agent.anthropic_adapter" in sys.modules:
                del sys.modules["agent.anthropic_adapter"]

            with (
                patch("run_agent.get_tool_definitions", return_value=[]),
                patch("run_agent.check_toolset_requirements", return_value={}),
                patch.dict("os.environ", env),
            ):
                from run_agent import AIAgent
                agent = AIAgent(
                    api_key="sk-ant-oat01-test-token",
                    model="claude-opus-4-20250514",
                    provider="anthropic",
                    api_mode="anthropic_messages",
                    quiet_mode=True,
                    skip_context_files=True,
                    skip_memory=True,
                )
                return agent
        finally:
            if needs_mock:
                sys.modules.pop("anthropic", None)
            if needs_httpx_mock:
                sys.modules.pop("httpx", None)

    def test_fallback_chain_auto_built_with_openrouter_key(self):
        agent = self._make_agent()
        assert agent._fallback_chain.has_fallbacks() is True
        assert agent._fallback_activated is False
        assert agent.api_mode == "anthropic_messages"
        assert agent._fallback_chain.entries[0].provider == "openrouter"

    def test_fallback_chain_empty_without_key(self):
        agent = self._make_agent(OPENROUTER_API_KEY="")
        assert agent._fallback_chain.has_fallbacks() is False

    def test_activate_fallback(self):
        from agent.fallback_chain import FallbackEntry
        agent = self._make_agent()
        entry = agent._fallback_chain.entries[0]
        with patch("run_agent.OpenAI") as mock_openai:
            result = agent._activate_fallback(entry)
            assert result is True
            assert agent._fallback_activated is True
            assert agent.api_mode == "chat_completions"
            assert agent.provider == "openrouter"
            assert agent.model == "anthropic/claude-opus-4-20250514"
            mock_openai.assert_called_once()

    def test_chain_exhaustion(self):
        """After activating the only entry, chain should be exhaustible."""
        from agent.fallback_chain import FallbackEntry
        agent = self._make_agent()
        entry = agent._fallback_chain.entries[0]
        with patch("run_agent.OpenAI"):
            assert agent._activate_fallback(entry) is True
            # Manually mark it failed to simulate exhaustion
            agent._fallback_chain.mark_failed(entry)
            assert agent._fallback_chain.is_exhausted() is True

    def test_model_remapping_bare_names(self):
        """Bare model names get anthropic/ prefix when falling back to OpenRouter with no entry model."""
        from agent.fallback_chain import FallbackEntry, FallbackChain
        agent = self._make_agent()
        agent.model = "claude-sonnet-4-20250514"
        # Create a chain entry WITHOUT a model override — triggers remapping
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter"),  # no model set
        ])
        agent._fallback_chain = chain
        entry = chain.entries[0]
        with patch("run_agent.OpenAI"):
            agent._activate_fallback(entry)
            assert agent.model == "anthropic/claude-sonnet-4-20250514"

    def test_already_prefixed_model_not_doubled(self):
        """Models already prefixed shouldn't get double-prefixed."""
        from agent.fallback_chain import FallbackEntry, FallbackChain
        agent = self._make_agent()
        agent.model = "anthropic/claude-opus-4-20250514"
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter"),  # no model override
        ])
        agent._fallback_chain = chain
        entry = chain.entries[0]
        with patch("run_agent.OpenAI"):
            agent._activate_fallback(entry)
            assert agent.model == "anthropic/claude-opus-4-20250514"

    def test_entry_model_override(self):
        """Entry-specified model should override the agent's current model."""
        from agent.fallback_chain import FallbackEntry, FallbackChain
        agent = self._make_agent()
        # Replace chain with custom entry that has a model override
        chain = FallbackChain(entries=[
            FallbackEntry(provider="lmstudio", base_url="http://localhost:1234/v1", model="qwen3-30b-a3b"),
        ])
        agent._fallback_chain = chain
        entry = chain.entries[0]
        with patch("run_agent.OpenAI"):
            agent._activate_fallback(entry)
            assert agent.model == "qwen3-30b-a3b"
            assert agent.provider == "lmstudio"
