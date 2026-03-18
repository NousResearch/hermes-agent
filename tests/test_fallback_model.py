"""Tests for the provider fallback model feature.

Verifies that AIAgent can switch to a configured fallback model/provider
when the primary fails after retries.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_resolve(base_url="https://openrouter.ai/api/v1", api_key="test-key"):
    """Helper to create a mock client for resolve_provider_client."""
    mock_client = MagicMock()
    mock_client.api_key = api_key
    mock_client.base_url = base_url
    return mock_client


# =============================================================================
# _try_activate_fallback()
# =============================================================================

class TestTryActivateFallback:
    def test_returns_false_when_not_configured(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False
        assert agent._fallback_activated is False

    def test_returns_false_for_empty_config(self):
        agent = _make_agent(fallback_model={"provider": "", "model": ""})
        assert agent._try_activate_fallback() is False

    def test_returns_false_for_missing_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4.1"})
        assert agent._try_activate_fallback() is False

    def test_returns_false_for_missing_model(self):
        agent = _make_agent(fallback_model={"provider": "openrouter"})
        assert agent._try_activate_fallback() is False

    def test_activates_openrouter_fallback(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        mock_client = _mock_resolve(
            api_key="sk-or-fallback-key",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "anthropic/claude-sonnet-4"),
        ):
            result = agent._try_activate_fallback()
            assert result is True
            assert agent._fallback_activated is True
            assert agent.model == "anthropic/claude-sonnet-4"
            assert agent.provider == "openrouter"
            assert agent.api_mode == "chat_completions"
            assert agent.client is mock_client

    def test_activates_zai_fallback(self):
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5"},
        )
        mock_client = _mock_resolve(
            api_key="sk-zai-key",
            base_url="https://open.z.ai/api/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "glm-5"),
        ):
            result = agent._try_activate_fallback()
            assert result is True
            assert agent.model == "glm-5"
            assert agent.provider == "zai"
            assert agent.client is mock_client

    def test_activates_kimi_fallback(self):
        agent = _make_agent(
            fallback_model={"provider": "kimi-coding", "model": "kimi-k2.5"},
        )
        mock_client = _mock_resolve(
            api_key="sk-kimi-key",
            base_url="https://api.moonshot.ai/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "kimi-k2.5"),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.model == "kimi-k2.5"
            assert agent.provider == "kimi-coding"

    def test_activates_minimax_fallback(self):
        agent = _make_agent(
            fallback_model={"provider": "minimax", "model": "MiniMax-M2.7"},
        )
        mock_client = _mock_resolve(
            api_key="sk-mm-key",
            base_url="https://api.minimax.io/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "MiniMax-M2.7"),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.model == "MiniMax-M2.7"
            assert agent.provider == "minimax"
            assert agent.client is mock_client

    def test_only_fires_once(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        mock_client = _mock_resolve(
            api_key="sk-or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "anthropic/claude-sonnet-4"),
        ):
            assert agent._try_activate_fallback() is True
            # Second attempt should return False
            assert agent._try_activate_fallback() is False

    def test_returns_false_when_no_api_key(self):
        """Fallback should fail gracefully when the API key env var is unset."""
        agent = _make_agent(
            fallback_model={"provider": "minimax", "model": "MiniMax-M2.7"},
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),
        ):
            assert agent._try_activate_fallback() is False
            assert agent._fallback_activated is False

    def test_custom_base_url(self):
        """Custom base_url in config should override the provider default."""
        agent = _make_agent(
            fallback_model={
                "provider": "custom",
                "model": "my-model",
                "base_url": "http://localhost:8080/v1",
                "api_key_env": "MY_CUSTOM_KEY",
            },
        )
        mock_client = _mock_resolve(
            api_key="custom-secret",
            base_url="http://localhost:8080/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "my-model"),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.client is mock_client
            assert agent.model == "my-model"

    def test_prompt_caching_enabled_for_claude_on_openrouter(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        mock_client = _mock_resolve(
            api_key="sk-or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "anthropic/claude-sonnet-4"),
        ):
            agent._try_activate_fallback()
            assert agent._use_prompt_caching is True

    def test_prompt_caching_disabled_for_non_claude(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "google/gemini-2.5-flash"},
        )
        mock_client = _mock_resolve(
            api_key="sk-or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "google/gemini-2.5-flash"),
        ):
            agent._try_activate_fallback()
            assert agent._use_prompt_caching is False

    def test_prompt_caching_disabled_for_non_openrouter(self):
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5"},
        )
        mock_client = _mock_resolve(
            api_key="sk-zai-key",
            base_url="https://open.z.ai/api/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "glm-5"),
        ):
            agent._try_activate_fallback()
            assert agent._use_prompt_caching is False

    def test_zai_alt_env_var(self):
        """Z.AI should also check Z_AI_API_KEY as fallback env var."""
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5"},
        )
        mock_client = _mock_resolve(
            api_key="sk-alt-key",
            base_url="https://open.z.ai/api/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "glm-5"),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.client is mock_client

    def test_activates_codex_fallback(self):
        """OpenAI Codex fallback should use OAuth credentials and codex_responses mode."""
        agent = _make_agent(
            fallback_model={"provider": "openai-codex", "model": "gpt-5.3-codex"},
        )
        mock_client = _mock_resolve(
            api_key="codex-oauth-token",
            base_url="https://chatgpt.com/backend-api/codex",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "gpt-5.3-codex"),
        ):
            result = agent._try_activate_fallback()
            assert result is True
            assert agent.model == "gpt-5.3-codex"
            assert agent.provider == "openai-codex"
            assert agent.api_mode == "codex_responses"
            assert agent.client is mock_client

    def test_codex_fallback_fails_gracefully_without_credentials(self):
        """Codex fallback should return False if no OAuth credentials available."""
        agent = _make_agent(
            fallback_model={"provider": "openai-codex", "model": "gpt-5.3-codex"},
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),
        ):
            assert agent._try_activate_fallback() is False
            assert agent._fallback_activated is False

    def test_activates_nous_fallback(self):
        """Nous Portal fallback should use OAuth credentials and chat_completions mode."""
        agent = _make_agent(
            fallback_model={"provider": "nous", "model": "nous-hermes-3"},
        )
        mock_client = _mock_resolve(
            api_key="nous-agent-key-abc",
            base_url="https://inference-api.nousresearch.com/v1",
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "nous-hermes-3"),
        ):
            result = agent._try_activate_fallback()
            assert result is True
            assert agent.model == "nous-hermes-3"
            assert agent.provider == "nous"
            assert agent.api_mode == "chat_completions"
            assert agent.client is mock_client

    def test_nous_fallback_fails_gracefully_without_login(self):
        """Nous fallback should return False if not logged in."""
        agent = _make_agent(
            fallback_model={"provider": "nous", "model": "nous-hermes-3"},
        )
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),
        ):
            assert agent._try_activate_fallback() is False
            assert agent._fallback_activated is False


# =============================================================================
# Fallback config init
# =============================================================================

class TestFallbackInit:
    def test_fallback_stored_when_configured(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        assert agent._fallback_model is not None
        assert agent._fallback_model["provider"] == "openrouter"
        assert agent._fallback_activated is False

    def test_fallback_none_when_not_configured(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_model is None
        assert agent._fallback_activated is False

    def test_fallback_none_for_non_dict(self):
        agent = _make_agent(fallback_model="not-a-dict")
        assert agent._fallback_model is None


# =============================================================================
# Provider credential resolution
# =============================================================================

class TestProviderCredentials:
    """Verify that each supported provider resolves via the centralized router."""

    @pytest.mark.parametrize("provider,env_var,base_url_fragment", [
        ("openrouter", "OPENROUTER_API_KEY", "openrouter"),
        ("zai", "ZAI_API_KEY", "z.ai"),
        ("kimi-coding", "KIMI_API_KEY", "moonshot.ai"),
        ("minimax", "MINIMAX_API_KEY", "minimax.io"),
        ("minimax-cn", "MINIMAX_CN_API_KEY", "minimaxi.com"),
    ])
    def test_provider_resolves(self, provider, env_var, base_url_fragment):
        agent = _make_agent(
            fallback_model={"provider": provider, "model": "test-model"},
        )
        mock_client = MagicMock()
        mock_client.api_key = "test-api-key"
        mock_client.base_url = f"https://{base_url_fragment}/v1"
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "test-model"),
        ):
            result = agent._try_activate_fallback()
            assert result is True, f"Failed to activate fallback for {provider}"
            assert agent.client is mock_client
            assert agent.model == "test-model"
            assert agent.provider == provider
