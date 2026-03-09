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
            api_key="test-key-primary",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


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
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-fallback-key"}),
            patch("run_agent.OpenAI") as mock_openai,
        ):
            result = agent._try_activate_fallback()
            assert result is True
            assert agent._fallback_activated is True
            assert agent.model == "anthropic/claude-sonnet-4"
            assert agent.provider == "openrouter"
            assert agent.api_mode == "chat_completions"
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "sk-or-fallback-key"
            assert "openrouter" in call_kwargs["base_url"].lower()
            # OpenRouter should get attribution headers
            assert "default_headers" in call_kwargs

    def test_activates_zai_fallback(self):
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5"},
        )
        with (
            patch.dict("os.environ", {"ZAI_API_KEY": "sk-zai-key"}),
            patch("run_agent.OpenAI") as mock_openai,
        ):
            result = agent._try_activate_fallback()
            assert result is True
            assert agent.model == "glm-5"
            assert agent.provider == "zai"
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "sk-zai-key"
            assert "z.ai" in call_kwargs["base_url"].lower()

    def test_activates_kimi_fallback(self):
        agent = _make_agent(
            fallback_model={"provider": "kimi-coding", "model": "kimi-k2.5"},
        )
        with (
            patch.dict("os.environ", {"KIMI_API_KEY": "sk-kimi-key"}),
            patch("run_agent.OpenAI"),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.model == "kimi-k2.5"
            assert agent.provider == "kimi-coding"

    def test_activates_minimax_fallback(self):
        agent = _make_agent(
            fallback_model={"provider": "minimax", "model": "MiniMax-M2.5"},
        )
        with (
            patch.dict("os.environ", {"MINIMAX_API_KEY": "sk-mm-key"}),
            patch("run_agent.OpenAI") as mock_openai,
        ):
            assert agent._try_activate_fallback() is True
            assert agent.model == "MiniMax-M2.5"
            assert agent.provider == "minimax"
            call_kwargs = mock_openai.call_args[1]
            assert "minimax.io" in call_kwargs["base_url"]

    def test_only_fires_once(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-key"}),
            patch("run_agent.OpenAI"),
        ):
            assert agent._try_activate_fallback() is True
            # Second attempt should return False
            assert agent._try_activate_fallback() is False

    def test_returns_false_when_no_api_key(self):
        """Fallback should fail gracefully when the API key env var is unset."""
        agent = _make_agent(
            fallback_model={"provider": "minimax", "model": "MiniMax-M2.5"},
        )
        # Ensure MINIMAX_API_KEY is not in the environment
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
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
        with (
            patch.dict("os.environ", {"MY_CUSTOM_KEY": "custom-secret"}),
            patch("run_agent.OpenAI") as mock_openai,
        ):
            assert agent._try_activate_fallback() is True
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:8080/v1"
            assert call_kwargs["api_key"] == "custom-secret"

    def test_prompt_caching_enabled_for_claude_on_openrouter(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-key"}),
            patch("run_agent.OpenAI"),
        ):
            agent._try_activate_fallback()
            assert agent._use_prompt_caching is True

    def test_prompt_caching_disabled_for_non_claude(self):
        agent = _make_agent(
            fallback_model={"provider": "openrouter", "model": "google/gemini-2.5-flash"},
        )
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-key"}),
            patch("run_agent.OpenAI"),
        ):
            agent._try_activate_fallback()
            assert agent._use_prompt_caching is False

    def test_prompt_caching_disabled_for_non_openrouter(self):
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5"},
        )
        with (
            patch.dict("os.environ", {"ZAI_API_KEY": "sk-zai-key"}),
            patch("run_agent.OpenAI"),
        ):
            agent._try_activate_fallback()
            assert agent._use_prompt_caching is False

    def test_zai_alt_env_var(self):
        """Z.AI should also check Z_AI_API_KEY as fallback env var."""
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5"},
        )
        with (
            patch.dict("os.environ", {"Z_AI_API_KEY": "sk-alt-key"}),
            patch("run_agent.OpenAI") as mock_openai,
        ):
            assert agent._try_activate_fallback() is True
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "sk-alt-key"


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
    """Verify that each supported provider resolves its API key correctly."""

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
        with (
            patch.dict("os.environ", {env_var: "test-key-123"}),
            patch("run_agent.OpenAI") as mock_openai,
        ):
            result = agent._try_activate_fallback()
            assert result is True, f"Failed to activate fallback for {provider}"
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "test-key-123"
            assert base_url_fragment in call_kwargs["base_url"].lower()
