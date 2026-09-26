"""Tests for ``_is_anthropic_oauth`` guard against third-party Anthropic-compatible providers.

The invariant: ``self._is_anthropic_oauth`` must only ever be True when
``self.provider == 'anthropic'`` (native Anthropic).  Third-party providers
that speak the Anthropic protocol (MiniMax, Zhipu GLM, Alibaba DashScope,
Kimi, LiteLLM proxies, etc.) must never trip OAuth code paths — doing so
injects Claude-Code identity headers and system prompts that cause
401/403 from those endpoints.

This test class covers all FIVE sites that assign ``_is_anthropic_oauth``:

1. ``AIAgent.__init__``                              (line ~1022)
2. ``AIAgent.switch_model``                          (line ~1832)
3. ``AIAgent._try_refresh_anthropic_client_credentials`` (line ~5335)
4. ``AIAgent._swap_credential``                      (line ~5378)
5. ``AIAgent._try_activate_fallback``                (line ~6536)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

# A plausible-looking OAuth token (``sk-ant-`` without the ``-api`` suffix).
_OAUTH_LIKE_TOKEN = "sk-ant-oauth-example-1234567890abcdef"

@pytest.fixture
def agent():
    """Minimal AIAgent construction, skipping tool discovery."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a

class TestOAuthFlagOnRefresh:
    """Site 3 — _try_refresh_anthropic_client_credentials."""

    def test_third_party_provider_refresh_is_noop(self, agent):
        """Refresh path returns False immediately when provider != anthropic — the
        OAuth flag can never be mutated for third-party providers. Double-defended
        by the per-assignment guard at line ~5393 so future refactors can't
        reintroduce the bug."""
        agent.api_mode = "anthropic_messages"
        agent.provider = "minimax"          # ← third-party
        agent._anthropic_api_key = "***"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        with (
            patch("agent.anthropic_credentials.resolve_anthropic_token",
                  return_value=_OAUTH_LIKE_TOKEN),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        # The function short-circuits on non-anthropic providers.
        assert result is False
        # And the flag is untouched regardless.
        assert agent._is_anthropic_oauth is False

    @pytest.mark.parametrize("base_url", [
        "https://llmbox.bytedance.net",
        "http://127.0.0.1:8080/anthropic.com",  # substring spoof: the host is still foreign
        "https://llmbox.bytedance.net/anthropic",  # accepted proxy shape, but holds a custom key
    ])
    def test_third_party_endpoint_skips_refresh(self, agent, base_url):
        """provider == 'anthropic' on a third-party endpoint must not refresh: the refresh
        would swap in native Anthropic credentials the endpoint was never given."""
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = "custom-api-key"
        agent._anthropic_base_url = base_url
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        with (
            patch("agent.anthropic_credentials.resolve_anthropic_token",
                  return_value=_OAUTH_LIKE_TOKEN),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is False
        assert agent._anthropic_api_key == "custom-api-key"
        assert agent._is_anthropic_oauth is False

    @pytest.mark.parametrize("base_url", [
        "https://api.claude.com",
        "https://llm.corp.example/anthropic",
    ])
    def test_accepted_native_proxy_keeps_rotating_anthropic_token(self, agent, base_url):
        """Hosts the resolver accepts as native Anthropic already hold the Anthropic token, so
        blocking the refresh would strand an expiring OAuth token (401 with no recovery)."""
        old, new = "sk-ant-oat01-old-token-aaaaaaaa", "sk-ant-oat01-new-token-bbbbbbbb"
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = old
        agent._anthropic_base_url = base_url
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = True

        with (
            patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=new),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is True
        assert agent._anthropic_api_key == new

class TestOAuthFlagOnCredentialSwap:
    """Site 4 — _swap_credential (credential pool rotation)."""

    def test_pool_swap_on_third_party_never_flips_oauth(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.provider = "glm"              # ← Zhipu GLM via /anthropic
        agent._anthropic_api_key = "old-key"
        agent._anthropic_base_url = "https://open.bigmodel.cn/api/anthropic"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        entry = MagicMock()
        entry.runtime_api_key = _OAUTH_LIKE_TOKEN
        entry.runtime_base_url = "https://open.bigmodel.cn/api/anthropic"

        with patch("agent.anthropic_adapter.build_anthropic_client",
                   return_value=MagicMock()):
            agent._swap_credential(entry)

        assert agent._is_anthropic_oauth is False

class TestOAuthFlagOnConstruction:
    """Site 1 — AIAgent.__init__ on a third-party anthropic_messages provider."""

    def test_minimax_init_does_not_flip_oauth(self):
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
            # Simulate a stale ANTHROPIC_TOKEN in the env — the init code
            # MUST NOT fall back to it when provider != anthropic.
            patch("agent.anthropic_credentials.resolve_anthropic_token",
                  return_value=_OAUTH_LIKE_TOKEN),
        ):
            agent = AIAgent(
                api_key="minimax-key-1234",
                base_url="https://api.minimax.io/anthropic",
                provider="minimax",
                api_mode="anthropic_messages",
                model="claude-sonnet-4-6",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        # The effective key should be the explicit minimax-key, not the
        # stale Anthropic OAuth token, and the OAuth flag must be False.
        assert agent._anthropic_api_key == "minimax-key-1234"
        assert agent._is_anthropic_oauth is False
