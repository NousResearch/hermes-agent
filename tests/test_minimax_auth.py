"""Tests for _uses_native_anthropic_auth() -- MiniMax and third-party providers
must not have their API key overwritten by resolve_anthropic_token()."""
from unittest.mock import patch, MagicMock
from run_agent import AIAgent


def _make_agent(provider, base_url, api_mode="anthropic_messages"):
    """Instantiate AIAgent with minimal args, bypassing full init."""
    agent = AIAgent.__new__(AIAgent)
    agent.provider = provider
    agent.base_url = base_url
    agent._anthropic_base_url = base_url
    agent.api_mode = api_mode
    return agent


class TestUsesNativeAnthropicAuth:
    def test_anthropic_provider_is_native(self):
        agent = _make_agent("anthropic", "https://api.anthropic.com")
        assert agent._uses_native_anthropic_auth() is True

    def test_api_anthropic_com_url_is_native(self):
        agent = _make_agent("custom", "https://api.anthropic.com/v1")
        assert agent._uses_native_anthropic_auth() is True

    def test_minimax_is_not_native(self):
        agent = _make_agent("minimax", "https://api.minimax.io/anthropic")
        assert agent._uses_native_anthropic_auth() is False

    def test_minimax_cn_is_not_native(self):
        agent = _make_agent("minimax-cn", "https://api.minimax.chat/anthropic")
        assert agent._uses_native_anthropic_auth() is False

    def test_dashscope_is_not_native(self):
        agent = _make_agent("alibaba", "https://dashscope.aliyuncs.com/anthropic")
        assert agent._uses_native_anthropic_auth() is False

    def test_custom_anthropic_compatible_is_not_native(self):
        agent = _make_agent("custom", "https://my-server.example.com/anthropic")
        assert agent._uses_native_anthropic_auth() is False


class TestRefreshCredentialsSkipsNonNative:
    def test_minimax_refresh_does_not_call_resolve_anthropic_token(self):
        agent = _make_agent("minimax", "https://api.minimax.io/anthropic")
        agent._anthropic_api_key = "minimax-key-123"
        with patch("agent.anthropic_adapter.resolve_anthropic_token") as mock_resolve:
            result = agent._try_refresh_anthropic_client_credentials()
        mock_resolve.assert_not_called()
        assert result is False

    def test_native_anthropic_refresh_calls_resolve_token_when_key_differs(self):
        agent = _make_agent("anthropic", "https://api.anthropic.com")
        agent._anthropic_api_key = "old-token"
        agent._anthropic_base_url = "https://api.anthropic.com"
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="new-token") as mock_resolve, \
             patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()):
            agent._try_refresh_anthropic_client_credentials()
        mock_resolve.assert_called_once()
