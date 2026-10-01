from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


@pytest.fixture
def agent():
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


def test_anthropic_token_refresh_reaches_the_compressor(agent):
    old, new = "sk-ant-oat01-old", "sk-ant-oat01-new"
    agent.api_mode, agent.provider = "anthropic_messages", "anthropic"
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = agent.api_key = agent.context_compressor.api_key = old

    with (
        patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=new),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        assert agent._try_refresh_anthropic_client_credentials() is True

    assert agent._anthropic_api_key == new
    assert agent.api_key == new
    assert agent.context_compressor.api_key == new


def test_openai_style_credential_refresh_reaches_the_compressor(agent):
    agent.context_compressor.api_key = agent.api_key

    with patch.object(agent, "_replace_primary_openai_client", return_value=True):
        assert agent._adopt_openai_credentials("fresh-key", "https://openrouter.ai/api/v1", reason="test") is True

    assert agent.context_compressor.api_key == "fresh-key"
