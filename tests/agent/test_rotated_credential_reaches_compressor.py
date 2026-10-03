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


def test_compression_picks_up_a_token_rotated_since_the_last_request(agent):
    from agent.conversation_compression import compress_context

    seen = []

    class _Compressor:
        api_key = "stale"
        _last_compress_aborted = False
        _last_summary_error = None
        compression_count = 1
        _last_compression_made_progress = True
        _last_summary_fallback_used = False
        last_compression_rough_tokens = last_prompt_tokens = last_completion_tokens = 0
        awaiting_real_usage_after_compression = False

        def compress(self, _messages, **_kwargs):
            seen.append(self.api_key)
            return [{"role": "user", "content": "[summary]"}, {"role": "assistant", "content": "tail"}]

    agent.api_mode, agent.provider = "anthropic_messages", "anthropic"
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = agent.api_key = "stale"
    agent.context_compressor = _Compressor()
    agent._compression_feasibility_checked = True
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

    with (
        patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="rotated"),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        compress_context(agent, messages, "system", approx_tokens=100_000, force=True)

    assert seen == ["rotated"]
