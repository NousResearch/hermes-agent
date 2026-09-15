"""Hard --model pin must not be swapped by the conversation-loop provider ladder.

Vector 2 of #100437: resolve-time is #105183; this is try_activate_fallback
walking fallback_providers onto a different model (Provider unreachable /
Model fallback:). Same-model provider swaps still run.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(fallback_model):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="pinned-model-x",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://sink.example.invalid/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


class TestPinnedRuntimeFallback:
    def test_different_model_entry_is_skipped_and_pin_holds(self):
        agent = _make_agent(
            fallback_model=[{"provider": "sinkok", "model": "sink-model-y"}],
        )
        agent._fallback_pin_model = "pinned-model-x"
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "sink-model-y"),
        ) as resolve:
            activated = agent._try_activate_fallback()
        assert activated is False
        assert agent.model == "pinned-model-x"
        resolve.assert_not_called()

    def test_same_model_provider_swap_still_runs(self):
        agent = _make_agent(
            fallback_model=[{"provider": "sinkok", "model": "pinned-model-x"}],
        )
        agent._fallback_pin_model = "pinned-model-x"
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "pinned-model-x"),
        ):
            activated = agent._try_activate_fallback()
        assert activated is True
        assert agent.model == "pinned-model-x"
        assert agent.provider == "sinkok"

    def test_deepseek_chat_alias_does_not_match_openrouter_deepseek_chat(self):
        """Primary deepseek-chat is deepseek-flash; OpenRouter deepseek-chat is another slug."""
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                provider="deepseek",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[{"provider": "openrouter", "model": "deepseek-chat"}],
            )
            agent.client = MagicMock()
        agent._fallback_pin_model = agent.model
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "deepseek/deepseek-chat"),
        ) as resolve:
            activated = agent._try_activate_fallback()
        assert agent.model == "deepseek-flash"
        assert activated is False
        resolve.assert_not_called()

    def test_vendor_prefixed_same_model_swap_still_runs(self):
        """gpt-5.4 pin vs openai/gpt-5.4 on openai-codex is the same identity."""
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                provider="openai",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[{"provider": "openai-codex", "model": "openai/gpt-5.4"}],
            )
            agent.client = MagicMock()
        agent._fallback_pin_model = agent.model
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url="https://chatgpt.com/backend-api"), "gpt-5.4"),
        ):
            activated = agent._try_activate_fallback()
        assert activated is True
        assert agent.model == "gpt-5.4"
