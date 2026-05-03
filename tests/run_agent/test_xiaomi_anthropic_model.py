from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def test_xiaomi_anthropic_transport_preserves_dotted_model_id():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://token-plan-cn.xiaomimimo.com/anthropic",
            provider="xiaomi",
            api_mode="anthropic_messages",
            model="mimo-v2.5-pro",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "ping"}])

    assert agent._anthropic_preserve_dots() is True
    assert kwargs["model"] == "mimo-v2.5-pro"
