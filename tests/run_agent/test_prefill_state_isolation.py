"""Regression coverage for per-agent ownership of API-time prefill state."""

from unittest.mock import MagicMock, patch

from agent.turn_recovery import _recover_unicode_encode_error
from run_agent import AIAgent


def test_unicode_recovery_isolated_across_agents_sharing_prefill_input():
    """In-place recovery in one agent must not rewrite its caller or a sibling agent."""
    source = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "parent\ud800prefill"}],
            "metadata": {"labels": ["shared"]},
        }
    ]

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch("agent.agent_init.fetch_model_metadata", return_value={}),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agents = [
            AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                prefill_messages=source,
            )
            for _ in range(2)
        ]

    agents[0]._unicode_sanitization_passes = 0
    recovered, _ = _recover_unicode_encode_error(
        agents[0],
        UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed"),
        messages=[],
        api_messages=[],
        api_kwargs={},
        active_system_prompt="",
    )

    assert recovered is True
    assert agents[0].prefill_messages[0]["content"][0]["text"] == "parent\ufffdprefill"
    assert source[0]["content"][0]["text"] == "parent\ud800prefill"
    assert agents[1].prefill_messages == source
    assert agents[0].prefill_messages is not source
    assert agents[0].prefill_messages[0]["metadata"] is not agents[1].prefill_messages[0]["metadata"]
