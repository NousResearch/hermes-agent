"""Regression coverage for auxiliary compression feasibility preflight timing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _response():
    message = SimpleNamespace(
        content="ok", tool_calls=None, reasoning_content=None, reasoning=None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def test_preflight_checks_auxiliary_feasibility_before_should_compress():
    """A near-boundary preflight must probe before it consults the trigger."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = True
    agent.context_compressor.context_length = 200_000
    agent.context_compressor.threshold_tokens = 150_000
    agent._compression_feasibility_checked = False
    agent.client.chat.completions.create.return_value = _response()
    history = [
        {"role": "user", "content": "x" * 200_000},
        {"role": "assistant", "content": "y" * 200_000},
    ]

    def _check_before_decision(_tokens):
        assert agent._compression_feasibility_checked is True
        return False

    with (
        patch("agent.turn_context.ensure_compression_feasibility_checked") as ensure,
        patch.object(
            agent.context_compressor,
            "should_compress",
            side_effect=_check_before_decision,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        ensure.side_effect = lambda subject: setattr(
            subject, "_compression_feasibility_checked", True
        )
        result = agent.run_conversation("hello", conversation_history=history)

    ensure.assert_called_once_with(agent)
    assert result["completed"] is True
