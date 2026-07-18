"""Config wiring for iteration-summary rollover."""

from unittest.mock import patch

from run_agent import AIAgent


def _agent_with_config(agent_config):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "hermes_cli.config.load_config",
            return_value={"agent": agent_config},
        ),
    ):
        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )


def test_iteration_summary_rollover_is_opt_in_by_default():
    agent = _agent_with_config({})
    assert getattr(agent, "_summarize_and_continue_on_limit") is False


def test_iteration_summary_rollover_loads_enabled_config():
    agent = _agent_with_config({"summarize_and_continue_on_limit": True})
    assert getattr(agent, "_summarize_and_continue_on_limit") is True
