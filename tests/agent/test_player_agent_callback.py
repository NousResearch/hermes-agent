"""Player request callback is retained by the public AIAgent constructor."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _tool_defs():
    return [{"type": "function", "function": {
        "name": "web_search", "description": "web_search tool",
        "parameters": {"type": "object", "properties": {}},
    }}]


def test_player_agent_callback_reaches_initialized_agent():
    callback = MagicMock(name="player_agent_callback")
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs()),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        initialized = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True, player_agent_callback=callback,
        )
    assert initialized.player_agent_callback is callback
