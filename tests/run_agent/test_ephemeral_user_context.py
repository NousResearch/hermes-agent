"""Tests for ephemeral user-context injection."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


EPHEMERAL_CONTEXT = "EPHEMERAL USER CONTEXT"


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "web_search tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _mock_response(content="ok"):
    message = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def _plugin_context_hook(hook_name, **kwargs):
    if hook_name == "pre_llm_call":
        return [{"context": EPHEMERAL_CONTEXT}]
    return []


def test_ephemeral_user_context_appends_text_part_to_multimodal_turn():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
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
    agent.client.chat.completions.create.return_value = _mock_response()

    user_message = [
        {"type": "text", "text": "what changed?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]
    with (
        patch.object(agent, "_model_supports_vision", return_value=True),
        patch("hermes_cli.plugins.invoke_hook", side_effect=_plugin_context_hook),
    ):
        agent.run_conversation(user_message)

    sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
    current_user = [msg for msg in sent if msg.get("role") == "user"][-1]

    assert current_user["content"][:-1] == user_message
    assert current_user["content"][-1] == {
        "type": "text",
        "text": EPHEMERAL_CONTEXT,
    }
