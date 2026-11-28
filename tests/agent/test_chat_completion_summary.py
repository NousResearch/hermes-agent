"""Regression for #66587: summaries replay Gemini tool calls without changing history."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import _iteration_summary_api_messages
from run_agent import AIAgent


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        instance.client = MagicMock()
        return instance


@pytest.mark.parametrize(
    ("provider", "model", "aggregator_model", "needs_signature"),
    [
        ("custom", "google/gemini-3-pro-preview", None, True),
        ("moa", "moa/example", "google/gemini-3-pro-preview", True),
        ("moa", "moa/gemini-preset", "mistralai/mistral-large", False),
    ],
)
def test_summary_replays_unsigned_tool_calls_for_gemini(agent, provider, model, aggregator_model, needs_signature):
    agent.provider = provider
    agent.model = model
    agent.client.last_aggregator_slot = {"model": aggregator_model} if aggregator_model else None
    messages = [
        {"role": "user", "content": "Do the work"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Done"},
    ]
    original = deepcopy(messages)

    api_messages = _iteration_summary_api_messages(agent, messages)
    kwargs = agent._build_api_kwargs(api_messages)
    call = next(m for m in kwargs["messages"] if m.get("tool_calls"))["tool_calls"][0]

    assert kwargs["model"] == model
    if needs_signature:
        assert call["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator"
    else:
        assert "extra_content" not in call
    assert messages == original
