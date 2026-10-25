"""Route-aware Conversation estimates; regression for #134915."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_breakdown import compute_session_context_breakdown
from agent.model_metadata import estimate_messages_tokens_rough


@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("reasoning_key", ["reasoning", "reasoning_content"])
def test_conversation_category_respects_stale_reasoning_policy(echo, reasoning_key):
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="deepseek" if echo else "custom",
        model="deepseek-chat" if echo else "local-test", base_url="http://localhost:9999/v1",
        tools=[], _memory_store=None, _usage_anchor=None, _turn_base_usage_anchor=None,
        context_compressor=SimpleNamespace(context_length=200_000, last_prompt_tokens=0),
    )
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer", reasoning_key: "old thought " * 1000},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "latest", reasoning_key: "live thought " * 50},
        {"role": "user", "content": "follow up"},
    ]
    original = deepcopy(history)
    expected_history = deepcopy(history)
    if not echo:
        del expected_history[1][reasoning_key]
    with patch("agent.system_prompt.build_system_prompt_parts", return_value={}):
        data = compute_session_context_breakdown(agent, history)
    conversation = next(row["tokens"] for row in data["categories"] if row["id"] == "conversation")
    assert conversation == estimate_messages_tokens_rough(expected_history)
    assert data["estimated_total"] == conversation
    assert data["context_used"] == conversation
    assert data["context_source"] == "local_estimate"
    assert history == original


def test_native_carriers_and_provider_usage_are_not_reinterpreted():
    """Native carrier projection is separate; this fix only selects stale-text policy."""
    from agent.usage_anchor import capture_usage_anchor

    history = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning": "stored " * 100,
         "anthropic_content_blocks": [
             {"type": "thinking", "thinking": "signed thought", "signature": "opaque"},
             {"type": "text", "text": "answer"},
         ]},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "latest", "reasoning": "new thought"},
    ]
    original = deepcopy(history)
    agent = SimpleNamespace(
        api_mode="anthropic_messages", provider="anthropic", model="claude-sonnet-4-6",
        base_url="https://api.anthropic.com", tools=[], _memory_store=None,
        _turn_base_usage_anchor=capture_usage_anchor(1000, 20, history),
        context_compressor=SimpleNamespace(context_length=200_000, last_prompt_tokens=0),
    )
    with patch("agent.system_prompt.build_system_prompt_parts", return_value={}):
        data = compute_session_context_breakdown(agent, history)
    conversation = next(row["tokens"] for row in data["categories"] if row["id"] == "conversation")
    assert conversation == estimate_messages_tokens_rough(history)
    assert data["context_used"] == 1020
    assert data["context_source"] == "provider_usage"
    assert history == original
