"""Honcho context stays bounded and excludes hidden reasoning at both prompt boundaries."""

import json
from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider


def test_first_turn_context_cleans_every_field_and_compacts_cards():
    provider = HonchoMemoryProvider()
    context = {
        "summary": "Visible summary <think>private plan</think> remains",
        "representation": "Uses Python <think>secret</think> daily. " * 200,
        "card": "\n".join(["Fact 1 <thinking>hidden</thinking>"] + [f"Fact {i}" for i in range(2, 12)]),
        "ai_representation": "Helpful <reasoning>hidden AI plan</reasoning> assistant",
        "ai_card": ["Identity 1 <think>hidden identity</think>", "Identity 2"],
    }

    result = provider._format_first_turn_context(context)

    assert "remains" in result
    assert "private plan" not in result and "secret" not in result
    assert "hidden" not in result
    assert "## User Peer Card\n- Fact 1" in result
    assert "and 3 more facts" in result
    assert "## AI Identity Card\n- Identity 1" in result
    assert len(result.split("## User Representation\n", 1)[1].split("\n\n", 1)[0]) <= 701


def test_context_tool_cleans_summary_representation_card_and_messages():
    provider = HonchoMemoryProvider()
    provider._manager = SimpleNamespace(get_session_context=lambda *a, **k: {
        "summary": "Hello <think>summary secret</think> world",
        "representation": "Rep <think>rep secret</think> visible",
        "card": ["alpha <think>card secret</think>", "beta"],
        "recent_messages": [
            {"role": "assistant", "content": "<think>message secret</think>shown"},
            {"role": "user", "content": "plain"},
        ],
    })
    provider._session_key = "test-session"

    result = json.loads(provider._tool_context({}))["result"]

    assert "secret" not in result
    assert "## Summary\nworld" in result
    assert "## Representation\nRep visible" in result
    assert "## Card\n- alpha\n- beta" in result
    assert "[assistant] shown" in result
