"""Display policy stays shared with turn estimates and last-response deltas."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_breakdown import compute_session_context_breakdown
from agent.model_metadata import estimate_messages_tokens_rough
from agent.turn_context import _agent_stale_thinking_on_wire
from agent.usage_anchor import capture_usage_anchor


def _history():
    return [
        {"role": "user", "content": "priced prompt"},
        {"role": "assistant", "content": "priced reply", "reasoning": "covered " * 100},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "old", "reasoning": "stale " * 1000},
        {"role": "user", "content": "next again"},
        {"role": "assistant", "content": "latest", "reasoning": "live " * 20},
        {"role": "user", "content": "follow up"},
    ]


def _agent(echo=False):
    return SimpleNamespace(
        api_mode="chat_completions", provider="deepseek" if echo else "custom",
        model="deepseek-chat" if echo else "local-test", base_url="http://localhost:9999/v1",
        tools=[], _memory_store=None, _turn_base_usage_anchor=None, _usage_anchor=None,
        context_compressor=SimpleNamespace(context_length=200_000, last_prompt_tokens=0),
    )


def _breakdown(agent, history):
    with patch("agent.system_prompt.build_system_prompt_parts", return_value={}):
        return compute_session_context_breakdown(agent, history)


def _conversation(data):
    return next(row["tokens"] for row in data["categories"] if row["id"] == "conversation")


@pytest.mark.parametrize("echo", [False, True])
def test_last_response_anchor_delta_uses_active_route(echo):
    history = _history()
    agent = _agent(echo)
    agent._usage_anchor = capture_usage_anchor(1000, 30, history[:1])
    original = deepcopy((history, agent._usage_anchor))
    expected_delta = deepcopy(history[2:])  # priced reply already covered by completion usage
    if not echo:
        del expected_delta[1]["reasoning"]
    data = _breakdown(agent, history)
    assert data["context_used"] == 1030 + estimate_messages_tokens_rough(expected_delta)
    assert data["context_source"] == "provider_usage_plus_estimate"
    assert data["context_estimated"] is True
    assert (history, agent._usage_anchor) == original


@pytest.mark.parametrize("failure", ["predicate", "route_attribute"])
def test_resolution_failure_conservatively_charges_category_and_delta(failure):
    history = _history()
    agent = _agent()
    if failure == "route_attribute":
        class BrokenRoute(SimpleNamespace):
            @property
            def provider(self):
                raise RuntimeError("route unavailable")
        fields = vars(agent).copy()
        fields.pop("provider")
        agent = BrokenRoute(**fields)
    agent._usage_anchor = capture_usage_anchor(1000, 30, history[:1])
    original = deepcopy(history)
    context = (
        patch("agent.message_sanitization.stale_thinking_reaches_wire", side_effect=RuntimeError("policy unavailable"))
        if failure == "predicate" else nullcontext()
    )
    with context:
        assert _agent_stale_thinking_on_wire(agent) is True
        try:
            data = _breakdown(agent, history)
        except RuntimeError as exc:
            pytest.fail(f"display must conservatively estimate rather than raise: {exc}")
    assert _conversation(data) == estimate_messages_tokens_rough(history)
    assert data["context_used"] == 1030 + estimate_messages_tokens_rough(history[2:])
    assert history == original


@pytest.mark.parametrize("echo", [False, True])
def test_turn_base_anchor_keeps_its_existing_priority_and_policy(echo):
    history = _history()
    agent = _agent(echo)
    agent._turn_base_usage_anchor = capture_usage_anchor(1000, 30, history[:1])
    agent._usage_anchor = capture_usage_anchor(9000, 40, history[:3])
    delta = deepcopy(history[2:])
    del delta[1]["reasoning"]
    data = _breakdown(agent, history)
    assert data["context_used"] == 1030 + estimate_messages_tokens_rough(delta)


@pytest.mark.parametrize("measured", [0, 777])
def test_invalid_anchors_preserve_measured_then_local_fallback(measured):
    history = _history()
    agent = _agent()
    agent._usage_anchor = capture_usage_anchor(1000, 30, history[:1])
    agent._turn_base_usage_anchor = deepcopy(agent._usage_anchor)
    history[0]["content"] = "rewritten prefix"
    agent.context_compressor.last_prompt_tokens = measured
    data = _breakdown(agent, history)
    assert data["context_used"] == (measured or data["estimated_total"])
    assert data["context_source"] == ("provider_usage" if measured else "local_estimate")


def test_empty_history_retains_zero_conversation():
    data = _breakdown(_agent(), [])
    assert _conversation(data) == data["context_used"] == 0
    assert data["context_source"] == "local_estimate"


def test_priced_reply_only_keeps_exact_usage():
    history = _history()[:2]
    agent = _agent()
    agent._usage_anchor = capture_usage_anchor(1000, 30, history[:1])
    data = _breakdown(agent, history)
    assert data["context_used"] == 1030
    assert data["context_source"] == "provider_usage"
    assert data["context_estimated"] is False
