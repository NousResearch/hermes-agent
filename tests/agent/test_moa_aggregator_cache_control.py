"""MoA aggregator cache decoration: the acting aggregator (``MoAChatCompletions``)
and advisor fan-out are re-read across iterations and stay decorated; the one-shot
``/moa <prompt>`` synthesis (``aggregate_moa_context``) embeds this call's reference
outputs, is never read back, and stays undecorated (no cache write).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest


def _response(content="synthesized guidance"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


@pytest.fixture
def captured_calls(monkeypatch):
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response()

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    monkeypatch.setattr(
        "agent.moa_loop._run_references_parallel",
        lambda *a, **k: [("advisor-a", "advice from a", None)],
    )
    return calls


def _aggregator_kwargs(calls):
    return next(c for c in calls if c.get("task") == "moa_aggregator")


def test_aggregator_synthesis_unmarked_on_native_anthropic_route(
    captured_calls, monkeypatch
):
    """The synthesis prompt embeds this call's reference outputs, so no later call reads
    it back: even on a cache-honoring slot it carries no cache_control (a marker would
    only pay a cache write)."""
    from agent import moa_loop

    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": "anthropic",
            "model": "claude-opus-4.8",
            "base_url": "",
            "api_mode": "anthropic_messages",
        },
    )

    moa_loop.aggregate_moa_context(
        user_prompt="what should I do next?",
        api_messages=[{"role": "user", "content": "help me plan"}],
        reference_models=[{"provider": "openrouter", "model": "openai/gpt-5.5"}],
        aggregator={"provider": "anthropic", "model": "claude-opus-4.8"},
    )

    synth_message = _aggregator_kwargs(captured_calls)["messages"][0]
    assert synth_message["role"] == "user"
    assert "cache_control" not in repr(synth_message)


def test_aggregator_synthesis_untouched_on_non_caching_route(
    captured_calls, monkeypatch
):
    """A non-cache-honoring aggregator slot (plain OpenAI) must not be
    decorated — proves the guard doesn't over-fire."""
    from agent import moa_loop

    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": "openai",
            "model": "gpt-5.5",
            "base_url": "",
            "api_mode": "chat_completions",
        },
    )

    moa_loop.aggregate_moa_context(
        user_prompt="what should I do next?",
        api_messages=[{"role": "user", "content": "help me plan"}],
        reference_models=[{"provider": "openrouter", "model": "openai/gpt-5.5"}],
        aggregator={"provider": "openai", "model": "gpt-5.5"},
    )

    agg_kwargs = _aggregator_kwargs(captured_calls)
    synth_message = agg_kwargs["messages"][0]
    assert isinstance(synth_message["content"], str), "must stay undecorated (plain string content)"


def test_prepared_aggregator_plans_tools_without_decorating_prepared_state(monkeypatch):
    from agent import moa_loop

    calls = []
    monkeypatch.setattr(moa_loop, "call_llm", lambda **kwargs: calls.append(kwargs) or _response())
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
        },
    )
    completions = moa_loop.MoAChatCompletions.__new__(moa_loop.MoAChatCompletions)
    completions._pending_trace = None
    prepared = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "lookup"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "lookup", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "lookup", "content": "result"},
        ],
        "guidance": None,
        "aggregator": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "aggregator_temperature": None,
    }
    canonical_prepared = copy.deepcopy(prepared)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]

    completions._call_prepared_aggregator(prepared, {"tools": tools})

    assert "cache_control" in calls[0]["tools"][-1]
    assert "cache_control" not in tools[-1]
    assert prepared == canonical_prepared
