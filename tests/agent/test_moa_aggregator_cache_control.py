"""Regression test: the MoA aggregator's one-shot synthesis call
(``aggregate_moa_context``, used by the ``/moa <prompt>`` command) must get
the same Anthropic-style prompt-caching decoration as the acting-aggregator
turn (``MoAChatCompletions.create``) and the advisor fan-out
(``_run_reference``).

22c5048d9 ("fix(moa): restore prompt caching for the aggregator and
advisors") fixed the other two MoA call paths but never touched
``aggregate_moa_context`` — a third, independent call path with its own
``call_llm(task="moa_aggregator", ...)`` invocation. Without this fix, every
``/moa <prompt>`` one-shot call re-bills its full input (system-less prompt
containing all joined reference outputs) with zero cache_control breakpoints,
even when the resolved aggregator slot is a cache-honoring route.
"""

from __future__ import annotations

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


def test_aggregator_synthesis_gets_cache_control_on_native_anthropic_route(
    captured_calls, monkeypatch
):
    """A cache-honoring aggregator slot (native Anthropic) must get
    cache_control breakpoints on its synthesis call."""
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

    agg_kwargs = _aggregator_kwargs(captured_calls)
    synth_message = agg_kwargs["messages"][0]
    assert synth_message["role"] == "user"
    content = synth_message["content"]
    # Native Anthropic layout places cache_control on inner content blocks,
    # so a cached message's content is a list of blocks rather than a bare
    # string once decorated.
    assert isinstance(content, list), "expected native cache_control block layout"
    assert any(
        isinstance(block, dict) and "cache_control" in block for block in content
    ), "aggregator synthesis message must carry a cache_control breakpoint"


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
