"""Regression tests for reasoning configuration on MoA aggregators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


MAX_REASONING = {"enabled": True, "effort": "max"}


def _response(content: str = "aggregated") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="gpt-5.6-sol")


@pytest.fixture
def captured_calls(monkeypatch):
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response()

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    monkeypatch.setattr(
        "agent.moa_loop._run_references_parallel",
        lambda *args, **kwargs: [("advisor", "reference advice", None)],
    )
    monkeypatch.setattr(
        "agent.moa_loop._slot_runtime",
        lambda slot: {
            "provider": slot["provider"],
            "model": slot["model"],
            "base_url": "https://example.invalid",
            "api_key": "test-key",
            "api_mode": "codex_responses",
        },
    )
    return calls


def _aggregator_call(calls):
    return next(call for call in calls if call["task"] == "moa_aggregator")


def test_acting_aggregator_receives_parent_reasoning_config(
    captured_calls, monkeypatch
):
    from run_agent import AIAgent

    monkeypatch.setattr(
        "hermes_cli.moa_config.resolve_moa_preset",
        lambda config, name: {
            "enabled": True,
            "reference_models": [
                {"provider": "zai", "model": "glm-5"},
            ],
            "aggregator": {
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
            },
        },
    )

    agent = AIAgent(
        api_key="moa-virtual-provider",
        base_url="moa://local",
        model="software-dev",
        provider="moa",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["file"],
        max_iterations=1,
        reasoning_config=MAX_REASONING,
    )
    agent.run_conversation("review this change")

    assert _aggregator_call(captured_calls)["reasoning_config"] == MAX_REASONING


def test_one_shot_aggregator_receives_parent_reasoning_config(captured_calls):
    from agent.moa_loop import aggregate_moa_context

    aggregate_moa_context(
        user_prompt="review this change",
        api_messages=[{"role": "user", "content": "review this change"}],
        reference_models=[{"provider": "zai", "model": "glm-5"}],
        aggregator={"provider": "openai-codex", "model": "gpt-5.6-sol"},
        reasoning_config=MAX_REASONING,
    )

    assert _aggregator_call(captured_calls)["reasoning_config"] == MAX_REASONING
