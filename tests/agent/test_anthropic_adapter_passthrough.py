"""The Anthropic completions adapter must honor request kwargs that
Router/MoA facades park in extra_body (call_llm's signature doesn't carry
them). extra_body is wire-equivalent for OpenAI SDK clients but this adapter
reads kwargs directly — without the hoist, a routed slot on a native
Anthropic endpoint silently loses tool_choice."""

from types import SimpleNamespace

import pytest

import agent.anthropic_adapter as anthropic_adapter
from agent.auxiliary_client import _AnthropicCompletionsAdapter


@pytest.fixture
def captured(monkeypatch):
    """Capture build_anthropic_kwargs input; stub the wire call + transport."""
    seen: dict = {}
    real_build = anthropic_adapter.build_anthropic_kwargs

    def spy_build(**kwargs):
        seen.update(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(anthropic_adapter, "build_anthropic_kwargs", spy_build)
    monkeypatch.setattr(
        anthropic_adapter,
        "create_anthropic_message",
        lambda client, kw: SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=None,
        ),
    )
    return seen


def _create(adapter_kwargs):
    adapter = _AnthropicCompletionsAdapter(real_client=object(), model="claude-x")
    return adapter.create(
        model="claude-x",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        max_tokens=64,
        **adapter_kwargs,
    )


def test_tool_choice_hoisted_from_extra_body(captured):
    _create({"extra_body": {"tool_choice": "required"}})
    assert captured.get("tool_choice") == "required"


def test_direct_tool_choice_wins_over_extra_body(captured):
    _create({"tool_choice": "auto", "extra_body": {"tool_choice": "required"}})
    assert captured.get("tool_choice") == "auto"


def test_no_tool_choice_stays_none(captured):
    _create({"extra_body": {"unrelated": 1}})
    assert captured.get("tool_choice") is None
