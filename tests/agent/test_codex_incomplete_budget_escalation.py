"""Responses-wire ``status=incomplete`` continuation must change the request when reasoning
consumed the whole output budget (#90393): a retry with the same ``max_output_tokens`` and the
same effort re-burns the budget identically and the turn can never converge. A fragment that
already carries visible text is a normal partial answer and keeps today's bare replay; with no
configured cap the escalation seeds from the ceiling the provider reported."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_truncation import continue_codex_incomplete


def _agent(max_tokens: int | None = 2000):
    agent = MagicMock()
    agent.max_tokens = max_tokens
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent._codex_incomplete_retries = 0
    agent._codex_reasoning_only_streak = 0
    agent._ephemeral_reasoning_off = False
    agent._ephemeral_max_output_tokens = None
    agent._build_assistant_message.side_effect = lambda msg, fr: {
        "role": "assistant", "content": msg.content or "", "finish_reason": fr,
        "codex_reasoning_items": msg.codex_reasoning_items,
    }
    agent._interim_assistant_visible_text.return_value = ""
    return agent


def _reasoning_only_message():
    return SimpleNamespace(
        content="", tool_calls=None, reasoning=None,
        codex_reasoning_items=[{"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}],
    )


def _run(agent, response):
    messages = [{"role": "user", "content": "write an essay"}]
    return continue_codex_incomplete(
        agent, _reasoning_only_message(), "incomplete", messages=messages,
        conversation_history=None, api_call_count=1, response=response,
    )


def test_budget_exhausted_empty_fragment_raises_cap_and_drops_reasoning():
    agent = _agent()
    exhausted = SimpleNamespace(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"},
        usage=SimpleNamespace(output_tokens=2000, output_tokens_details={"reasoning_tokens": 1997}),
    )
    assert _run(agent, exhausted) is None
    assert agent._ephemeral_reasoning_off is True
    first_boost = agent._ephemeral_max_output_tokens
    assert first_boost > 2000
    agent._ephemeral_max_output_tokens = None  # request builder consumes it
    assert _run(agent, exhausted) is None
    assert agent._ephemeral_max_output_tokens > first_boost


def test_no_configured_cap_seeds_escalation_from_observed_ceiling():
    agent = _agent(max_tokens=None)
    exhausted = SimpleNamespace(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"},
        usage=SimpleNamespace(output_tokens=6000, output_tokens_details={"reasoning_tokens": 5997}),
    )
    assert _run(agent, exhausted) is None
    assert agent._ephemeral_max_output_tokens > 6000
    # A fragment that carries visible text is a normal partial answer: no override, today's replay.
    agent = _agent()
    agent._interim_assistant_visible_text.return_value = "First paragraph of the essay"
    messages = [{"role": "user", "content": "write an essay"}]
    partial = SimpleNamespace(content="First paragraph of the essay", tool_calls=None, reasoning=None,
                              codex_reasoning_items=None)
    continue_codex_incomplete(agent, partial, "incomplete", messages=messages,
                              conversation_history=None, api_call_count=1, response=exhausted)
    assert agent._ephemeral_reasoning_off is False
    assert agent._ephemeral_max_output_tokens is None
