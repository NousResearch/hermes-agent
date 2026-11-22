"""Responses ``status=incomplete`` raises the output cap without disabling reasoning.

The bounded escalation seeds from the observed ceiling when no cap is configured
(#90393). Visible partial answers retain the existing replay without a cap override.
"""

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
    agent._ephemeral_max_output_tokens = None
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    agent._reasoning_disable_rejected = False
    agent._reasoning_effort_rejected = False
    agent._try_activate_fallback.return_value = False
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


def test_budget_exhausted_empty_fragment_raises_cap_and_keeps_reasoning():
    from agent.chat_completion_helpers import _reasoning_config_for_wire

    agent = _agent()
    exhausted = SimpleNamespace(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"},
        usage=SimpleNamespace(output_tokens=2000, output_tokens_details={"reasoning_tokens": 1997}),
    )
    assert _run(agent, exhausted) is None
    first_boost = agent._ephemeral_max_output_tokens
    assert first_boost == 4000
    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "high"}
    agent._ephemeral_max_output_tokens = None  # request builder consumes it
    assert _run(agent, exhausted) is None
    assert agent._ephemeral_max_output_tokens == 8000
    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "high"}
    agent._ephemeral_max_output_tokens = None
    terminal = _run(agent, exhausted)
    assert terminal is not None
    assert terminal["completed"] is False
    assert terminal["partial"] is True
    assert "3 continuation attempts" in terminal["final_response"]
    assert agent._ephemeral_max_output_tokens is None
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}


def test_no_configured_cap_seeds_escalation_from_observed_ceiling():
    agent = _agent(max_tokens=None)
    exhausted = SimpleNamespace(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"},
        usage=SimpleNamespace(output_tokens=6000, output_tokens_details={"reasoning_tokens": 5997}),
    )
    assert _run(agent, exhausted) is None
    assert agent._ephemeral_max_output_tokens == 12000
    # A fragment that carries visible text is a normal partial answer: no override, today's replay.
    agent = _agent()
    agent._interim_assistant_visible_text.return_value = "First paragraph of the essay"
    messages = [{"role": "user", "content": "write an essay"}]
    partial = SimpleNamespace(content="First paragraph of the essay", tool_calls=None, reasoning=None,
                              codex_reasoning_items=None)
    continue_codex_incomplete(agent, partial, "incomplete", messages=messages,
                              conversation_history=None, api_call_count=1, response=exhausted)
    assert agent._ephemeral_max_output_tokens is None
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}


def test_completely_empty_fragment_is_not_appended():
    agent = _agent()
    messages = [{"role": "user", "content": "write an essay"}]
    empty = _reasoning_only_message()
    empty.codex_reasoning_items = []
    exhausted = SimpleNamespace(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"},
        usage=SimpleNamespace(output_tokens=2000),
    )
    assert continue_codex_incomplete(
        agent, empty, "incomplete", messages=messages,
        conversation_history=None, api_call_count=1, response=exhausted,
    ) is None
    assert messages == [{"role": "user", "content": "write an essay"}]
    agent._emit_interim_assistant_message.assert_not_called()
    assert agent._ephemeral_max_output_tokens == 4000
