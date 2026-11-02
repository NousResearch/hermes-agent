"""Bounded auto-continue after iteration-budget exhaustion.

When the tool-calling loop exhausts its iteration budget mid-task, the turn must
produce a checkpoint summary and continue in a fresh budget window (bounded), rather
than ending with a "here's what I found so far" terminal answer.
"""

from types import SimpleNamespace

import pytest

from agent.conversation_loop import (
    _MAX_BUDGET_CONTINUE_WINDOWS,
    _continue_after_budget_exhaustion,
)
from agent.iteration_budget import IterationBudget


def _state(**overrides):
    base = dict(
        api_call_count=250,
        final_response=None,
        interrupted=False,
        failed=False,
        messages=[],
        _cumulative_api_calls=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _agent(**overrides):
    budget = IterationBudget(250)
    for _ in range(250):
        budget.consume()  # fully exhausted
    base = dict(
        max_iterations=250,
        iteration_budget=budget,
    )
    base.update(overrides)
    agent = SimpleNamespace(**base)

    def _fake_summary(messages, api_call_count):
        # Mirrors the real handler's side effect: it appends its request + the summary.
        messages.append({"role": "user", "content": "summarize so far"})
        messages.append({"role": "assistant", "content": "checkpoint summary"})
        return "checkpoint summary"

    agent._handle_max_iterations = _fake_summary
    return agent


def test_reset_restores_full_remaining():
    b = IterationBudget(3)
    for _ in range(3):
        assert b.consume()
    assert b.remaining == 0
    b.reset()
    assert b.remaining == 3
    assert b.consume()


def test_does_not_continue_when_budget_not_exhausted():
    agent = _agent()
    s = _state(api_call_count=10)
    agent.iteration_budget.reset()  # budget available
    assert _continue_after_budget_exhaustion(agent, s, 0) is False


def test_does_not_continue_beyond_window_limit():
    agent = _agent()
    s = _state()
    assert _continue_after_budget_exhaustion(agent, s, _MAX_BUDGET_CONTINUE_WINDOWS) is False


def test_does_not_continue_when_interrupted_or_failed_or_answered():
    for kw in ({"interrupted": True}, {"failed": True}, {"final_response": "done"}):
        agent = _agent()
        assert _continue_after_budget_exhaustion(agent, _state(**kw), 0) is False


def test_continues_and_checkpoints_when_budget_exhausted():
    agent = _agent()
    s = _state()
    assert _continue_after_budget_exhaustion(agent, s, 0) is True
    # A checkpoint summary plus a resume instruction were appended.
    assert any(m.get("role") == "assistant" for m in s.messages)
    assert any(m.get("role") == "user" for m in s.messages)
    # Budget and per-window counter were reset for the next window.
    assert s.api_call_count == 0
    assert agent.iteration_budget.remaining == 250
    # Cumulative accounting retained the completed window.
    assert s._cumulative_api_calls == 250


def test_does_not_continue_when_compression_breakdown():
    for kw in ({"_compression_timeout_exhausted": True}, {"_preflight_compression_blocked": True}):
        agent = _agent()
        assert _continue_after_budget_exhaustion(agent, _state(**kw), 0) is False


def test_terminal_summary_not_produced_when_final_response_already_set():
    agent = _agent()
    s = _state(final_response="already answered")
    assert _continue_after_budget_exhaustion(agent, s, 0) is False
    assert s.messages == []
