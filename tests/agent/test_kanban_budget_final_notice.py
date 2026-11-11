"""Kanban workers get a handoff-first checkpoint and a final stop notice near the cap."""
from types import SimpleNamespace

import pytest


class _Budget:
    def __init__(self, max_total):
        self.max_total = max_total
        self.used = 0


def _agent(max_total, ratio=0.8):
    return SimpleNamespace(
        _interrupt_requested=False, budget_warning_ratio=ratio,
        valid_tool_names={"kanban_complete", "kanban_comment"},
        iteration_budget=_Budget(max_total),
        _iteration_budget_warning_injected=False,
        _iteration_budget_final_warning_injected=False,
    )


def _step(agent, used):
    from agent.turn_iteration_prep import _maybe_inject_iteration_budget_warning
    agent.iteration_budget.used = used
    messages = [{"role": "tool", "tool_call_id": "t", "content": "ok"}]
    _maybe_inject_iteration_budget_warning(agent, messages)
    return messages[-1]["content"]


@pytest.fixture
def worker(monkeypatch):
    import agent.delegation_context as dc
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_final")
    monkeypatch.setattr(dc, "is_dispatcher_owned_worker_context", lambda: True)


def test_kanban_worker_gets_handoff_then_final_notice(worker):
    agent = _agent(40)
    assert _step(agent, 10) == "ok"
    first = _step(agent, 32)
    assert "NEXT tool call must be kanban_comment" in first
    assert "kanban_complete" in first
    assert _step(agent, 33) == "ok"
    final = _step(agent, 35)
    assert "iteration budget final" in final and "5 iterations left" in final
    assert "kanban_block" in final
    assert _step(agent, 37) == "ok"  # one-shot


def test_small_budget_gets_no_final_notice(worker):
    agent = _agent(4)
    assert "kanban_comment" in _step(agent, 3)
    assert _step(agent, 3) == "ok"


def test_ordinary_agent_never_gets_final_notice(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent = _agent(40)
    first = _step(agent, 32)
    assert "budget checkpoint" in first and "kanban_comment" not in first
    assert _step(agent, 36) == "ok"
