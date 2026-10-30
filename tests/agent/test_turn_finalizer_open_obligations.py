"""#16004: a turn that stops on the iteration limit with open todo items must not read as finished."""

import pytest

from tests.agent.test_turn_finalizer_iteration_limit_exit import _LimitAgent, _finalize
from tools.todo_tool import TodoStore


def _limit_agent(**kw):
    agent = _LimitAgent(**kw)
    agent._emit_diagnostic_status = lambda *_a, **_k: None   # the summary branch reports status; not under test here
    return agent


def _agent_with_todos(statuses, **kw):
    agent = _limit_agent(**kw)
    store = TodoStore()
    store.write([{"id": str(i + 1), "content": f"stage {i + 1}", "status": s} for i, s in enumerate(statuses)])
    agent._todo_store = store
    return agent


@pytest.fixture(autouse=True)
def _no_hooks(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])


def test_exhausted_turn_with_open_todos_appends_a_deterministic_incomplete_notice():
    # the field case: 11 obligations, 2 done, 9 still open, model summary reads as finished
    agent = _agent_with_todos(["completed"] * 2 + ["in_progress"] + ["pending"] * 8)
    result = _finalize(agent, final_response=None, exit_reason="budget_exhausted")

    assert agent._handle_max_iterations_called
    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    assert result["open_obligations"] == 9 and result["resume_required"] is True
    text = result["final_response"]
    assert text.startswith("summary from extra call")          # the model's summary is kept, not replaced
    assert "Stopped at the iteration limit (60/60) with 9 task(s) still open. This work is not complete:" in text
    assert "- [>] 3. stage 3 (in_progress)" in text and "- [ ] 4. stage 4 (pending)" in text
    assert "stage 1 (" not in text                                # finished items are not listed
    assert "... and" not in text                                   # 9 <= 10 items listed in full


def test_long_open_list_is_capped():
    agent = _agent_with_todos(["pending"] * 14)
    text = _finalize(agent, final_response=None, exit_reason="budget_exhausted")["final_response"]
    assert "with 14 task(s) still open" in text and "- ... and 4 more" in text


def test_all_todos_closed_adds_nothing():
    agent = _agent_with_todos(["completed", "cancelled"])
    result = _finalize(agent, final_response=None, exit_reason="budget_exhausted")
    assert result["final_response"] == "summary from extra call"
    assert result["open_obligations"] == 0 and result["resume_required"] is False


def test_normal_turn_with_open_todos_is_untouched():
    # open todos alone are normal mid-work state; only the iteration-limit exit is gated
    agent = _agent_with_todos(["pending"], budget_remaining=10)
    result = _finalize(agent, final_response="here is my answer", exit_reason="text_response(stop)", api_call_count=5)
    assert result["final_response"] == "here is my answer"
    assert result["open_obligations"] == 0 and result["resume_required"] is False


def test_no_todo_store_is_a_no_op():
    agent = _limit_agent()
    result = _finalize(agent, final_response=None, exit_reason="budget_exhausted")
    assert result["final_response"] == "summary from extra call" and result["open_obligations"] == 0
