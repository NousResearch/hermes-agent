"""Exhausted-budget exits must not phantom-count a provider call.

``begin_iteration`` increments ``api_call_count`` before the budget check;
a failed ``consume()`` (which does not itself increment) ends the turn with
no provider call, so the increment must be taken back — otherwise api_calls
reports max+1 and budget_exhausted accounting flips. Sibling no-call exits
refund the same way.
"""

from unittest.mock import MagicMock, patch

from agent.iteration_budget import IterationBudget
from agent.turn_iteration_prep import begin_iteration


def _agent(budget_max, grace=False):
    agent = MagicMock()
    agent._drain_pending_redirect.return_value = None
    agent._interrupt_requested = False
    agent._budget_grace_call = grace
    agent.iteration_budget = IterationBudget(budget_max)
    agent.quiet_mode = True
    agent._api_call_count = 5
    return agent


def _begin(agent, count=5):
    with patch(
        "agent.conversation_loop._review_input_budget_exhausted", return_value=False
    ):
        return begin_iteration(
            agent, messages=[], conversation_history=[],
            original_user_message="hi", api_call_count=count,
            interrupted=False, _turn_exit_reason=None,
        )


class TestBudgetExhaustedNoPhantom:
    def test_exhausted_break_takes_back_increment(self):
        # Entered at 5: the increment-then-failed-consume must net to no
        # change (pre-fix this reported 6 — a call that never happened).
        verdict = _begin(_agent(0))
        assert verdict.action == "break"
        assert verdict._turn_exit_reason == "budget_exhausted"
        assert verdict.api_call_count == 5

    def test_agent_mirror_count_matches(self):
        agent = _agent(0)
        _begin(agent)
        assert agent._api_call_count == 5

    def test_failed_consume_leaves_budget_untouched(self):
        agent = _agent(0)
        _begin(agent)
        assert agent.iteration_budget.used == 0

    def test_grace_call_keeps_its_count(self):
        # The grace flag means a real call follows: no take-back.
        agent = _agent(0, grace=True)
        verdict = _begin(agent)
        assert verdict.action == "fallthrough"
        assert verdict.api_call_count == 6
        assert agent._budget_grace_call is False
