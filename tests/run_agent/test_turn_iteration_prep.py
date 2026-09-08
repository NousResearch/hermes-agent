"""Bound the refunding restart paths so a runaway turn cannot hold the turn lease.

``apply_retry_restarts`` has three restart paths that refund the iteration
budget and re-issue the iteration. The redirect and rebuilt-for-fallback paths
were unbounded: a runaway interrupt/redirect that keeps re-arming the restart
flag refunded the budget forever, so the turn loop never exited, the durable
turn lease (``turn_facade_lease.py``) never released in the ``finally`` block,
and concurrent processes blocked on the lease for up to ``LEASE_WAIT_SECONDS``.

The compression path was already capped (via the shared per-turn
``compression_attempts`` backstop); this change gives redirect and rebuilt the
same treatment with a per-turn ``restart_count`` accumulator capped at
``max_retries`` (the retry-loop bound).

These tests assert the *behavior contract* — a repeated restart eventually
``break`` s instead of refunding forever — by driving ``apply_retry_restarts``
directly. No source inspection.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.turn_iteration_prep import apply_retry_restarts
from agent.turn_retry_state import TurnRetryState


class _RefundBudget:
    """Minimal iteration budget that counts refunds (the runaway signal)."""

    def __init__(self) -> None:
        self.refund_count = 0

    def refund(self) -> None:
        self.refund_count += 1


def _apply(agent, _retry, restart_count: int):
    """Invoke ``apply_retry_restarts`` with a minimal loop-locals payload."""
    return apply_retry_restarts(
        agent,
        _retry=_retry,
        response=None,
        interrupted=False,
        messages=[],
        conversation_history=[],
        user_message="hi",
        api_kwargs={},
        current_turn_user_idx=0,
        final_response=None,
        retry_count=0,
        max_retries=agent._api_max_retries,
        api_call_count=1,
        restart_count=restart_count,
        length_continue_retries=0,
        _preflight_compression_blocked=False,
        _turn_exit_reason="unknown",
    )


def _make_agent(max_retries: int) -> SimpleNamespace:
    return SimpleNamespace(iteration_budget=_RefundBudget(), _api_max_retries=max_retries)


def _drive_repeated_restart(agent, arm):
    """Re-arm one restart flag every iteration, threading ``restart_count`` back in."""
    restart_count = 0
    actions = []
    reasons = []
    for _ in range(agent._api_max_retries + 5):
        _retry = TurnRetryState()
        arm(_retry)
        verdict = _apply(agent, _retry, restart_count=restart_count)
        restart_count = verdict.restart_count
        actions.append(verdict.action)
        reasons.append(verdict._turn_exit_reason)
        if verdict.action == "break":
            break
    return actions, reasons


class TestRedirectRestartBound:
    def test_single_redirect_restart_still_continues(self):
        """A lone user correction must still re-issue the iteration (one refund)."""
        agent = _make_agent(max_retries=3)
        _retry = TurnRetryState()
        _retry.restart_with_redirected_messages = True
        verdict = _apply(agent, _retry, restart_count=0)

        assert verdict.action == "continue"
        assert agent.iteration_budget.refund_count == 1

    def test_repeated_redirect_restarts_break_after_bounded_count(self):
        """A runaway redirect must break instead of refunding forever."""
        agent = _make_agent(max_retries=3)
        actions, reasons = _drive_repeated_restart(
            agent, lambda r: setattr(r, "restart_with_redirected_messages", True)
        )

        assert "break" in actions, "redirect restarts must eventually break, not refund forever"
        # Exactly ``max_retries`` refunds are allowed before the cap trips.
        assert actions.count("continue") == agent._api_max_retries
        assert agent.iteration_budget.refund_count == agent._api_max_retries
        assert reasons[-1] == "redirect_restart_limit_exceeded"


class TestRebuiltRestartBound:
    def test_single_rebuilt_restart_still_continues(self):
        """A lone fallback activation must still re-issue the iteration (one refund)."""
        agent = _make_agent(max_retries=3)
        _retry = TurnRetryState()
        _retry.restart_with_rebuilt_messages = True
        verdict = _apply(agent, _retry, restart_count=0)

        assert verdict.action == "continue"
        assert agent.iteration_budget.refund_count == 1
        # The single consumer still clears the preflight block for the fallback.
        assert verdict._preflight_compression_blocked is False

    def test_repeated_rebuilt_restarts_break_after_bounded_count(self):
        """A stall that keeps re-escalating to fallback must break, not refund forever."""
        agent = _make_agent(max_retries=3)
        actions, reasons = _drive_repeated_restart(
            agent, lambda r: setattr(r, "restart_with_rebuilt_messages", True)
        )

        assert "break" in actions, "rebuilt restarts must eventually break, not refund forever"
        assert actions.count("continue") == agent._api_max_retries
        assert agent.iteration_budget.refund_count == agent._api_max_retries
        assert reasons[-1] == "rebuilt_restart_limit_exceeded"
