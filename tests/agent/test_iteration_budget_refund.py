"""Tests for iteration budget refund on API failure (#77305).

When an API call fails with a retryable error (429, timeout, auth) and
the fallback chain activates, the iteration budget consumed for the
failed call must be refunded so the fallback provider gets a fair
chance. Without this, a burst of 429s deep in a long subagent run
burns the entire budget on provider failures before the configured
fallback_providers chain gets a turn.
"""

from __future__ import annotations

from agent.iteration_budget import IterationBudget


def test_refund_decrements_used_count():
    """refund() must decrement the used counter."""
    budget = IterationBudget(max_total=5)
    assert budget.consume() is True
    assert budget.used == 1
    budget.refund()
    assert budget.used == 0
    assert budget.remaining == 5


def test_refund_does_not_go_negative():
    """refund() must not decrement below zero."""
    budget = IterationBudget(max_total=5)
    budget.refund()  # no-op: nothing consumed yet
    assert budget.used == 0
    assert budget.remaining == 5


def test_refund_restores_budget_after_failure():
    """Simulate the #77305 scenario: consume for a failed API call,
    then refund so the fallback gets the slot back."""
    budget = IterationBudget(max_total=3)
    # Consume for the initial call
    assert budget.consume() is True
    assert budget.remaining == 2
    # API call fails with 429 — refund
    budget.refund()
    assert budget.remaining == 3
    # Fallback call consumes the refunded slot
    assert budget.consume() is True
    assert budget.remaining == 2


def test_burst_of_failures_does_not_exhaust_budget():
    """The core #77305 scenario: a burst of 429s should not exhaust the
    budget if each failed call is refunded."""
    budget = IterationBudget(max_total=10)
    # Simulate 8 failed API calls, each refunded
    for _ in range(8):
        budget.consume()
        budget.refund()
    # Budget should be fully intact
    assert budget.remaining == 10
    # The 9th call (first successful one) should consume normally
    assert budget.consume() is True
    assert budget.remaining == 9


def test_refund_is_thread_safe():
    """refund() uses a lock, so concurrent refunds are safe."""
    import threading

    budget = IterationBudget(max_total=100)
    # Consume all 100
    for _ in range(100):
        budget.consume()

    # Concurrently refund from multiple threads
    def _refund_n(n: int):
        for _ in range(n):
            budget.refund()

    threads = [threading.Thread(target=_refund_n, args=(25,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 100 should be refunded
    assert budget.used == 0
    assert budget.remaining == 100