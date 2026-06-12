"""Regression tests for unbounded iteration budgets."""

from agent.iteration_budget import IterationBudget


def test_non_positive_iteration_budget_is_unbounded_and_still_counts_usage():
    budget = IterationBudget(0)

    assert budget.is_unbounded
    for _ in range(5):
        assert budget.consume() is True

    assert budget.used == 5
    assert budget.remaining == float("inf")


def test_positive_iteration_budget_still_exhausts_at_limit():
    budget = IterationBudget(2)

    assert not budget.is_unbounded
    assert budget.consume() is True
    assert budget.consume() is True
    assert budget.consume() is False
    assert budget.remaining == 0
