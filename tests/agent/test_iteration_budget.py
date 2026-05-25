"""Tests for agent/iteration_budget.py — thread-safe iteration counter."""

import pytest
from agent.iteration_budget import IterationBudget


class TestIterationBudgetInit:
    def test_default_initialization(self):
        budget = IterationBudget(max_total=90)
        assert budget.max_total == 90
        assert budget.used == 0
        assert budget.remaining == 90

    def test_custom_max_total(self):
        budget = IterationBudget(max_total=50)
        assert budget.max_total == 50
        assert budget.remaining == 50

    def test_zero_max_total(self):
        budget = IterationBudget(max_total=0)
        assert budget.max_total == 0
        assert budget.remaining == 0
        assert budget.used == 0

    def test_negative_max_total(self):
        """Negative max_total: remaining floors at 0."""
        budget = IterationBudget(max_total=-5)
        assert budget.remaining == 0


class TestConsume:
    def test_consume_decrements_remaining(self):
        budget = IterationBudget(max_total=10)
        assert budget.consume() is True
        assert budget.used == 1
        assert budget.remaining == 9

    def test_multiple_consumes(self):
        budget = IterationBudget(max_total=5)
        results = [budget.consume() for _ in range(5)]
        assert results == [True, True, True, True, True]
        assert budget.used == 5
        assert budget.remaining == 0

    def test_consume_at_limit_returns_false(self):
        budget = IterationBudget(max_total=3)
        for _ in range(3):
            assert budget.consume() is True
        # Budget exhausted
        assert budget.consume() is False
        assert budget.used == 3
        assert budget.remaining == 0

    def test_consume_beyond_limit_stays_at_limit(self):
        budget = IterationBudget(max_total=2)
        budget.consume()  # 1
        budget.consume()  # 2 — exhausted
        for _ in range(10):
            assert budget.consume() is False
        assert budget.used == 2

    def test_consume_zero_budget(self):
        budget = IterationBudget(max_total=0)
        assert budget.consume() is False
        assert budget.used == 0
        assert budget.remaining == 0


class TestRefund:
    def test_refund_increases_remaining(self):
        budget = IterationBudget(max_total=10)
        budget.consume()
        budget.consume()
        assert budget.used == 2
        budget.refund()
        assert budget.used == 1
        assert budget.remaining == 9

    def test_refund_at_zero_does_nothing(self):
        budget = IterationBudget(max_total=5)
        budget.refund()
        assert budget.used == 0
        assert budget.remaining == 5

    def test_refund_after_exhaustion_allows_consume(self):
        budget = IterationBudget(max_total=3)
        for _ in range(3):
            budget.consume()
        assert budget.consume() is False  # exhausted
        budget.refund()
        assert budget.remaining == 1
        assert budget.consume() is True
        assert budget.remaining == 0

    def test_multiple_refunds_cannot_go_below_zero(self):
        budget = IterationBudget(max_total=5)
        budget.consume()
        budget.refund()
        budget.refund()
        budget.refund()
        assert budget.used == 0


class TestProperties:
    def test_used_tracks_consumes(self):
        budget = IterationBudget(max_total=10)
        for i in range(1, 6):
            budget.consume()
            assert budget.used == i

    def test_remaining_exact_after_consume_and_refund(self):
        budget = IterationBudget(max_total=10)
        budget.consume()
        budget.consume()
        budget.refund()
        assert budget.remaining == 9
        budget.consume()
        assert budget.remaining == 8

    def test_remaining_never_negative(self):
        budget = IterationBudget(max_total=1)
        budget.consume()  # used=1, remaining=0
        # Even if consume returns false, remaining stays non-negative
        assert budget.remaining >= 0


class TestEdgeCases:
    def test_large_max_total(self):
        budget = IterationBudget(max_total=1_000_000)
        assert budget.max_total == 1_000_000
        assert budget.remaining == 1_000_000
        assert budget.consume() is True
        assert budget.remaining == 999_999

    def test_consume_refund_cycle(self):
        budget = IterationBudget(max_total=100)
        # Simulate a session with execute_code refunds
        for _ in range(50):
            budget.consume()
        for _ in range(10):
            budget.refund()
        assert budget.used == 40
        assert budget.remaining == 60

    def test_full_exhaustion_and_partial_refund(self):
        budget = IterationBudget(max_total=100)
        for _ in range(100):
            assert budget.consume() is True
        assert budget.consume() is False
        for _ in range(30):
            budget.refund()
        assert budget.used == 70
        assert budget.remaining == 30

    def test_refund_preserves_consume_capacity(self):
        """After refund, exactly one consume should succeed per refund."""
        budget = IterationBudget(max_total=5)
        for _ in range(5):
            budget.consume()
        refunds = 3
        for _ in range(refunds):
            budget.refund()
        successes = sum(1 for _ in range(10) if budget.consume())
        assert successes == refunds
        assert budget.used == 5
