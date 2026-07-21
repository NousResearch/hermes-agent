"""Tests for IterationBudget refund cap and stuck-loop detection.

The refund cap prevents infinite loops when the model keeps calling
execute_code without making progress.  After max_refunds (default 15),
refund() returns False and the budget erodes normally.
"""


def test_refund_cap_limits_total_refunds():
    """refund() must stop granting refunds after max_refunds is reached."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=10, max_refunds=3)

    # Consume 5
    for _ in range(5):
        budget.consume()
    assert budget.used == 5

    # Refund 3 times (all should succeed)
    assert budget.refund() is True
    assert budget.refund() is True
    assert budget.refund() is True
    assert budget.used == 2

    # 4th refund should be denied (cap reached)
    assert budget.refund() is False
    assert budget.used == 2  # unchanged


def test_refund_cap_default_is_15():
    """Default max_refunds should be 15."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=90)
    assert budget.max_refunds == 15


def test_refund_returns_false_when_used_is_zero():
    """refund() returns False when nothing has been consumed."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=10, max_refunds=5)
    assert budget.refund() is False
    assert budget.used == 0


def test_refund_cap_prevents_budget_from_growing_unbounded():
    """With refund cap, budget eventually exhausts even with constant refunds."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=5, max_refunds=3)
    iterations = 0

    # Simulate the conversation loop: consume then refund every iteration
    while budget.remaining > 0:
        if not budget.consume():
            break
        iterations += 1
        budget.refund()  # may or may not succeed after cap

    # Without cap: would loop forever. With cap of 3:
    # After 3 refunds, budget erodes. Total iterations = max_total + max_refunds = 8
    assert iterations == 5 + 3  # 5 base + 3 refunded


def test_refunds_remaining_property():
    """refunds_remaining tracks how many refunds are still available."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=10, max_refunds=3)
    assert budget.refunds_remaining == 3

    budget.consume()
    budget.refund()
    assert budget.refunds_remaining == 2

    budget.consume()
    budget.refund()
    assert budget.refunds_remaining == 1

    budget.consume()
    budget.refund()
    assert budget.refunds_remaining == 0

    # No more refunds available
    budget.consume()
    assert budget.refund() is False
    assert budget.refunds_remaining == 0
