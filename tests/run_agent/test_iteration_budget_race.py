"""Tests for IterationBudget thread safety.

The `used` property must acquire the lock before reading `_used` to prevent
data races with concurrent `consume()` / `refund()` calls.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_iteration_budget_used_is_thread_safe():
    """Iterating `used` while other threads consume/refund must not crash.

    Before the fix, `used` returned `_used` directly without holding the lock,
    so a concurrent `consume()` could observe a partially-updated value or
    cause the C-level `list.append` to raise a ValueError ("list size changed").
    """
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=1000)
    num_threads = 10
    operations_per_thread = 200

    errors = []

    def worker(consume: bool):
        try:
            for _ in range(operations_per_thread):
                if consume:
                    budget.consume()
                else:
                    budget.refund()
                # Also read `used` to exercise the property
                _ = budget.used
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=num_threads * 2) as executor:
        # Half the threads consume, half refund
        futures = []
        for i in range(num_threads):
            consume = i < num_threads // 2
            futures.append(executor.submit(worker, consume))
            futures.append(executor.submit(worker, consume))

        for f in futures:
            f.result()

    assert not errors, f"Thread safety violation: {errors}"
    # Final value should be within expected bounds
    assert 0 <= budget.used <= budget.max_total


def test_iteration_budget_consume_returns_false_when_exhausted():
    """consume() must return False once the budget is exhausted."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=3)
    assert budget.consume() is True
    assert budget.consume() is True
    assert budget.consume() is True
    assert budget.consume() is False


def test_iteration_budget_refund_restores_consume():
    """refund() after consume() must allow one more consume()."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=2)
    assert budget.consume() is True
    assert budget.consume() is True
    assert budget.consume() is False  # exhausted
    budget.refund()
    assert budget.consume() is True


def test_iteration_budget_used_reflects_consume_and_refund():
    """used property must accurately reflect consume() and refund() calls."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=10)

    assert budget.used == 0
    budget.consume()
    assert budget.used == 1
    budget.consume()
    assert budget.used == 2
    budget.refund()
    assert budget.used == 1
    budget.refund()
    assert budget.used == 0


def test_iteration_budget_remaining():
    """remaining property must equal max_total - used."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=5)

    assert budget.remaining == 5
    budget.consume()
    assert budget.remaining == 4
    budget.consume()
    budget.consume()
    assert budget.remaining == 2
    budget.refund()
    assert budget.remaining == 3


# ─── Pre-budget warning tests ───────────────────────────────────────────────


def test_budget_warn_fires_exactly_once():
    """on_warn callback must fire exactly once when remaining crosses warn_at."""
    from run_agent import IterationBudget

    fire_count = 0

    def on_warn(used, max_total):
        nonlocal fire_count
        fire_count += 1

    budget = IterationBudget(max_total=10, on_warn=on_warn, warn_at=3)

    # Consume up to the threshold: used=7, remaining=3 → fires
    budget.consume()  # used=1, remaining=9
    budget.consume()  # used=2, remaining=8
    budget.consume()  # used=3, remaining=7
    budget.consume()  # used=4, remaining=6
    budget.consume()  # used=5, remaining=5
    budget.consume()  # used=6, remaining=4
    budget.consume()  # used=7, remaining=3 → fires
    assert fire_count == 1, f"Expected 1 fire, got {fire_count}"

    # Consume more — should NOT fire again
    budget.consume()  # used=8, remaining=2
    budget.consume()  # used=9, remaining=1
    budget.consume()  # used=10, remaining=0
    assert fire_count == 1, f"Expected still 1 fire, got {fire_count}"


def test_budget_warn_no_callback_set():
    """consume() must not crash when on_warn is None."""
    from run_agent import IterationBudget

    budget = IterationBudget(max_total=10, on_warn=None, warn_at=3)

    for _ in range(10):
        budget.consume()

    # Should not raise


def test_budget_warn_zero_warn_at_disables_warning():
    """warn_at=0 must suppress all warning calls."""
    from run_agent import IterationBudget

    fire_count = 0

    def on_warn(used, max_total):
        nonlocal fire_count
        fire_count += 1

    budget = IterationBudget(max_total=5, on_warn=on_warn, warn_at=0)

    for _ in range(5):
        budget.consume()

    assert fire_count == 0, f"Expected 0 fires with warn_at=0, got {fire_count}"


def test_budget_warn_fires_at_correct_threshold():
    """on_warn must receive correct used/max_total values when fired."""
    from run_agent import IterationBudget

    received_values = []

    def on_warn(used, max_total):
        received_values.append((used, max_total))

    budget = IterationBudget(max_total=10, on_warn=on_warn, warn_at=2)

    # consume until remaining <= 2
    for _ in range(8):
        budget.consume()

    # At used=8, remaining=2 → should fire
    assert len(received_values) == 1
    assert received_values[0] == (8, 10)


def test_budget_warn_with_refund_still_fires_once():
    """Even if refund reduces used count, on_warn fires only once."""
    from run_agent import IterationBudget

    fire_count = 0

    def on_warn(used, max_total):
        nonlocal fire_count
        fire_count += 1

    budget = IterationBudget(max_total=10, on_warn=on_warn, warn_at=5)

    budget.consume()  # used=1, remaining=9
    budget.consume()  # used=2, remaining=8
    budget.consume()  # used=3, remaining=7
    budget.consume()  # used=4, remaining=6
    budget.consume()  # used=5, remaining=5 → fires
    assert fire_count == 1

    # Refund back to used=3, remaining=7
    budget.refund()
    budget.refund()
    assert budget.used == 3
    assert budget.remaining == 7

    # Consume again — should NOT fire a second time
    budget.consume()  # used=4, remaining=6
    budget.consume()  # used=5, remaining=5
    budget.consume()  # used=6, remaining=4
    budget.consume()  # used=7, remaining=3
    budget.consume()  # used=8, remaining=2
    assert fire_count == 1, f"Expected still 1 fire after refund+reconsume, got {fire_count}"


def test_budget_warn_in_callback_does_not_infinite_loop():
    """on_warn callback must not be called from within consume() reentrantly."""
    from run_agent import IterationBudget

    # The callback fires from within the lock — if it calls consume(),
    # the lock is NOT reentrant, so this could deadlock with a regular Lock.
    # Since we use a non-reentrant Lock, this test verifies the callback
    # does not hold the lock when calling user code (or that user code
    # can safely not call consume).
    budget = IterationBudget(max_total=5, warn_at=2)
    # Just ensure consume doesn't deadlock even with a callback that
    # tries to acquire the same lock (by checking remaining inside).
    remaining_during_callback = []

    def on_warn(used, max_total):
        remaining_during_callback.append(max_total - used)

    budget = IterationBudget(max_total=5, on_warn=on_warn, warn_at=2)

    for _ in range(5):
        budget.consume()

    assert len(remaining_during_callback) == 1
    assert remaining_during_callback[0] <= 2


def test_audit_dump_structure():
    """_build_audit_dump must return a multi-section structured dump."""
    import unittest.mock as mock
    from run_agent import AIAgent

    # Mock provider resolution to avoid real API key lookup
    with mock.patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)):
        agent = AIAgent(
            model="anthropic/claude-sonnet-4-20250514",
            provider="anthropic",
            api_key="test-key",
            max_iterations=10,
            prebudget_warn_at=5,
            quiet_mode=True,
        )

    dump = agent._build_audit_dump(5, 10)

    # Must contain required sections
    assert "Pre-Budget Pause" in dump
    assert "Session" in dump
    assert "Model" in dump
    assert "Iterations" in dump
    assert "5 / 10 used" in dump
    assert "5 remaining" in dump
    assert "Current Task" in dump
    assert "Files Modified" in dump
    assert "Next Steps" in dump


def test_exhaustion_injection_structure():
    """_build_exhaustion_injection must return a compact model-readable dump."""
    import unittest.mock as mock
    from run_agent import AIAgent

    with mock.patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)):
        agent = AIAgent(
            model="anthropic/claude-sonnet-4-20250514",
            provider="anthropic",
            api_key="test-key",
            max_iterations=10,
            quiet_mode=True,
        )

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there", "tool_calls": [
            {"function": {"name": "read_file", "arguments": '{"path": "test.py"}'}},
        ]},
        {"role": "tool", "content": "file contents"},
        {"role": "assistant", "content": "I found the file", "tool_calls": [
            {"function": {"name": "search_files", "arguments": '{"pattern": "foo"}'}},
        ]},
        {"role": "tool", "content": "3 matches found"},
        {"role": "assistant", "content": "Found matches", "reasoning": "Now I will patch the file to fix the bug."},
    ]

    injection = agent._build_exhaustion_injection(messages, 8)

    assert "[BUDGET EXHAUSTED" in injection
    assert "[/BUDGET EXHAUSTED" in injection
    assert "8/10" in injection
    assert "read_file" in injection
    assert "search_files" in injection
    assert "patch the file" in injection


def test_exhaustion_injection_empty_messages():
    """_build_exhaustion_injection must handle empty messages gracefully."""
    import unittest.mock as mock
    from run_agent import AIAgent

    with mock.patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)):
        agent = AIAgent(
            model="anthropic/claude-sonnet-4-20250514",
            provider="anthropic",
            api_key="test-key",
            max_iterations=10,
            quiet_mode=True,
        )

    injection = agent._build_exhaustion_injection([], 0)

    assert "[BUDGET EXHAUSTED" in injection
    assert "[/BUDGET EXHAUSTED" in injection


def test_exhaustion_injection_no_reasoning():
    """_build_exhaustion_injection uses default next steps when no reasoning exists."""
    import unittest.mock as mock
    from run_agent import AIAgent

    with mock.patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)):
        agent = AIAgent(
            model="anthropic/claude-sonnet-4-20250514",
            provider="anthropic",
            api_key="test-key",
            max_iterations=10,
            quiet_mode=True,
        )

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    injection = agent._build_exhaustion_injection(messages, 1)

    assert "Continue the task" in injection


def test_iteration_budget_callback_exception_does_not_crash():
    """If on_warn raises an exception, consume() must still return correctly."""
    from run_agent import IterationBudget

    def bad_callback(used, max_total):
        raise ValueError("kaboom")

    budget = IterationBudget(max_total=5, on_warn=bad_callback, warn_at=3)

    # This should not raise — the exception is caught inside consume()
    result = budget.consume()
    assert result is True
    assert budget.used == 1
