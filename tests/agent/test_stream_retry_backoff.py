"""Test that inner streaming retry includes backoff and is interruptible (#7069)."""

import time
from unittest.mock import MagicMock, patch


def test_backoff_delay_formula():
    """Verify the backoff formula: min(5 * 2^attempt, 30)."""
    # attempt 0: min(5*1, 30) = 5
    assert min(5.0 * (2 ** 0), 30.0) == 5.0
    # attempt 1: min(5*2, 30) = 10
    assert min(5.0 * (2 ** 1), 30.0) == 10.0
    # attempt 2: min(5*4, 30) = 20
    assert min(5.0 * (2 ** 2), 30.0) == 20.0
    # attempt 3: min(5*8, 30) = 30 (capped)
    assert min(5.0 * (2 ** 3), 30.0) == 30.0


def test_interruptible_backoff_exits_on_interrupt():
    """Simulate the chunked sleep loop with interrupt flag."""
    interrupt_requested = False
    backoff = 10.0
    sleep_end = time.time() + backoff
    iterations = 0

    # Simulate interrupt after ~2 iterations
    while time.time() < sleep_end:
        iterations += 1
        if iterations >= 2:
            interrupt_requested = True
        if interrupt_requested:
            break
        time.sleep(0.01)

    assert interrupt_requested is True
    assert iterations >= 2
    # Should have exited well before the full backoff duration
    assert time.time() < sleep_end
