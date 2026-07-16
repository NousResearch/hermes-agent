"""Tests for issue #65746 — MoA/local calls crash after 30s with
``OverflowError: cannot convert float infinity to integer``.

Local providers (Ollama, MoA virtual endpoints) set the stale timeout to
``float("inf")`` to disable the stale detector. When the 30-second
wait-status heartbeat fires and formats the deadline with ``int(_deadline)``,
this raises ``OverflowError`` which propagates through the retry layer as
a spurious API failure.

The fix: ``_safe_int_seconds()`` guards against non-finite values before
calling ``int()``, returning a large sentinel (999999) instead of crashing.
"""

from __future__ import annotations

import math

import pytest

from agent.chat_completion_helpers import _safe_int_seconds


# --------------------------------------------------------------------------- #
# _safe_int_seconds
# --------------------------------------------------------------------------- #


def test_safe_int_with_finite_value():
    """Normal finite values pass through int() unchanged."""
    assert _safe_int_seconds(30.0) == 30
    assert _safe_int_seconds(0.0) == 0
    assert _safe_int_seconds(180.5) == 180
    assert _safe_int_seconds(999.99) == 999


def test_safe_int_with_infinity():
    """float('inf') must not raise OverflowError — returns sentinel."""
    # This is the exact crash from the bug report:
    #   int(float('inf')) → OverflowError: cannot convert float infinity to integer
    result = _safe_int_seconds(float("inf"))
    assert result == 999999
    # Must not raise
    assert isinstance(result, int)


def test_safe_int_with_negative_infinity():
    """float('-inf') must also be handled without crashing."""
    result = _safe_int_seconds(float("-inf"))
    assert result == 999999


def test_safe_int_with_nan():
    """NaN must also be handled without crashing."""
    result = _safe_int_seconds(float("nan"))
    assert result == 999999


def test_safe_int_does_not_raise_on_any_input():
    """The helper must never raise — it's used in display formatting."""
    # Test a range of edge cases
    for val in [0, 0.0, -0.0, 1, 1e10, 1e-10, float("inf"), float("-inf"),
                float("nan"), -100.5, 3.14159]:
        result = _safe_int_seconds(val)
        assert isinstance(result, int)


def test_old_int_crashes_on_infinity():
    """Verify the original bug: int(float('inf')) raises OverflowError.

    This test documents the crash that _safe_int_seconds prevents.
    """
    with pytest.raises(OverflowError):
        int(float("inf"))


# --------------------------------------------------------------------------- #
# Integration: the wait-status heartbeat format string
# --------------------------------------------------------------------------- #


def test_wait_notice_format_with_infinite_deadline():
    """The format string that crashed must now work with infinite deadline.

    Before the fix:
        f"...auto-reconnect at {int(_deadline)}s)"  → OverflowError

    After the fix:
        f"...auto-reconnect at {_safe_int_seconds(_deadline)}s)"  → OK
    """
    _deadline = float("inf")  # The value that caused the crash
    _elapsed = 30.0

    # This is the exact format string from the heartbeat (line ~622)
    msg = (
        f"⏳ waiting on the provider — "
        f"{int(_elapsed)}s with no response yet (provider may be slow "
        f"or overloaded; auto-reconnect at {_safe_int_seconds(_deadline)}s)"
    )

    assert "999999s" in msg
    assert "30s" in msg
    assert "auto-reconnect" in msg


def test_timeout_error_message_with_infinite_stale_timeout():
    """The TimeoutError format strings must also handle infinite stale timeout."""
    _stale_timeout = float("inf")
    _elapsed = 45.0

    # This is the format from the non-streaming timeout path (line ~778)
    msg = (
        f"Non-streaming API call timed out after {int(_elapsed)}s "
        f"with no response (threshold: {_safe_int_seconds(_stale_timeout)}s)"
    )

    assert "999999s" in msg
    assert "45s" in msg