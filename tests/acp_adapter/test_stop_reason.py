"""ACP ``stopReason`` precedence (``acp_adapter/stop_reason.py``).

The loopback E2E that drives the real turn loop into the output-cap verdict lives in
``test_failed_turn_closure.py`` next to the provider fixture it uses.
"""

from __future__ import annotations

import pytest

from acp_adapter.stop_reason import acp_stop_reason


@pytest.mark.parametrize(
    "result,cancelled,expected",
    [
        ({"failure_reason": "truncated", "completed": False}, True, "cancelled"),  # the user's stop wins
        ({"failure_reason": "timeout", "completed": False, "failed": True}, False, "end_turn"),  # transport, not a cap
        ({"turn_exit_reason": "max_iterations_reached(3/3)", "completed": False}, False, "max_turn_requests"),
        ({"turn_exit_reason": "max_iterations_reached(3/3)", "completed": True}, False, "end_turn"),
        ({"completed": True}, False, "end_turn"),
        (None, False, "end_turn"),
    ],
)
def test_stop_reason_precedence(result, cancelled, expected):
    assert acp_stop_reason(result, cancelled=cancelled) == expected
