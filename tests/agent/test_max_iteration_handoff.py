"""Shared classification of a non-failed iteration-cap turn (#102213).

Cron delivery and the TUI ``/goal`` judge gate must agree on what a resumable
``max_iterations_reached`` handoff is, so both read one predicate instead of
parsing the exit reason independently.
"""

import pytest

from agent.turn_failure_copy import is_max_iteration_handoff
from cron.scheduler import _final_response_from_result

_HANDOFF = {
    "final_response": "checkpoint: two of three steps done",
    "completed": False,
    "failed": False,
    "interrupted": False,
    "turn_exit_reason": "max_iterations_reached(3/3)",
}


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, True),
        ({"failed": True}, False),
        ({"interrupted": True}, False),
        ({"completed": True}, False),
        ({"turn_exit_reason": "budget_exhausted"}, False),
        ({"turn_exit_reason": "error_near_max_iterations(3/3)"}, False),
        ({"final_response": "   "}, False),
    ],
)
def test_only_a_non_failed_capped_turn_with_a_summary_is_a_handoff(override, expected):
    assert is_max_iteration_handoff({**_HANDOFF, **override}) is expected


def test_cron_delivers_exactly_the_handoffs_the_predicate_accepts():
    assert _final_response_from_result(dict(_HANDOFF), "j", "job", object) == _HANDOFF["final_response"]
    with pytest.raises(RuntimeError):
        _final_response_from_result({**_HANDOFF, "failed": True}, "j", "job", object)
