"""Shared exit policy distinguishes verified final billing failures from ambiguous reports."""

import pytest

from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE, KANBAN_TERMINAL_PROVIDER_EXIT_CODE
from hermes_cli.turn_exit import turn_exit_code


@pytest.mark.parametrize("failure", [{"failed": True}, {"partial": True}, {"completed": False}])
@pytest.mark.parametrize("kanban_worker", [True, False])
def test_verified_billing_is_terminal_only_for_failed_worker_turns(failure, kanban_worker):
    result = {**failure, "failure_reason": "billing", "billing_unverified": False}
    expected = KANBAN_TERMINAL_PROVIDER_EXIT_CODE if kanban_worker else 1
    assert turn_exit_code(result, kanban_worker=kanban_worker) == expected


@pytest.mark.parametrize("metadata", [
    {}, {"billing_unverified": True}, {"billing_unverified": None},
    {"billing_unverified": 0}, {"billing_unverified": "false"},
])
def test_ambiguous_billing_stays_retryable(metadata):
    result = {"failed": True, "failure_reason": "billing", **metadata}
    assert turn_exit_code(result, kanban_worker=True) == KANBAN_RATE_LIMIT_EXIT_CODE


@pytest.mark.parametrize(("outcome", "expected"), [
    ({"completed": True}, 0),
    ({"failed": True, "interrupted": True}, 130),
])
def test_billing_metadata_preserves_completion_and_interruption(outcome, expected):
    result = {**outcome, "failure_reason": "billing", "billing_unverified": False}
    assert turn_exit_code(result, kanban_worker=True) == expected
