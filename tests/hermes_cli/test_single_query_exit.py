"""Exit-code contract for non-interactive single-query runs.

Regression cover for the drift that made a Copilot session limit look
like an agent protocol violation on the kanban board (2026-09-02, card
``t_d16778b1``): the quiet path owned the 0/1/75 contract while the
human-facing ``chat -q`` path — the one the dispatcher actually spawns —
never inspected the turn result and always exited 0.
"""

import pytest

from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
from hermes_cli.single_query_exit import single_query_exit_code


def test_successful_turn_exits_zero():
    result = {"final_response": "hi", "completed": True}
    assert single_query_exit_code(result, env={}) == 0


def test_successful_turn_in_kanban_worker_exits_zero():
    result = {"final_response": "hi", "completed": True}
    assert (
        single_query_exit_code(result, env={"HERMES_KANBAN_TASK": "t_1"}) == 0
    )


def test_failed_turn_exits_one():
    result = {"failed": True, "error": "boom", "failure_reason": "api_error"}
    assert single_query_exit_code(result, env={}) == 1


def test_interrupted_but_not_failed_turn_exits_zero():
    result = {"final_response": "", "interrupted": True}
    assert single_query_exit_code(result, env={}) == 0


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_quota_wall_in_kanban_worker_exits_tempfail(reason):
    """The dispatcher must see EX_TEMPFAIL, not a task failure.

    Its reap classifier maps 75 to ``rate_limited`` and releases the card
    back to ``ready`` without counting a failure, so a multi-hour quota
    window cannot trip the circuit breaker.
    """
    result = {"failed": True, "error": "quota", "failure_reason": reason}
    code = single_query_exit_code(result, env={"HERMES_KANBAN_TASK": "t_1"})
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE
    assert code == 75


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_quota_wall_outside_kanban_exits_one(reason):
    """Plain automation wrappers keep the 0/1 contract they expect."""
    result = {"failed": True, "error": "quota", "failure_reason": reason}
    assert single_query_exit_code(result, env={}) == 1


def test_kanban_worker_task_failure_still_exits_one():
    """A genuine task error must count against the card, not bounce it."""
    result = {"failed": True, "error": "boom", "failure_reason": "api_error"}
    assert (
        single_query_exit_code(result, env={"HERMES_KANBAN_TASK": "t_1"}) == 1
    )


def test_no_turn_result_exits_one():
    """``None`` means the turn never reached the agent — credentials or
    agent init failed, so no answer was produced."""
    assert single_query_exit_code(None, env={}) == 1


def test_non_dict_result_exits_zero():
    """The quiet path prints ``str(result)`` for an odd shape; something
    was produced, so it is not reported as a failure."""
    assert single_query_exit_code("plain text answer", env={}) == 0


def test_env_defaults_to_os_environ(monkeypatch):
    result = {"failed": True, "error": "quota", "failure_reason": "rate_limit"}
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_env")
    assert single_query_exit_code(result) == KANBAN_RATE_LIMIT_EXIT_CODE
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert single_query_exit_code(result) == 1
