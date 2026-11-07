"""A cron run that exhausts empty-response retries is a failed run, not "ok".

A job whose turn ended ``empty_response_exhausted`` kept ``last_status: ok`` in jobs.json:
the agent substitutes leaked reasoning or an explainer as ``final_response``, so the
scheduler's empty-response soft-failure check never fired.
"""
import pytest

from cron.scheduler import _final_response_from_result


def _result(**overrides):
    result = {"final_response": "Some leaked reasoning text", "completed": True, "failed": False,
              "turn_exit_reason": "text_response(finish_reason=stop)"}
    result.update(overrides)
    return result


def test_empty_response_exhausted_raises_even_with_substituted_text():
    with pytest.raises(RuntimeError, match="empty_response_exhausted"):
        _final_response_from_result(
            _result(turn_exit_reason="empty_response_exhausted"), "job1", "Job", None)


def test_normal_text_response_is_still_delivered():
    assert _final_response_from_result(_result(), "job1", "Job", None) == "Some leaked reasoning text"


def test_max_iteration_summary_still_delivers():
    summary = _result(completed=False, turn_exit_reason="max_iterations_reached(60/60)")
    assert _final_response_from_result(summary, "job1", "Job", None) == "Some leaked reasoning text"
