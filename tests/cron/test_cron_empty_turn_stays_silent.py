"""An abnormal empty turn (``empty_response_exhausted``) is delivered as silence, not as a
'⚠️ Cron ... failed' alert: the finalizer stamps ``failure_reason`` for the UI but keeps
``failed`` False, and the scheduler recognises the model-named explainer text."""

import pytest

from agent.turn_explainers import TurnExplainersMixin
from cron.scheduler import _final_response_from_result


class _AIAgent:
    _format_turn_completion_explanation = staticmethod(TurnExplainersMixin._format_turn_completion_explanation)


def test_empty_response_exhausted_turn_delivers_silence_not_a_failure_alert():
    explainer = TurnExplainersMixin._format_turn_completion_explanation("empty_response_exhausted", model="llama3")
    result = {
        # finalize_turn leaves ``failed`` False for this exit reason (and ``completed`` follows the
        # ordinary rule); only the descriptor code is stamped.
        "final_response": explainer, "failed": False, "completed": True, "model": "llama3",
        "turn_exit_reason": "empty_response_exhausted", "failure_reason": "empty_response",
        "failure_retryable": True, "messages": [], "api_calls": 3,
    }
    assert _final_response_from_result(result, "job1", "Morning brief", _AIAgent) == ""


def test_a_failed_turn_still_raises_so_the_job_is_marked_failed():
    result = {"final_response": "stopped", "failed": True, "completed": False, "error": "boom",
              "turn_exit_reason": "repeated_outer_errors(x)", "messages": [], "api_calls": 3}
    with pytest.raises(RuntimeError):
        _final_response_from_result(result, "job1", "Morning brief", _AIAgent)


def test_a_tool_guardrail_halt_is_a_failed_run_not_a_report():
    result = {"final_response": "I stopped retrying because I kept running web_search 10 times without making progress.",
              "failed": False, "completed": True, "messages": [], "api_calls": 12,
              "guardrail": {"action": "block", "code": "loop_web_search_cap", "message": "Blocked web_search",
                            "tool_name": "web_search", "count": 10}}
    with pytest.raises(RuntimeError, match="loop_web_search_cap"):
        _final_response_from_result(result, "job1", "Monthly feed", _AIAgent)
