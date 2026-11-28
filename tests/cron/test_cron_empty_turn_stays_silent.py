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


def test_stop_finished_markdown_stub_raises_instead_of_being_delivered():
    """A symbol-plus-markdown stub is not a usable completed cron report."""
    result = {
        "final_response": "🔍 **", "failed": False, "completed": True,
        "turn_exit_reason": "text_response(finish_reason=stop)",
        "messages": [{"role": "user", "content": "Send the morning report"}],
    }

    with pytest.raises(RuntimeError, match="degenerate near-empty"):
        _final_response_from_result(result, "job1", "Morning brief", _AIAgent)


@pytest.mark.parametrize("final_response", ["Done", "✅", "[SILENT]", "{}"])
def test_stop_finished_legitimate_short_output_fails_open(final_response):
    result = {
        "final_response": final_response, "failed": False, "completed": True,
        "turn_exit_reason": "text_response(finish_reason=stop)",
        "messages": [{"role": "user", "content": "Send the morning report"}],
    }

    assert _final_response_from_result(result, "job1", "Morning brief", _AIAgent) == final_response


def test_short_response_without_stop_metadata_fails_open():
    result = {
        "final_response": "🔍 **", "failed": False, "completed": True,
        "messages": [{"role": "user", "content": "Send the morning report"}],
    }

    assert _final_response_from_result(result, "job1", "Morning brief", _AIAgent) == "🔍 **"
