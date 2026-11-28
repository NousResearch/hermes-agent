"""A cron job killed by a system producer names that producer, not just the phase in flight.

``last_status`` used to read "RuntimeError: Operation interrupted: retrying API call after error
(retry 1/3)." — the retry that was running when the turn-lease watchdog stopped the turn. The cause
lives in ``turn_exit_reason``; the failure text now carries both (#139959).
"""

import pytest

from agent.turn_recovery import abort_turn_on_interrupt
from cron.scheduler import _final_response_from_result

PLACEHOLDER = "Operation interrupted: retrying API call after error (retry 1/3)."
REASON = "interrupted_during_api_call(session_turn_lease_lost)"


class _AIAgent:
    pass


class _StubAgent:
    log_prefix = ""
    quiet_mode = True

    def _vprint(self, *args, **kwargs):
        return None

    def _drop_trailing_empty_response_scaffolding(self, messages):
        return None

    def _persist_session(self, messages, conversation_history):
        return None

    def clear_interrupt(self, *, preserve_redirect=False, hard_cancel=False):
        self._interrupt_requested = False
        self._interrupt_message = self._tool_interrupt_reason = None
        return True


def _aborted_result():
    agent = _StubAgent()
    agent._interrupt_requested = True
    agent._tool_interrupt_reason = "session turn lease lost"
    agent._interrupt_message = "Session turn lease could not be refreshed; stopping to protect the transcript."
    return abort_turn_on_interrupt(
        agent, messages=[], conversation_history=None, api_call_count=1,
        abort_message="Interrupt detected during retry wait, aborting.", interrupt_text=PLACEHOLDER,
    )


def test_a_watchdog_abort_names_the_cause_in_the_job_failure():
    result = _aborted_result()

    with pytest.raises(RuntimeError) as excinfo:
        _final_response_from_result(result, "job1", "tech-daily", _AIAgent)

    text = str(excinfo.value)
    assert PLACEHOLDER in text and REASON in text


def test_a_turn_whose_text_already_names_the_reason_is_not_doubled():
    result = {"final_response": f"stopped: {REASON}", "completed": False, "interrupted": True,
              "turn_exit_reason": REASON, "messages": [], "api_calls": 1}

    with pytest.raises(RuntimeError) as excinfo:
        _final_response_from_result(result, "job1", "Morning brief", _AIAgent)

    assert str(excinfo.value).count(REASON) == 1


def test_a_plain_failure_keeps_its_own_error_text():
    result = {"final_response": "stopped", "failed": True, "completed": False, "error": "boom",
              "messages": [], "api_calls": 3}

    with pytest.raises(RuntimeError) as excinfo:
        _final_response_from_result(result, "job1", "Morning brief", _AIAgent)

    assert str(excinfo.value) == "boom"
