"""An abort during an API-error retry backoff must still say WHO stopped the turn.

``abort_turn_on_interrupt`` is an early return: it never reaches ``finalize_turn``, so its result
dict is the only carrier of ``turn_exit_reason``. Without it a lease/watchdog abort surfaces as
whichever retry placeholder was in flight — "Operation interrupted: retrying API call after error
(retry 1/3)" — and books a system stop as a user stop (#112647, #139959, #142161).
"""

from agent.turn_recovery import abort_turn_on_interrupt, interruptible_backoff_sleep

PLACEHOLDER = "Operation interrupted: retrying API call after error (retry 1/3)."
LEASE_MESSAGE = "Session turn lease could not be refreshed; stopping to protect the transcript."


class _StubAgent:
    """The slice of ``AIAgent`` these two helpers touch, with the real ``clear_interrupt`` wipe."""

    log_prefix = ""
    quiet_mode = True

    def __init__(self, reason=None, message=None):
        self._interrupt_requested = True
        self._tool_interrupt_reason = reason
        self._interrupt_message = message
        self.persisted = False

    def _vprint(self, *args, **kwargs):
        return None

    def _safe_print(self, *args, **kwargs):
        return None

    def _touch_activity(self, label):
        return None

    def _adopt_nous_key_before_expiry(self):
        return None

    def _drop_trailing_empty_response_scaffolding(self, messages):
        return None

    def _persist_session(self, messages, conversation_history):
        self.persisted = True

    def clear_interrupt(self, *, preserve_redirect=False, hard_cancel=False):
        # Mirrors InterruptControlMixin.clear_interrupt: the cause is dropped with the flag.
        self._interrupt_requested = False
        self._interrupt_message = self._tool_interrupt_reason = None
        return True


def _abort(agent):
    return abort_turn_on_interrupt(
        agent, messages=[{"role": "user", "content": "hi"}], conversation_history=None,
        api_call_count=7, abort_message="Interrupt detected during retry wait, aborting.",
        interrupt_text=PLACEHOLDER,
    )


def test_a_lease_lost_abort_keeps_the_issuer_in_the_exit_reason():
    agent = _StubAgent(reason="session turn lease lost", message=LEASE_MESSAGE)

    result = _abort(agent)

    assert result["turn_exit_reason"] == "interrupted_during_api_call(session_turn_lease_lost)"
    # The phase text still says what was in flight; the exit reason says why it stopped.
    assert result["final_response"] == PLACEHOLDER
    assert result["interrupted"] is True and result["completed"] is False
    assert agent.persisted is True and agent._interrupt_requested is False


def test_a_user_stop_stays_a_bare_interrupt_without_an_issuer():
    agent = _StubAgent(reason="user interrupt")

    assert _abort(agent)["turn_exit_reason"] == "interrupted_during_api_call"


def test_the_retry_backoff_path_records_the_reason_too():
    """The real call site: the backoff sleep is where the incident was aborted."""
    agent = _StubAgent(reason="session turn lease lost", message=LEASE_MESSAGE)

    result = interruptible_backoff_sleep(
        agent, 5.0, None, messages=[{"role": "user", "content": "hi"}], conversation_history=None,
        api_call_count=7, abort_message="Interrupt detected during retry wait, aborting.",
        interrupt_text=PLACEHOLDER, activity_label="error retry backoff (1/3)",
    )

    assert result["turn_exit_reason"] == "interrupted_during_api_call(session_turn_lease_lost)"
