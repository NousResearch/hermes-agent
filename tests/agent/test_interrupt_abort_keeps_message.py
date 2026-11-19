"""An interrupt that lands during API-error recovery or a retry backoff keeps its message.

``abort_turn_on_interrupt`` is an early return that never reaches ``finalize_turn``, so its result
dict is the only carrier of ``interrupt_message``. The gateway re-queues that message as the user's
next turn and reads restart/shutdown reasons from it; dropping it loses the user's message.
"""

from __future__ import annotations

import threading

from agent.turn_recovery import interruptible_backoff_sleep
from tools.interrupt import set_interrupt

PLACEHOLDER = "Operation interrupted: retrying API call after error (retry 1/3)."


def _bare_agent():
    """A real ``AIAgent`` interrupt surface (``interrupt`` / ``clear_interrupt``); persistence and
    printing are no-ops."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._pending_redirect = None
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent._vprint = lambda *a, **k: None
    agent._persist_session = lambda messages, conversation_history: None
    return agent


def _sleep_through_interrupt(agent):
    return interruptible_backoff_sleep(
        agent, 30.0, None, messages=[{"role": "user", "content": "summarise the report"}],
        conversation_history=None, api_call_count=2,
        abort_message="Interrupt detected during retry wait, aborting.",
        interrupt_text=PLACEHOLDER, activity_label="error retry backoff (1/3)",
    )


def test_a_message_sent_during_retry_backoff_is_returned_for_requeue():
    agent = _bare_agent()
    try:
        agent.interrupt("actually, use the March numbers")

        result = _sleep_through_interrupt(agent)

        assert result["interrupted"] is True and result["completed"] is False
        assert result["interrupt_message"] == "actually, use the March numbers"
        assert result["final_response"] == PLACEHOLDER
        assert agent._interrupt_requested is False and agent._interrupt_message is None
    finally:
        set_interrupt(False)


def test_a_bare_stop_during_retry_backoff_carries_no_message():
    agent = _bare_agent()
    try:
        agent.interrupt()

        assert "interrupt_message" not in _sleep_through_interrupt(agent)
    finally:
        set_interrupt(False)
