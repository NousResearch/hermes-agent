"""Tests for ``agent/turn_api_call.py::handle_api_interrupt`` — the plain (non-redirect)
interrupt that lands mid provider call and records the streamed partial as the interrupted
assistant row."""
from __future__ import annotations

import threading
import time

from agent.agent_runtime_helpers import _INTERRUPTED_PLACEHOLDER
from agent.repetition_guard import REPETITION_LOOP_INTERRUPTED
from agent.turn_api_call import handle_api_interrupt
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent


def _bare_agent(streamed: str) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._current_streamed_assistant_text = streamed
    agent._strip_think_blocks = lambda content: content
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent.thinking_callback = None
    agent._print_fn = lambda *args, **kwargs: None
    agent._persist_session = lambda *args, **kwargs: None
    return agent


def _interrupt(streamed: str):
    messages = [{"role": "user", "content": "start"}]
    verdict = handle_api_interrupt(
        _bare_agent(streamed), _retry=TurnRetryState(), thinking_spinner=None, messages=messages,
        conversation_history=[], api_start_time=time.time(), interrupted=False, final_response=None,
    )
    return messages, verdict


def test_repetition_dominated_partial_is_not_kept_as_the_interrupted_row():
    """A looped partial replayed as the interrupted assistant row re-seeds the loop on the next
    turn (#112764): the row keeps the neutral placeholder and the user is told what happened."""
    looped = "I. " * 1941

    messages, verdict = _interrupt(looped)

    # Same hidden shape as the redirect placeholder: no visible bubble in transcript replays.
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == ""
    assert messages[-1]["display_kind"] == "hidden"
    assert messages[-1]["api_content"] == _INTERRUPTED_PLACEHOLDER
    assert verdict.final_response == REPETITION_LOOP_INTERRUPTED
    assert "I. I. I." not in verdict.final_response


def test_distinct_batch_rows_are_not_mistaken_for_a_loop():
    """Legitimately repetitive output (distinct INSERT rows sharing a long prefix) trips the
    window scan but is not a runaway loop: the partial must stay the interrupted row and must
    not be relabelled as a degenerate reply."""
    rows = "\n".join(
        f"INSERT INTO users (id, name, email, created_at) VALUES ({i}, 'user{i}', 'user{i}@example.com', NOW());"
        for i in range(12)
    )
    messages, verdict = _interrupt(rows)

    assert (messages[-1]["role"], messages[-1]["content"]) == ("assistant", rows)
    assert verdict.final_response == rows


def test_ordinary_partial_is_kept_as_the_interrupted_row():
    messages, verdict = _interrupt("Visible draft.")

    assert (messages[-1]["role"], messages[-1]["content"]) == ("assistant", "Visible draft.")
    assert verdict.final_response == "Visible draft."
