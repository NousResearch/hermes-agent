"""Unit test: _retry_truncated_tool_call refuses immediately when the prompt filled the
context window — mirrors #106571's _continue_text ceiling exit for the tool-call path.
A max_tokens boost cannot help when there is no room for output, so the 4 retries must
not burn; the turn ends naming the context window as the cause."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_truncation import (
    _retry_truncated_tool_call,
    _Trunc,
    _CONTEXT_OVERFLOW_PARTIAL_FINAL,
)


def _make_state(*, window_filled=None, truncated_tool_call_retries=0):
    agent = SimpleNamespace(
        max_tokens=4096,
        log_prefix="",
        _flush_status_buffer=MagicMock(),
        _vprint=MagicMock(),
        _buffer_vprint=MagicMock(),
        _cleanup_task_resources=MagicMock(),
        _persist_session=MagicMock(),
        _requested_output_cap_from_api_kwargs=MagicMock(return_value=None),
        _ephemeral_max_output_tokens=None,
    )
    st = _Trunc(
        agent=agent,
        response=SimpleNamespace(id="resp", usage=SimpleNamespace(prompt_tokens=32638)),
        finish_reason="length",
        conversation_history=[],
        api_call_count=1,
        effective_task_id="task-1",
        current_turn_user_idx=0,
        messages=[],
        length_continue_retries=0,
        truncated_response_parts=[],
        truncated_tool_call_retries=truncated_tool_call_retries,
        retry_count=0,
        compression_attempts=0,
        window_filled=window_filled,
    )
    return st, agent


def test_window_filled_refuses_without_retrying():
    """When the prompt filled the window, no retry — surface the real cause."""
    st, agent = _make_state(window_filled=(32638, 32768), truncated_tool_call_retries=0)
    with patch("agent.turn_truncation.close_interrupted_tool_sequence"):
        verdict = _retry_truncated_tool_call(st, api_kwargs={})
    # No retry happened.
    assert st.truncated_tool_call_retries == 0
    assert verdict.action == "return"  # end_turn stamps "return"
    # max_tokens was never boosted.
    assert agent._ephemeral_max_output_tokens is None
    # The real cause (context window) is surfaced, not the generic truncated-final.
    assert verdict.result is not None
    assert verdict.result["final_response"] == _CONTEXT_OVERFLOW_PARTIAL_FINAL
    # The vprint named the context window with the measured numbers.
    vprint_text = agent._vprint.call_args[0][0]
    assert "context window" in vprint_text.lower()
    assert "32,638" in vprint_text and "32,768" in vprint_text


def test_headroom_still_retries_with_boosted_max_tokens():
    """With headroom (window not filled), the retry path is unchanged — max_tokens boosted."""
    st, agent = _make_state(window_filled=None, truncated_tool_call_retries=0)
    with patch("agent.turn_truncation.close_interrupted_tool_sequence"):
        verdict = _retry_truncated_tool_call(st, api_kwargs={})
    assert st.truncated_tool_call_retries == 1
    assert verdict.action == "continue"  # st.done("continue")
    # 4096 * 2^1 = 8192, capped at min(8192, max(32768, 0)) = 8192.
    assert agent._ephemeral_max_output_tokens == 8192
    # No end-turn partial was produced.
    assert verdict.result is None
