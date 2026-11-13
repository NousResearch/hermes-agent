"""A reasoning-channel repetition cut gets one reasoning-off retry.

The visible answer is often still unwritten when the thinking stream loops.
Replaying the looped text would re-enter it, but the tool results already in
the transcript are enough for one fresh call with reasoning disabled. A second
loop, or a loop in the visible text, still discards the partial.
"""

from types import SimpleNamespace

from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import _REPETITION_STREAM_CUT, recover_from_truncation


class _Agent:
    log_prefix = ""
    _ephemeral_reasoning_off = False

    def _vprint(self, *_args, **_kwargs):
        return None

    def _flush_status_buffer(self):
        return None

    def _cleanup_task_resources(self, _task_id):
        return None

    def _persist_session(self, _messages, _history):
        return None


def _response(channel):
    response = SimpleNamespace(_runaway_repetition=True)
    if channel is not None:
        response._runaway_channel = channel
    return response


def _recover(response, retry, messages):
    return recover_from_truncation(
        _Agent(), response, "length", retry,
        messages=messages, conversation_history=None, api_kwargs={},
        api_call_count=2, effective_task_id=None, current_turn_user_idx=0,
        length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
    )


def _messages():
    return [
        {"role": "user", "content": "how many"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "rows"},
    ]


def test_reasoning_loop_retries_once_without_touching_tool_results():
    agent = _Agent()
    retry = TurnRetryState()
    messages = _messages()
    verdict = recover_from_truncation(
        agent, _response("reasoning"), "length", retry,
        messages=messages, conversation_history=None, api_kwargs={},
        api_call_count=2, effective_task_id=None, current_turn_user_idx=0,
        length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
    )

    assert verdict.action == "continue"
    assert retry.reasoning_runaway_retry_attempted is True
    assert agent._ephemeral_reasoning_off is True
    assert messages == _messages()


def test_second_reasoning_loop_discards_the_partial():
    retry = TurnRetryState(reasoning_runaway_retry_attempted=True)
    verdict = _recover(_response("reasoning"), retry, _messages())

    assert verdict.action == "return"
    assert "Repetition Detected" in (verdict.result or {})["final_response"]
    assert (verdict.result or {})["final_response"] == _REPETITION_STREAM_CUT[1]


def test_visible_content_loop_is_not_retried():
    retry = TurnRetryState()
    verdict = _recover(_response("content"), retry, _messages())

    assert verdict.action == "return"
    assert retry.reasoning_runaway_retry_attempted is False
    assert "Repetition Detected" in (verdict.result or {})["final_response"]


def test_unmarked_channel_keeps_the_terminal_cut():
    retry = TurnRetryState()
    verdict = _recover(_response(None), retry, _messages())

    assert verdict.action == "return"
    assert retry.reasoning_runaway_retry_attempted is False
