"""Behavioral contracts for one-shot fallback after a runaway response."""

from types import SimpleNamespace
from unittest.mock import Mock

from agent.error_classifier import FailoverReason
from agent.repetition_recovery import repetition_tool_recovery_mode, tools_for_repetition_retry
from agent.turn_tool_validation import validate_tool_calls
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import recover_from_truncation
from run_agent import AIAgent


def _call(call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "terminal", "arguments": '{"command":"true"}'},
    }


def test_repetition_recovery_keeps_tools_when_no_action_ran():
    assert repetition_tool_recovery_mode([{"role": "user", "content": "run it"}], 0) == "tools"


def test_repetition_recovery_is_answer_only_after_completed_tool_results():
    messages = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "tool_calls": [_call()]},
        {"role": "tool", "tool_call_id": "call-1", "name": "terminal", "content": "done"},
    ]

    assert repetition_tool_recovery_mode(messages, 0) == "answer_only"


def test_repetition_recovery_fails_closed_for_unresolved_tool_calls():
    messages = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "tool_calls": [_call()]},
    ]

    assert repetition_tool_recovery_mode(messages, 0) == "unsafe"


def test_repetition_recovery_fails_closed_for_unknown_tool_effects():
    messages = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "tool_calls": [_call()]},
        {
            "role": "tool", "tool_call_id": "call-1", "name": "terminal",
            "content": "timed out", "effect_disposition": "unknown",
        },
    ]

    assert repetition_tool_recovery_mode(messages, 0) == "unsafe"


def _recover(messages, *, fallback=True, already_retried=False):
    agent = SimpleNamespace(
        log_prefix="", _fallback_index=0, _fallback_chain=[{"provider": "healthy"}],
        _repetition_fallback_attempted=already_retried,
        _vprint=Mock(), _emit_diagnostic_status=Mock(), _try_activate_fallback=Mock(return_value=fallback),
        _cleanup_task_resources=Mock(), _persist_session=Mock(),
    )
    retry = TurnRetryState()
    response = SimpleNamespace(_runaway_repetition=True)
    verdict = recover_from_truncation(
        agent, response, "length", retry, messages=messages,
        conversation_history=messages, api_kwargs={}, api_call_count=1,
        effective_task_id=None, current_turn_user_idx=0, length_continue_retries=0,
        truncated_response_parts=[], truncated_tool_call_retries=0,
        retry_count=0, compression_attempts=0,
    )
    return agent, retry, verdict


def test_repetition_fallback_preserves_completed_tool_context_and_disables_tools():
    messages = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "tool_calls": [_call()]},
        {"role": "tool", "tool_call_id": "call-1", "name": "terminal", "content": "done"},
    ]
    original = [dict(message) for message in messages]

    agent, retry, verdict = _recover(messages)

    assert verdict.action == "break"
    assert messages == original
    assert retry.restart_with_rebuilt_messages is True
    assert agent._repetition_fallback_attempted is True
    assert tools_for_repetition_retry(agent, [{"function": {"name": "terminal"}}]) == []
    agent._try_activate_fallback.assert_called_once_with(reason=FailoverReason.incomplete_response)


def test_repetition_fallback_keeps_tools_when_no_tool_action_ran():
    tools = [{"function": {"name": "terminal"}}]
    agent, retry, verdict = _recover([{"role": "user", "content": "run it"}])

    assert verdict.action == "break"
    assert retry.restart_with_rebuilt_messages is True
    assert tools_for_repetition_retry(agent, tools) == tools


def test_repetition_fallback_does_not_retry_twice():
    agent, _retry, verdict = _recover(
        [{"role": "user", "content": "run it"}], already_retried=True,
    )

    assert verdict.action == "return"
    agent._try_activate_fallback.assert_not_called()


def test_repetition_fallback_does_not_retry_with_an_unresolved_tool_call():
    messages = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "tool_calls": [_call()]},
    ]
    agent, _retry, verdict = _recover(messages)

    assert verdict.action == "return"
    agent._try_activate_fallback.assert_not_called()


def test_answer_only_fallback_never_executes_a_provider_tool_call():
    agent = AIAgent.__new__(AIAgent)
    agent._repetition_fallback_no_tools = True
    agent.valid_tool_names = {"terminal"}
    agent._invalid_tool_retries = agent._invalid_json_retries = 0
    agent._persist_session = Mock()
    agent._build_assistant_message = lambda message, _finish: {
        "role": "assistant", "content": message.content,
        "tool_calls": [{
            "id": call.id, "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        } for call in message.tool_calls],
    }
    call = SimpleNamespace(
        id="call-repeated", type="function",
        function=SimpleNamespace(name="terminal", arguments='{"command":"true"}'),
    )
    messages = [{"role": "user", "content": "run it"}]

    verdict = validate_tool_calls(
        agent, SimpleNamespace(content="", tool_calls=[call]), "tool_calls",
        messages=messages, conversation_history=messages,
        api_call_count=2, effective_task_id=None,
    )

    assert verdict.action == "return"
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert "Not executed" in messages[-2]["content"]
    agent._persist_session.assert_called_once()
