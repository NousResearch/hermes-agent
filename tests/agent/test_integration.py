"""Integration tests: the anti-refusal retry harness is wired into ``perform_api_call``.

The turn loop drives phases by introspecting ``perform_api_call``'s signature and passing
the loop-state fields it names, so these tests call it exactly as the loop would — including
the fields the harness hand-off added (``api_messages`` / ``active_system_prompt`` /
``compression_attempts``). The last test drives the REAL harness through the wrapper to
prove the resend reaches the provider path with the frame-repair context attached.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

# Pre-import at collection: ``perform_api_call`` lazily imports conversation_loop after a
# lane switch, and its bootstrap (hermes-bootstrap marker probe) must not run while the
# home-I/O guard (tests/home_io_guard.py) is active.
import agent.conversation_loop

from agent.anti_refusal_retry import RetryResult
from agent.turn_api_call import perform_api_call

MESSAGES = [{"role": "user", "content": "Hello"}]


def _agent():
    agent = Mock()
    agent.api_mode = "chat"
    agent._model_request_active = None
    agent._pending_redirect_lock = None
    agent._has_pending_redirect = Mock(return_value=False)
    agent.session_id = "test-session"
    agent.provider = "test-provider"
    agent.model = "test-model"
    agent.base_url = "http://test"
    agent.platform = "test"
    return agent


def _call(agent, api_kwargs, _retry):
    return perform_api_call(
        agent=agent,
        api_kwargs=api_kwargs,
        _original_api_kwargs=api_kwargs,
        _llm_middleware_trace=[],
        _moa_prepared_request=None,
        _retry=_retry,
        thinking_spinner=None,
        retry_count=0,
        api_call_count=1,
        api_request_id="test-req",
        effective_task_id="test-task",
        turn_id="test-turn",
        interrupted=False,
        api_messages=api_kwargs["messages"],
        active_system_prompt="system prompt",
        compression_attempts=0,
    )


def test_harness_is_wired_into_perform_api_call():
    agent = _agent()
    mock_response = Mock()
    mock_response.content = "This is a good response."
    api_kwargs = {"messages": MESSAGES, "model": "test-model"}

    with patch("hermes_cli.middleware.run_llm_execution_middleware", return_value=mock_response):
        with patch("agent.anti_refusal_retry.anti_refusal_retry_harness") as mock_harness:
            mock_harness.return_value = RetryResult(
                response=mock_response,
                attempts=1,
                verdicts=[],
                should_fall_through=False,
            )
            result = _call(agent, api_kwargs, _retry=Mock())

    assert mock_harness.called, "Anti-refusal retry harness was not called!"
    call_args = mock_harness.call_args
    assert call_args[1]["agent"] is agent
    assert call_args[1]["initial_response"] is mock_response
    assert call_args[1]["original_messages"] == MESSAGES

    assert result.action == "fallthrough"
    assert result.response is mock_response


def test_lane_switch_arms_the_rebuilt_restart():
    agent = _agent()
    mock_response = Mock()
    mock_response.content = "I cannot help with that."
    api_kwargs = {"messages": MESSAGES, "model": "test-model"}
    retry = Mock()

    with patch("hermes_cli.middleware.run_llm_execution_middleware", return_value=mock_response):
        with patch("agent.anti_refusal_retry.anti_refusal_retry_harness") as mock_harness:
            mock_harness.return_value = RetryResult(
                response=mock_response,
                attempts=3,
                verdicts=[],
                should_fall_through=False,
                frame_repaired=True,
                lane_switched=True,
                restart_on_fallback=True,
            )
            result = _call(agent, api_kwargs, retry)

    assert result.action == "break"
    assert retry.restart_with_rebuilt_messages is True
    assert result.retry_count == 0
    assert result.compression_attempts == 0


def test_real_harness_resend_carries_frame_repair_through_the_wrapper():
    agent = _agent()
    refusal = Mock()
    refusal.content = "I cannot help with that."
    repaired_ok = Mock()
    repaired_ok.content = "Here is the scene..."
    api_kwargs = {"messages": MESSAGES, "model": "test-model"}
    middleware = Mock(side_effect=[refusal, repaired_ok])

    with patch("hermes_cli.middleware.run_llm_execution_middleware", middleware):
        with patch("agent.anti_refusal_retry.detect_soft_refusal") as mock_detect:
            mock_detect.side_effect = [
                Mock(is_soft_refusal=True, judge_score=0.9),
                Mock(is_soft_refusal=False, judge_score=0.2),
            ]
            result = _call(agent, api_kwargs, _retry=Mock())

    assert result.action == "fallthrough"
    assert result.response is repaired_ok
    assert mock_detect.call_count == 2
    assert middleware.call_count == 2

    initial_request = middleware.call_args_list[0][0][0]
    resend_request = middleware.call_args_list[1][0][0]
    assert initial_request["messages"] == MESSAGES
    assert "[FRAME REPAIR" in resend_request["messages"][-1]["content"]
    assert resend_request["messages"][-1]["content"].startswith("Hello")
