"""handle_content_policy_refusal: an Anthropic refusal with an empty body still tells the user why."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import handle_content_policy_refusal


def _agent():
    import agent.transports.anthropic
    from agent.transports import get_transport

    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model, agent.log_prefix = "anthropic_messages", "anthropic", "claude", ""
    agent._is_anthropic_oauth = False
    agent._get_transport.return_value = get_transport("anthropic_messages")
    agent._has_pending_fallback.return_value = False
    agent._try_activate_fallback.return_value = False
    agent._extract_reasoning.return_value = ""
    return agent


def test_empty_refusal_reports_stop_details_explanation():
    response = SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "general_harms", "explanation": "classifier halt"},
    )
    verdict = handle_content_policy_refusal(
        _agent(), response, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
        api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
    )
    assert verdict.action == "return"
    assert "classifier halt" in verdict.result["error"]


def test_has_explicit_refusal_distinguishes_stream_drop_from_real_refusal():
    from agent.turn_truncation import has_explicit_refusal

    drop_resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(refusal=None))], stop_reason=None)
    drop_norm = SimpleNamespace(provider_data={})
    assert has_explicit_refusal(drop_resp, drop_norm) is False

    openai_refusal = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(refusal="I cannot help"))], stop_reason=None)
    assert has_explicit_refusal(openai_refusal, drop_norm) is True

    anthropic_refusal = SimpleNamespace(choices=None, stop_reason="refusal")
    assert has_explicit_refusal(anthropic_refusal, drop_norm) is True

    provider_data_refusal = SimpleNamespace(provider_data={"stop_details": {"type": "refusal"}})
    assert has_explicit_refusal(drop_resp, provider_data_refusal) is True

    responses_api_refusal = SimpleNamespace(
        choices=None, stop_reason=None, status="incomplete", incomplete_details={"reason": "content_filter"},
    )
    assert has_explicit_refusal(responses_api_refusal, drop_norm) is True


def test_empty_content_filter_without_explicit_refusal_does_not_leak_reasoning():
    agent = _agent()
    agent._extract_reasoning.return_value = "internal chain of thought before stream dropped"
    drop_resp = SimpleNamespace(
        content=[], stop_reason="end_turn", usage=None, model="claude", stop_details=None,
    )
    verdict = handle_content_policy_refusal(
        agent, drop_resp, TurnRetryState(), thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        active_system_prompt=None, conversation_history=[], api_call_count=1, effective_task_id="t", turn_id="u",
        api_request_id="r", api_start_time=0.0, retry_count=0, max_retries=0,
    )
    assert verdict.action == "return"
    assert "internal chain of thought" not in verdict.result["error"]


def _chat_agent():
    import agent.transports.chat_completions
    from agent.transports import get_transport

    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model, agent.log_prefix = "chat_completions", "openai", "gpt-4o", ""
    agent._is_anthropic_oauth = False
    agent._get_transport.return_value = get_transport("chat_completions")
    agent._has_pending_fallback.return_value = False
    agent._try_activate_fallback.return_value = False
    agent._extract_reasoning.return_value = "partial internal thought"
    agent._should_treat_stop_as_truncated.return_value = False
    return agent


def test_chat_completions_legitimate_content_filter_without_extra_metadata_is_not_retried():
    """(a) Standards-compliant Chat Completions content_filter with content=None and refusal=None
    must enter refusal handling immediately, not be retried as a transient stream drop."""
    from agent.turn_response_check import check_api_response

    agent = _chat_agent()
    msg = SimpleNamespace(role="assistant", content=None, refusal=None, tool_calls=None, reasoning_content=None)
    response = SimpleNamespace(
        id="chatcmpl-real-filter",
        model="gpt-4o",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        choices=[SimpleNamespace(index=0, message=msg, finish_reason="content_filter")],
    )
    verdict = check_api_response(
        agent, response=response, _retry=TurnRetryState(), thinking_spinner=None, messages=[],
        api_messages=[], api_kwargs={}, active_system_prompt=None, conversation_history=[],
        finish_reason="content_filter", retry_count=0, max_retries=2, compression_attempts=0,
        max_compression_attempts=2, length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, current_turn_user_idx=0, api_call_count=1,
        api_request_id="r1", api_start_time=0.0, effective_task_id="t1", turn_id="u1",
        _preflight_compression_blocked=False, _last_preflight_pressure=None,
    )
    assert verdict.action == "return"
    assert verdict.retry_count == 0
    assert verdict.result["failed"] is True


def test_chat_completions_positively_flagged_proxy_stream_drop_retries_without_leaking_reasoning(monkeypatch):
    """(b) A positively flagged proxy stream drop (_proxy_stream_drop=True) is retried on the same
    model while retries remain, and suppresses partial reasoning if retries are exhausted."""
    import time
    from agent.turn_response_check import check_api_response

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    agent = _chat_agent()
    msg = SimpleNamespace(
        role="assistant", content=None, refusal=None, tool_calls=None,
        reasoning_content="partial internal thought before proxy stream cut",
    )
    response = SimpleNamespace(
        id="stream-drop-1",
        model="claude-via-proxy",
        usage=None,
        choices=[SimpleNamespace(index=0, message=msg, finish_reason="content_filter")],
        _proxy_stream_drop=True,
    )
    first = check_api_response(
        agent, response=response, _retry=TurnRetryState(), thinking_spinner=None, messages=[],
        api_messages=[], api_kwargs={}, active_system_prompt=None, conversation_history=[],
        finish_reason="content_filter", retry_count=0, max_retries=2, compression_attempts=0,
        max_compression_attempts=2, length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, current_turn_user_idx=0, api_call_count=1,
        api_request_id="r2", api_start_time=0.0, effective_task_id="t2", turn_id="u2",
        _preflight_compression_blocked=False, _last_preflight_pressure=None,
    )
    assert first.action == "continue"
    assert first.retry_count == 1

    exhausted = check_api_response(
        agent, response=response, _retry=TurnRetryState(), thinking_spinner=None, messages=[],
        api_messages=[], api_kwargs={}, active_system_prompt=None, conversation_history=[],
        finish_reason="content_filter", retry_count=1, max_retries=2, compression_attempts=0,
        max_compression_attempts=2, length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, current_turn_user_idx=0, api_call_count=2,
        api_request_id="r3", api_start_time=0.0, effective_task_id="t2", turn_id="u2",
        _preflight_compression_blocked=False, _last_preflight_pressure=None,
    )
    assert exhausted.action == "return"
    assert "partial internal thought" not in exhausted.result["error"]
