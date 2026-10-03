"""ChatGPT plan requests retain tool history and require successful stream completion."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest


def test_chatgpt_wire_contract_preserves_tools_and_history_without_mutating_other_routes():
    from agent.codex_runtime import _sanitize_consumer_codex_request

    tool = {"type": "function", "name": "read_file", "parameters": {"type": "object"}, "strict": False}
    history = [
        {"type": "message", "role": "system", "content": "Be concise."},
        {"role": "user", "content": "Read the file."},
        {"type": "function_call", "call_id": "call_read", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_read", "output": "file contents"},
    ]
    request = {
        "model": "gpt-test", "input": history, "tools": [tool],
        "instructions": "Keep the complete history.", "temperature": 0.5,
        "max_output_tokens": 200, "previous_response_id": "resp_old",
        "tool_choice": {"type": "function", "name": "read_file"},
        "extra_body": {"prompt_cache_retention": "24h", "store": True, "stream": False},
    }
    original = deepcopy(request)
    siwc = SimpleNamespace(provider="openai-chatgpt")
    prepared = _sanitize_consumer_codex_request(siwc, request)
    assert prepared["store"] is False and prepared["stream"] is True
    assert not {"temperature", "max_output_tokens", "previous_response_id"} & prepared.keys()
    assert prepared.get("extra_body", {}) == {}
    namespace = prepared["tools"][0]
    assert namespace["type"] == "namespace" and namespace["tools"] == [tool]
    assert prepared["input"][0]["role"] == "developer"
    assert prepared["input"][2]["namespace"] == namespace["name"]
    assert prepared["tool_choice"]["namespace"] == namespace["name"]
    assert prepared["input"][1] == history[1] and prepared["input"][3] == history[3]
    assert request == original
    assert _sanitize_consumer_codex_request(SimpleNamespace(provider="openai"), request) == original
    # Reapplying at the final wire boundary must not nest namespaces or lose history.
    assert _sanitize_consumer_codex_request(siwc, prepared) == prepared


@pytest.mark.parametrize("terminal", [None, "incomplete", "failed", "completed"])
def test_chatgpt_stream_requires_completed_even_after_text_or_tool_output(terminal):
    from agent.codex_runtime import _consume_codex_event_stream
    from run_agent import _StreamErrorEvent

    events = [{"type": "response.output_text.delta", "delta": "partial"}]
    if terminal:
        events.append({"type": f"response.{terminal}", "response": {
            "status": terminal, "error": {"code": "subscription_sharing_usage_limit_exceeded",
                                            "message": "App allowance exhausted", "param": "usage"},
            "incomplete_details": {"reason": "content_filter"},
        }})
    if terminal == "completed":
        result = _consume_codex_event_stream(events, model="gpt-test", require_completed=True)
        assert result.status == "completed" and result.output_text == "partial"
    elif terminal == "failed":
        with pytest.raises(_StreamErrorEvent) as raised:
            _consume_codex_event_stream(events, model="gpt-test", require_completed=True)
        assert raised.value.code == "subscription_sharing_usage_limit_exceeded"
        assert raised.value.param == "usage"
        assert raised.value.body["error"]["message"] == "App allowance exhausted"
    else:
        with pytest.raises(RuntimeError, match="response.completed"):
            _consume_codex_event_stream(events, model="gpt-test", require_completed=True)
    if terminal is None:
        # Existing compatible Responses providers retain their partial-stream behavior.
        assert _consume_codex_event_stream(events, model="gpt-test").status == "completed"


def test_namespaced_tool_call_executes_locally_and_replays_on_the_next_request(tmp_path):
    from openai.types.responses import ResponseFunctionToolCall

    from agent.chatgpt_responses import prepare_chatgpt_request
    from agent.codex_runtime import _consume_codex_event_stream
    from agent.transports.codex import ResponsesApiTransport
    from model_tools import handle_function_call

    path = tmp_path / "note.txt"
    path.write_text("The result is forty-two.", encoding="utf-8")
    tools = [{"type": "function", "function": {
        "name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }}]
    messages = [{"role": "user", "content": "Read the note."}]
    transport = ResponsesApiTransport()

    def request():
        kwargs = transport.build_kwargs("gpt-test", messages, tools, provider="openai-chatgpt",
                                        base_url="https://api.openai.com/v1")
        return prepare_chatgpt_request(transport.preflight_kwargs(kwargs))

    first = request()
    namespace = first["tools"][0]["name"]
    call = ResponseFunctionToolCall(type="function_call", call_id="call_read", name="read_file",
                                   namespace=namespace, arguments=json.dumps({"path": str(path)}), status="completed")
    final = _consume_codex_event_stream([
        SimpleNamespace(type="response.output_item.done", item=call),
        {"type": "response.completed", "response": {"status": "completed"}},
    ], model="gpt-test", require_completed=True)
    normalized = transport.normalize_response(final)
    tool_call = normalized.tool_calls[0]
    output = handle_function_call(tool_call.name, json.loads(tool_call.arguments), task_id="chatgpt-wire-contract")
    assert "forty-two" in output
    messages.extend([
        {"role": "assistant", "tool_calls": [{"id": tool_call.id, "type": "function", "function": {
            "name": tool_call.name, "arguments": tool_call.arguments,
        }}]},
        {"role": "tool", "tool_call_id": tool_call.id, "content": output},
    ])
    second = request()
    replay = next(item for item in second["input"] if item.get("type") == "function_call")
    result = next(item for item in second["input"] if item.get("type") == "function_call_output")
    assert replay["namespace"] == namespace and replay["name"] == tool_call.name
    assert replay["call_id"] == result["call_id"] and result["output"] == output
    completed = _consume_codex_event_stream([
        {"type": "response.output_text.delta", "delta": "The note says forty-two."},
        {"type": "response.completed", "response": {"status": "completed"}},
    ], model="gpt-test", require_completed=True)
    assert transport.normalize_response(completed).content == "The note says forty-two."


@pytest.mark.parametrize("code,status", [
    ("subscription_sharing_user_not_eligible", 403),
    ("subscription_sharing_usage_limit_exceeded", 429),
    ("subscription_sharing_usage_limit_exceeded", None),  # Late SSE errors have no HTTP status.
    ("subscription_sharing_unsupported_capability", 400),
    ("subscription_sharing_route_not_supported", 403),
    ("subscription_sharing_invalid_user", 401),
    ("chatpass_v2_scope_not_authorized", 403),
    ("chatpass_v2_invalid_authorization_context", 403),
    (None, 401),
    (None, 403),
])
def test_chatgpt_terminal_errors_never_retry_rotate_or_switch_billing(code, status, monkeypatch):
    from unittest.mock import Mock

    from agent.agent_runtime_helpers import recover_with_credential_pool
    from agent.error_classifier import RETRYABLE_CLIENT_REASONS, FailoverReason, classify_api_error
    from agent.turn_api_error import settle_unrecovered_error
    from run_agent import _StreamErrorEvent

    error = _StreamErrorEvent("ChatGPT plan request rejected", code=code, param="tools", status_code=status)
    classified = classify_api_error(error, provider="openai-chatgpt")
    assert classified.reason == FailoverReason.provider_policy_blocked
    assert classified.reason not in RETRYABLE_CLIENT_REASONS
    assert not classified.retryable and not classified.should_rotate_credential
    assert not classified.should_fallback and not classified.is_auth
    pool = Mock(provider="openai-chatgpt")
    agent = SimpleNamespace(_credential_pool=pool, provider="openai-chatgpt", api_key="selected-account",
                            base_url="https://api.openai.com/v1", _credential_pool_entry_id="account-1")
    assert recover_with_credential_pool(agent, status_code=status, has_retried_429=False,
                                        classified_reason=classified.reason) == (False, False)
    pool.mark_exhausted_and_rotate.assert_not_called()
    pool.refresh_current.assert_not_called()
    assert error.body["error"]["code"] == code and error.body["error"]["param"] == "tools"
    # Exercise the real terminal decision with a fallback available: false hints
    # alone do not prevent the generic rate-limit/auth families from using it.
    agent._try_activate_fallback = Mock(return_value=True)
    monkeypatch.setattr("agent.turn_api_error.settle_delivered_partial", lambda *args: "")
    monkeypatch.setattr("agent.turn_api_error.nonretryable_client_error_result", lambda *args, **kwargs: {"failed": True})
    verdict = settle_unrecovered_error(
        agent, api_error=error, classified=classified, _retry=SimpleNamespace(),
        status_code=status, error_msg=str(error), is_context_length_error=False, is_rate_limited=False,
        _is_zai_coding_overload=False, _provider=agent.provider, _base=agent.base_url, _model="test",
        messages=[], api_messages=[], api_kwargs={}, active_system_prompt="", conversation_history=[],
        approx_tokens=0, retry_count=1, max_retries=3, compression_attempts=0, api_call_count=1,
    )
    assert verdict.action == "return" and verdict.result["failed"]
    agent._try_activate_fallback.assert_not_called()


@pytest.mark.parametrize("code", [
    "subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable", None,
])
def test_chatgpt_unavailable_uses_existing_bounded_server_retry(code, monkeypatch):
    from agent.chatgpt_responses import classify_chatgpt_error
    from agent.error_classifier import FailoverReason, classify_api_error
    from providers import get_provider_profile
    from run_agent import _StreamErrorEvent

    monkeypatch.setattr(get_provider_profile("openai-chatgpt"), "classify_api_error", classify_chatgpt_error)
    error = _StreamErrorEvent("Temporarily unavailable", code=code, status_code=503)
    classified = classify_api_error(error, provider="openai-chatgpt")
    assert classified.reason == FailoverReason.server_error and classified.retryable
    assert not classified.should_rotate_credential and not classified.should_fallback
    # The same public host with API-key auth must retain its own classifier.
    assert classify_api_error(error, provider="openai").reason != FailoverReason.provider_policy_blocked


@pytest.mark.parametrize("url", ["https://other.example/v1", "https://api.openai.com/v1/other", "http://api.openai.com/v1"])
def test_chatgpt_main_refuses_a_different_endpoint_before_dispatch(url):
    from unittest.mock import Mock

    from agent.codex_runtime import run_codex_stream

    client = Mock(base_url=url)
    with pytest.raises(ValueError, match="https://api.openai.com/v1"):
        run_codex_stream(SimpleNamespace(provider="openai-chatgpt"), {"model": "test", "input": []}, client=client)
    client.responses.create.assert_not_called()


def test_chatgpt_usage_guidance_reaches_chat_without_changing_provider_evidence():
    from agent.error_classifier import classify_api_error
    from agent.turn_failure_copy import nonretryable_copy
    from run_agent import _StreamErrorEvent

    error = _StreamErrorEvent("App allowance exhausted", code="subscription_sharing_usage_limit_exceeded",
                             param="usage", status_code=429)
    original = deepcopy(error.body)
    classified = classify_api_error(error, provider="openai-chatgpt")
    copy = nonretryable_copy(classified, provider="openai-chatgpt", model="test", summary=str(error))
    assert "https://chatgpt.com/settings/usage" in copy
    assert "app-specific" in copy and "App allowance exhausted" in copy
    assert error.body == original and error.status_code == 429 and error.param == "usage"
