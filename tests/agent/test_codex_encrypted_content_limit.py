"""Synthetic field-cap failures must never erase the only retained context."""

import hashlib
import json
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.codex_responses_adapter import (
    _normalize_codex_response,
    _preflight_codex_api_kwargs,
    _preflight_codex_input_items,
)
from agent.error_classifier import classify_api_error
from agent.encrypted_content import has_oversized_latest_checkpoint
from agent.transports.codex import ResponsesApiTransport
from agent.turn_api_error import handle_api_error
from agent.turn_retry_state import TurnRetryState


# Provider's encrypted_content character limit, independent of token budgets.
CAP = 20_971_520


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _history(blob):
    return [
        {"role": "user", "content": "Preserve the synthetic task."},
        {"role": "assistant", "content": "Synthetic plaintext handoff.",
         "_compressed_summary": True},
        {"role": "assistant", "content": "Checking synthetic evidence.",
         "codex_reasoning_items": [{"type": "compaction", "encrypted_content": blob}],
         "tool_calls": [{"id": "call_synthetic", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_synthetic", "content": "Synthetic evidence."},
        {"role": "user", "content": "Continue the synthetic task."},
    ]


def _handle(error, messages, api_kwargs):
    agent = MagicMock()
    agent.provider = "openai-codex"
    agent.model = "gpt-5.4"
    agent.api_mode = "codex_responses"
    agent.base_url = "https://example.test/v1"
    agent.log_prefix = "synthetic: "
    agent.context_compressor = None
    agent.thinking_callback = None
    agent._recover_with_credential_pool.return_value = (False, False)
    agent._disable_codex_reasoning_replay.return_value = {"items": 1, "messages": 1}
    retry = TurnRetryState()
    verdict = handle_api_error(
        agent, api_error=error, _retry=retry, thinking_spinner=None,
        messages=messages, api_messages=messages, api_kwargs=api_kwargs,
        system_message="Synthetic stable system.", active_system_prompt="Synthetic stable system.",
        conversation_history=messages, approx_tokens=100, retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=2, api_call_count=1,
        api_request_id="synthetic", api_start_time=time.time(),
        effective_task_id="synthetic", turn_id="synthetic",
    )
    return agent, retry, verdict


def test_capture_keeps_response_semantics_and_warns_that_oversized_checkpoint_cannot_replay(caplog):
    blob = "x" * 23_849_740
    response = SimpleNamespace(status="completed", output=[
        SimpleNamespace(type="compaction", encrypted_content=blob),
        SimpleNamespace(type="message", role="assistant", status="completed",
                        content=[SimpleNamespace(type="output_text", text="Synthetic reply.")]),
        SimpleNamespace(type="function_call", status="completed", id="fc_synthetic",
                        call_id="call_synthetic", name="terminal", arguments="{}"),
    ])
    message, finish = _normalize_codex_response(response, issuer_kind="codex_backend")
    assert message.content == "Synthetic reply."
    assert finish == "tool_calls"
    assert message.tool_calls[0].function.name == "terminal"
    assert message.codex_reasoning_items[0]["encrypted_content"] is blob
    assert "exceeds" in caplog.text and "replay" in caplog.text
    assert blob[:100] not in caplog.text


def test_persisted_oversized_checkpoint_replay_fails_without_changing_context():
    # Exercise the production persistence row and serializer without opening a database.
    from agent.session_persistence import _db_flush_row
    from hermes_state_messages import SessionMessagesMixin

    messages = _history("x" * (CAP + 1))
    row = _db_flush_row(SimpleNamespace(), messages[2], False)
    sidecar = row["codex_reasoning_items"]
    messages[2]["codex_reasoning_items"] = json.loads(SessionMessagesMixin._reasoning_json_text(sidecar))
    before = _fingerprint(messages)
    kwargs = ResponsesApiTransport().build_kwargs(
        "gpt-5.4", messages, instructions="Synthetic stable system.",
        is_codex_backend=True, context_management=[{"type": "compaction", "compact_threshold": 1000}],
    )
    assert kwargs["input"][0]["type"] == "compaction"
    with pytest.raises(ValueError, match="encrypted_content.*exceeds") as caught:
        _preflight_codex_api_kwargs(kwargs)
    agent, retry, verdict = _handle(caught.value, messages, kwargs)
    assert verdict.action == "return"
    assert verdict.result["failure_retryable"] is False
    assert verdict.result["messages"] is messages
    assert _fingerprint(messages) == before
    agent._disable_codex_reasoning_replay.assert_not_called()
    agent._dump_api_request_debug.assert_not_called()
    agent._persist_session.assert_not_called()
    assert not retry.invalid_encrypted_content_retry_attempted


def test_structured_remote_field_cap_is_terminal_before_encrypted_recovery():
    error = RuntimeError("Synthetic field validation failed")
    error.status_code = 400
    error.body = {"error": {
        "code": "string_above_max_length", "param": "input[0].encrypted_content",
        # Prose must not override the structured field/code and activate strip-all.
        "message": "invalid_encrypted_content exceeds maximum context length",
    }}
    messages = _history("synthetic-opaque-context")
    before = _fingerprint(messages)
    agent, retry, verdict = _handle(error, messages, {"input": deepcopy(messages)})
    assert verdict.action == "return"
    assert verdict.result["failure_reason"] == "encrypted_content_too_large"
    assert verdict.result["failure_retryable"] is False
    assert verdict.result["messages"] is messages
    assert _fingerprint(messages) == before
    agent._disable_codex_reasoning_replay.assert_not_called()
    agent._recover_with_credential_pool.assert_not_called()
    agent._dump_api_request_debug.assert_not_called()
    agent._persist_session.assert_not_called()
    assert not retry.invalid_encrypted_content_retry_attempted
    classified = classify_api_error(error, provider="openai-codex")
    assert classified.reason.value == "encrypted_content_too_large"
    assert not any((classified.retryable, classified.should_compress,
                    classified.should_fallback, classified.should_rotate_credential))


@pytest.mark.parametrize("item_type", ["compaction", "reasoning"])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_encrypted_field_character_boundary_preserves_opaque_bytes(item_type, offset):
    blob = "x" * (CAP + offset)
    item = {"type": item_type, "encrypted_content": blob, "summary": []}
    if offset <= 0:
        wire = _preflight_codex_input_items([item])
        assert wire[0]["encrypted_content"] is blob
    else:
        with pytest.raises(ValueError) as caught:
            _preflight_codex_input_items([item])
        # Wrapped local guards must remain terminal too.
        wrapped = RuntimeError("Synthetic wrapper")
        wrapped.__cause__ = caught.value
        assert classify_api_error(wrapped).reason.value == "encrypted_content_too_large"
        assert blob[:100] not in str(caught.value)
    assert item["encrypted_content"] is blob


@pytest.mark.parametrize("status,code,param,expected", [
    (400, "string_above_max_length", "input[12].encrypted_content", True),
    (400, "string_above_max_length", "input[0].content", False),
    (400, "string_above_max_length", "encrypted_content", False),
    (400, "string_above_max_length", "input[0].encrypted_content.extra", False),
    (400, "invalid_encrypted_content", "input[0].encrypted_content", False),
    (500, "string_above_max_length", "input[0].encrypted_content", False),
])
def test_remote_field_cap_classification_requires_exact_structured_evidence(status, code, param, expected):
    error = RuntimeError("Synthetic validation error")
    error.status_code = status
    # The SDK also exposes an unwrapped body, without the outer error key.
    error.body = {"code": code, "param": param, "message": "Synthetic validation error"}
    assert (classify_api_error(error).reason.value == "encrypted_content_too_large") is expected


def test_superseded_oversized_checkpoint_does_not_block_valid_newest_context():
    messages = _history("x" * (CAP + 1))
    messages.append({"role": "assistant", "content": "Synthetic checkpoint renewal.",
                     "codex_reasoning_items": [{"type": "compaction", "encrypted_content": "new-context"}]})
    messages.append({"role": "user", "content": "Synthetic next request."})
    before = _fingerprint(messages)
    agent = SimpleNamespace(api_mode="codex_responses", model="gpt-5.6",
                            provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex",
                            codex_responses_native_compaction=True)
    assert not has_oversized_latest_checkpoint(agent, messages)
    kwargs = ResponsesApiTransport().build_kwargs(
        "gpt-5.4", messages, instructions="Synthetic stable system.",
        is_codex_backend=True, context_management=[{"type": "compaction", "compact_threshold": 1000}],
    )
    wire = _preflight_codex_api_kwargs(kwargs)["input"]
    assert wire[0] == {"type": "compaction", "encrypted_content": "new-context"}
    assert _fingerprint(messages) == before
