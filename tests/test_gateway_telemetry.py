from __future__ import annotations

import json
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import (
    _build_gateway_action_telemetry_payload,
    _gateway_action_status,
    _result_response_chars,
)
from gateway.session import SessionSource
from hermes_telemetry import safe_emit_event


def test_gateway_action_payload_hashes_identifiers_and_omits_text():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-123",
        chat_type="dm",
        user_id="user-456",
        user_name="Alice Example",
        thread_id="thread-789",
    )
    event = MessageEvent(
        text="/deploy production with secret details",
        message_type=MessageType.TEXT,
        source=source,
        media_urls=["/tmp/private-image.png"],
        media_types=["image/png"],
    )
    result = {
        "final_response": "Deployment completed with sensitive prose",
        "api_calls": 2,
        "tools": ["terminal", "read_file"],
        "model": "provider/private-model-name",
        "input_tokens": 100,
        "output_tokens": 20,
    }

    payload = _build_gateway_action_telemetry_payload(
        source=source,
        event=event,
        session_key="telegram:chat-123:user-456",
        result=result,
        duration_ms=42,
    )

    serialized = json.dumps(payload, ensure_ascii=False)
    assert payload["platform"] == "telegram"
    assert payload["chat_type"] == "dm"
    assert payload["message_type"] == "text"
    assert payload["is_command"] is True
    assert payload["media_count"] == 1
    assert payload["response_chars"] == len(result["final_response"])
    assert payload["api_calls"] == 2
    assert payload["tools_count"] == 2
    assert payload["duration_ms"] == 42

    # No raw text, names, IDs, paths, command names, or model names in telemetry.
    assert "deploy" not in serialized
    assert "secret details" not in serialized
    assert "Deployment completed" not in serialized
    assert "Alice" not in serialized
    assert "chat-123" not in serialized
    assert "user-456" not in serialized
    assert "thread-789" not in serialized
    assert "private-image" not in serialized
    assert "private-model-name" not in serialized

    assert payload["chat_id_hash"]
    assert payload["user_id_hash"]
    assert payload["thread_id_hash"]
    assert payload["session_key_hash"]
    assert payload["command_hash"]
    assert payload["model_hash"]


def test_gateway_action_status_classification():
    assert _gateway_action_status("plain response") == "ok"
    assert _gateway_action_status({"final_response": "ok"}) == "ok"
    assert _gateway_action_status({"interrupted": True}) == "interrupted"
    assert _gateway_action_status({"partial": True}) == "partial"
    assert _gateway_action_status({"failed": True}) == "error"
    assert _gateway_action_status({"error": "raw provider failure"}) == "error"
    assert _gateway_action_status(None, RuntimeError("boom")) == "error"


def test_result_response_chars_never_returns_content():
    assert _result_response_chars("hello") == 5
    assert _result_response_chars({"final_response": "hello world"}) == 11
    assert _result_response_chars(SimpleNamespace(final_response="ignored")) == 0


def test_safe_emit_gateway_action_redacts_sensitive_payload_keys(tmp_path):
    path = safe_emit_event(
        "gateway_action",
        {
            "platform": "telegram",
            "prompt_text": "do not store me",
            "completion_text": "do not store me either",
            "user_id_hash": "abc123",
        },
        source="gateway",
        hermes_home=tmp_path,
    )

    assert path is not None
    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert event["event_type"] == "gateway_action"
    assert event["source"] == "gateway"
    assert event["payload"]["prompt_text"] == "[REDACTED]"
    assert event["payload"]["completion_text"] == "[REDACTED]"
    assert event["payload"]["user_id_hash"] == "abc123"
