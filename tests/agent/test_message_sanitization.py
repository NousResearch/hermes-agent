from agent.message_sanitization import sanitize_chat_completion_messages_for_wire


def test_chat_completion_wire_sanitizer_preserves_clean_identity():
    messages = [{"role": "user", "content": "hello"}]

    assert sanitize_chat_completion_messages_for_wire(messages) is messages


def test_chat_completion_wire_sanitizer_strips_internal_fields_and_controls():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "tool_name": "vault_search",
            "_empty_recovery_synthetic": True,
            "content": [{"type": "text", "text": "\x00json:{\"ok\": true}"}],
        }
    ]

    sanitized = sanitize_chat_completion_messages_for_wire(messages)

    assert sanitized is not messages
    assert sanitized[0]["content"] == "json:{\"ok\": true}"
    assert "tool_name" not in sanitized[0]
    assert "_empty_recovery_synthetic" not in sanitized[0]
    assert "tool_name" in messages[0]
