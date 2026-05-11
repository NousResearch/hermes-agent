"""Test that _sanitize_api_messages strips Hermes-internal metadata keys
before sending to the API. Strict providers like Fireworks reject unknown
fields (HTTP 400), so keys like _empty_recovery_synthetic must be removed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from run_agent import AIAgent


def test_sanitize_strips_internal_metadata_keys():
    """Internal keys prefixed with _ must be stripped from all messages."""
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "_empty_recovery_synthetic": True},
        {"role": "user", "content": "continue", "_empty_terminal_sentinel": True},
        {"role": "assistant", "content": "done"},
    ]
    result = AIAgent._sanitize_api_messages(messages)

    # All messages should have valid roles only
    assert all(m["role"] in ("user", "assistant", "system", "tool") for m in result)

    # No internal metadata keys should survive
    for msg in result:
        for key in msg:
            assert not key.startswith("_"), f"Internal key {key!r} leaked into API message"

    # Content preserved
    assert result[0]["content"] == "hello"
    assert result[1]["content"] == ""
    assert result[2]["content"] == "continue"
    assert result[3]["content"] == "done"


def test_sanitize_preserves_standard_keys():
    """Standard OpenAI message keys must be preserved."""
    messages = [
        {
            "role": "assistant",
            "content": "result",
            "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "tool output"},
    ]
    result = AIAgent._sanitize_api_messages(messages)
    assert len(result) == 2
    assert result[0]["tool_calls"][0]["id"] == "call_123"
    assert result[1]["tool_call_id"] == "call_123"


def test_sanitize_strips_multiple_internal_keys_on_same_message():
    """A single message with multiple internal keys should have all stripped."""
    messages = [
        {
            "role": "assistant",
            "content": "ok",
            "_empty_recovery_synthetic": True,
            "_empty_terminal_sentinel": True,
            "_custom_flag": 42,
        },
    ]
    result = AIAgent._sanitize_api_messages(messages)
    assert len(result) == 1
    assert set(result[0].keys()) == {"role", "content"}
    assert result[0]["content"] == "ok"
