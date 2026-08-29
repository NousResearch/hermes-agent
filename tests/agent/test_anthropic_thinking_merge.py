"""Tests for thinking signature invalidation when merging consecutive assistant messages."""

from agent.anthropic_adapter import convert_messages_to_anthropic

SIG = "valid_cryptographic_signature_abc123"


def test_consecutive_assistant_merge_invalidates_thinking_signature():
    """When an assistant message carrying signed thinking is merged with another assistant message,
    its signature is invalidated and demoted to text to prevent HTTP 400."""
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "first thought", "signature": SIG},
                {"type": "text", "text": "part 1"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "part 2"},
            ],
        },
    ]

    _sys, out = convert_messages_to_anthropic(messages)
    assert len(out) == 2  # user + merged assistant
    assistant = out[1]
    assert assistant["role"] == "assistant"

    # Signature must be dead / demoted to text block
    types = [b.get("type") for b in assistant["content"] if isinstance(b, dict)]
    assert "thinking" not in types
    texts = [b.get("text") for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "text"]
    assert "first thought" in texts
    assert "part 1" in texts
    assert "part 2" in texts


def test_single_assistant_message_preserves_thinking_signature():
    """A standalone assistant message retains its valid thinking signature."""
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "first thought", "signature": SIG},
                {"type": "text", "text": "part 1"},
            ],
        },
    ]

    _sys, out = convert_messages_to_anthropic(messages)
    assert len(out) == 2
    assistant = out[1]
    types = [b.get("type") for b in assistant["content"] if isinstance(b, dict)]
    assert "thinking" in types
    thinking_block = [b for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "thinking"][0]
    assert thinking_block["signature"] == SIG
