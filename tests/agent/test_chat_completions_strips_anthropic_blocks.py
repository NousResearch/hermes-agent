"""Regression: anthropic_content_blocks must not leak into chat_completions payloads.

Reproduces HTTP 400:
``Extra inputs are not permitted, field: 'messages[N].anthropic_content_blocks'``

When a session switches from an anthropic_messages-routed model (e.g. MiniMax
on opencode-go) to a chat_completions-routed model (e.g. GLM-5.2, DeepSeek V4 Pro),
assistant messages carry ``anthropic_content_blocks`` — a provider-data field the
``anthropic_messages`` transport writes for thinking-block replay. The
``chat_completions`` transport must strip this field before the message hits the
wire because strict providers reject unknown top-level message keys.

Fix: ``ChatCompletionsTransport.convert_messages()`` now pops
``anthropic_content_blocks`` alongside the other provider-specific fields
(codex_reasoning_items, codex_message_items, tool_name, timestamp, extra_content).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from agent.transports.chat_completions import ChatCompletionsTransport


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


class TestAnthropicContentBlocksStrip:
    """Verify convert_messages strips anthropic_content_blocks."""

    def test_strips_from_assistant_message(self, transport):
        """anthropic_content_blocks should be removed from assistant messages."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me think about this...",
                "anthropic_content_blocks": [
                    {"type": "thinking", "thinking": "planning...", "signature": "s1"},
                    {"type": "text", "text": "Let me think..."},
                ],
            }
        ]
        clean = transport.convert_messages(messages)
        assert "anthropic_content_blocks" not in clean[0], (
            f"anthropic_content_blocks leaked: {clean[0]}"
        )
        assert clean[0]["role"] == "assistant"
        assert clean[0]["content"] == "Let me think about this..."

    def test_strips_from_multiple_messages(self, transport):
        """All messages should be cleaned, not just the first."""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "hi",
                "anthropic_content_blocks": [{"type": "thinking", "thinking": "x", "signature": "s"}],
            },
            {"role": "user", "content": "what?"},
            {
                "role": "assistant",
                "content": "ok",
                "anthropic_content_blocks": [{"type": "text", "text": "ok"}],
            },
        ]
        clean = transport.convert_messages(messages)
        assert len(clean) == 4
        assert "anthropic_content_blocks" not in clean[1]
        assert "anthropic_content_blocks" not in clean[3]
        # user messages should be untouched
        assert clean[0] == {"role": "user", "content": "hello"}

    def test_noop_when_field_absent(self, transport):
        """Messages without the field should pass through unchanged."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        clean = transport.convert_messages(messages)
        assert clean == messages
        # Should return the same list (no deep-copy when no sanitization needed)
        assert clean is messages

    def test_deepcopy_isolates_original(self, transport):
        """Original messages should not be mutated (deep copy)."""
        messages = [
            {
                "role": "assistant",
                "content": "hi",
                "anthropic_content_blocks": [{"type": "text", "text": "ok"}],
            }
        ]
        original_block = messages[0]["anthropic_content_blocks"]
        clean = transport.convert_messages(messages)
        # Original should be untouched
        assert "anthropic_content_blocks" in messages[0], "original was mutated!"
        assert messages[0]["anthropic_content_blocks"] is original_block
        # Copy should be clean
        assert "anthropic_content_blocks" not in clean[0]

    def test_real_world_leak_scenario(self, transport):
        """Simulate a MiniMax → GLM-5.2 session switch leaking anthropic_content_blocks."""
        # Messages 0-144 are earlier history (abbreviated for the test)
        messages = [{"role": m["role"], "content": m["content"]} for m in [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "fix the bug"},
        ]]
        # Message 145 is a MiniMax response (anthropic_messages) with thinking blocks
        messages.append({
            "role": "assistant",
            "content": "I'll check the logs.",
            "anthropic_content_blocks": [
                {"type": "thinking", "thinking": "Let me analyze the error...", "signature": "sig-abc"},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                 "input": {"path": "/tmp/log.txt"}},
            ],
            # Also contains _anthropic_content_blocks (the stashed variant)
            "_anthropic_content_blocks": [
                {"type": "thinking", "thinking": "Let me analyze...", "signature": "sig-abc"},
            ],
        })
        # Message 146 is a tool result
        messages.append({"role": "tool", "content": "error: 404 Not Found"})

        # Now the model switches to GLM-5.2 (chat_completions)
        # convert_messages should strip the Anthropic-specific fields
        clean = transport.convert_messages(messages)

        # The assistant message (index 2, 0-based) should be stripped
        stripped_msg = clean[2]
        assert "anthropic_content_blocks" not in stripped_msg, (
            f"anthropic_content_blocks leaked into chat_completions payload"
        )
        assert "_anthropic_content_blocks" not in stripped_msg, (
            f"_anthropic_content_blocks leaked into chat_completions payload"
        )
        assert stripped_msg.get("content") == "I'll check the logs."
        assert stripped_msg.get("role") == "assistant"

    def test_also_strips_underscore_variant(self, transport):
        """_anthropic_content_blocks (stashed variant) should also be stripped.

        model_metadata.py stashes thinking blocks as _anthropic_content_blocks
        for image counting. This is already handled by the ``_``-prefixed key
        stripping, but verify explicitly.
        """
        messages = [
            {
                "role": "assistant",
                "content": "hi",
                "_anthropic_content_blocks": [{"type": "image", "source": {}}],
            }
        ]
        clean = transport.convert_messages(messages)
        assert "_anthropic_content_blocks" not in clean[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
