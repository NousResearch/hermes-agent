"""Regression tests for thinking block signature recovery (issue #24401).

When switching from a provider that emits thinking blocks (e.g. MiniMax via
Anthropic transport) to Anthropic proper, the thinking block signatures from
the previous provider are invalid.  The recovery path must strip thinking
blocks from assistant message *content* arrays, not just the top-level
``reasoning_details`` field — otherwise ``convert_messages_to_anthropic`` on
the retry keeps the signed blocks from the last assistant message and the
API rejects them again with the same HTTP 400.
"""

import pytest


class TestThinkingSignatureRecovery:
    """Verify that the thinking-signature recovery strips content blocks."""

    @staticmethod
    def _simulate_recovery(messages: list) -> None:
        """Replay the recovery logic from run_agent.py (thinking_signature branch)."""
        _THINKING_TYPES = ("thinking", "redacted_thinking")
        for _m in messages:
            if not isinstance(_m, dict):
                continue
            _m.pop("reasoning_details", None)
            if _m.get("role") == "assistant" and isinstance(_m.get("content"), list):
                _m["content"] = [
                    _b for _b in _m["content"]
                    if not (isinstance(_b, dict) and _b.get("type") in _THINKING_TYPES)
                ] or [{"type": "text", "text": "(thinking elided)"}]

    def test_strips_thinking_from_content_array(self):
        """Thinking blocks in assistant content arrays must be removed."""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think...", "signature": "miniMaxSig123"},
                    {"type": "text", "text": "Here is my answer."},
                ],
            },
        ]
        self._simulate_recovery(messages)
        assistant = messages[1]
        assert len(assistant["content"]) == 1
        assert assistant["content"][0]["type"] == "text"
        assert assistant["content"][0]["text"] == "Here is my answer."

    def test_strips_redacted_thinking_from_content(self):
        """Redacted thinking blocks must also be stripped."""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "encrypted_stuff"},
                    {"type": "text", "text": "Response."},
                ],
            },
        ]
        self._simulate_recovery(messages)
        assert len(messages[1]["content"]) == 1
        assert messages[1]["content"][0]["type"] == "text"

    def test_all_thinking_stripped_leaves_placeholder(self):
        """When all content was thinking, a placeholder is inserted."""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Only thinking, no text.", "signature": "sig"},
                ],
            },
        ]
        self._simulate_recovery(messages)
        assert len(messages[1]["content"]) == 1
        assert messages[1]["content"][0]["type"] == "text"
        assert "thinking elided" in messages[1]["content"][0]["text"]

    def test_strips_reasoning_details_field(self):
        """The legacy reasoning_details field is also removed."""
        messages = [
            {"role": "assistant", "content": "text", "reasoning_details": [{"type": "reasoning"}]},
        ]
        self._simulate_recovery(messages)
        assert "reasoning_details" not in messages[0]

    def test_does_not_touch_user_messages(self):
        """User messages are left untouched."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        self._simulate_recovery(messages)
        assert len(messages[0]["content"]) == 1

    def test_multiple_assistant_messages_all_stripped(self):
        """Thinking blocks in ALL assistant messages are stripped, not just the last."""
        messages = [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "old think", "signature": "old_sig"},
                    {"type": "text", "text": "old answer"},
                ],
            },
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "new think", "signature": "new_sig"},
                    {"type": "text", "text": "new answer"},
                ],
            },
        ]
        self._simulate_recovery(messages)
        # Both assistant messages should have thinking stripped
        assert messages[1]["content"] == [{"type": "text", "text": "old answer"}]
        assert messages[3]["content"] == [{"type": "text", "text": "new answer"}]

    def test_string_content_untouched(self):
        """Assistant messages with string content (not list) are not modified."""
        messages = [
            {"role": "assistant", "content": "plain text"},
        ]
        self._simulate_recovery(messages)
        assert messages[0]["content"] == "plain text"
