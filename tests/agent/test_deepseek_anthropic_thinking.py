"""Regression guard: preserve thinking blocks on DeepSeek's /anthropic endpoint.

DeepSeek's ``api.deepseek.com/anthropic`` route speaks the Anthropic Messages
protocol but, when thinking mode is enabled, requires every ``thinking`` /
``redacted_thinking`` block returned in assistant messages to be replayed
unchanged on subsequent turns.  The signatures are DeepSeek's own (not
Anthropic's), so DeepSeek validates them and rejects the request with HTTP
400 if they are missing::

    The content[].thinking in the thinking mode must be passed back to the
    API.

The generic third-party path strips all thinking blocks (because Anthropic
signatures cannot be validated by other proxies); DeepSeek needs to opt out
of that stripping.  See :func:`agent.anthropic_adapter._is_deepseek_anthropic_endpoint`.
"""

from __future__ import annotations

import pytest


class TestDeepSeekAnthropicPreservesThinking:
    """convert_messages_to_anthropic must replay DeepSeek thinking blocks."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.deepseek.com/anthropic",
            "https://api.deepseek.com/anthropic/",
            "https://api.deepseek.com/anthropic/v1",
            "https://API.DeepSeek.com/anthropic",
        ],
    )
    def test_thinking_block_with_signature_is_preserved(self, base_url: str) -> None:
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello!",
                "reasoning_details": [
                    {
                        "type": "thinking",
                        "thinking": "the user said hi",
                        "signature": "deepseek-sig-abc",
                    }
                ],
            },
            {"role": "user", "content": "again"},
        ]

        _system, anthropic_messages = convert_messages_to_anthropic(
            messages, base_url=base_url
        )

        assistant = next(m for m in anthropic_messages if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "the user said hi"
        assert thinking_blocks[0]["signature"] == "deepseek-sig-abc"

    def test_thinking_preserved_on_non_latest_assistant_too(self) -> None:
        """DeepSeek validates thinking on every prior assistant turn, not just the last."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": "a1",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "t1", "signature": "sig1"},
                ],
            },
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": "a2",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "t2", "signature": "sig2"},
                ],
            },
            {"role": "user", "content": "q3"},
        ]

        _system, anthropic_messages = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )

        assistants = [m for m in anthropic_messages if m["role"] == "assistant"]
        assert len(assistants) == 2
        for assistant, expected_sig in zip(assistants, ("sig1", "sig2")):
            blocks = [
                b for b in assistant["content"]
                if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            assert len(blocks) == 1
            assert blocks[0]["signature"] == expected_sig

    def test_redacted_thinking_with_data_is_preserved(self) -> None:
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "ok",
                "reasoning_details": [
                    {"type": "redacted_thinking", "data": "encrypted-blob"},
                ],
            },
            {"role": "user", "content": "again"},
        ]

        _system, anthropic_messages = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )

        assistant = next(m for m in anthropic_messages if m["role"] == "assistant")
        redacted = [
            b for b in assistant["content"]
            if isinstance(b, dict) and b.get("type") == "redacted_thinking"
        ]
        assert len(redacted) == 1
        assert redacted[0]["data"] == "encrypted-blob"

    def test_cache_control_stripped_from_thinking_block(self) -> None:
        """cache_control must still be stripped from thinking blocks even on DeepSeek."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "ok",
                "reasoning_details": [
                    {
                        "type": "thinking",
                        "thinking": "t",
                        "signature": "sig",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {"role": "user", "content": "again"},
        ]

        _system, anthropic_messages = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )

        assistant = next(m for m in anthropic_messages if m["role"] == "assistant")
        thinking = next(
            b for b in assistant["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        )
        assert "cache_control" not in thinking
        assert thinking["signature"] == "sig"


class TestDeepSeekDoesNotAffectOtherEndpoints:
    """The DeepSeek special case must not change behavior on other endpoints."""

    def test_minimax_third_party_still_strips_signed_thinking(self) -> None:
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello!",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "x", "signature": "anthropic-sig"},
                ],
            },
            {"role": "user", "content": "again"},
        ]

        _system, anthropic_messages = convert_messages_to_anthropic(
            messages, base_url="https://api.minimax.io/anthropic"
        )

        assistant = next(m for m in anthropic_messages if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == []

    def test_native_anthropic_unaffected(self) -> None:
        """Direct Anthropic still keeps the signed block on the latest assistant only."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": "a1",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "t1", "signature": "sig1"},
                ],
            },
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": "a2",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "t2", "signature": "sig2"},
                ],
            },
            {"role": "user", "content": "q3"},
        ]

        _system, anthropic_messages = convert_messages_to_anthropic(
            messages, base_url=None
        )

        assistants = [m for m in anthropic_messages if m["role"] == "assistant"]
        # First assistant: thinking stripped (not latest)
        first_thinking = [
            b for b in assistants[0]["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert first_thinking == []
        # Latest assistant: signed thinking kept
        last_thinking = [
            b for b in assistants[1]["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(last_thinking) == 1
        assert last_thinking[0]["signature"] == "sig2"


class TestEndpointDetection:
    """Boundary tests for _is_deepseek_anthropic_endpoint."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://api.deepseek.com/anthropic", True),
            ("https://api.deepseek.com/anthropic/", True),
            ("https://api.deepseek.com/anthropic/v1/messages", True),
            ("HTTPS://API.DEEPSEEK.COM/anthropic", True),
            ("https://api.deepseek.com/v1", False),  # OpenAI-compat route
            ("https://api.deepseek.com", False),
            ("https://api.minimax.io/anthropic", False),
            ("https://api.kimi.com/coding", False),
            (None, False),
            ("", False),
        ],
    )
    def test_endpoint_detection(self, url, expected: bool) -> None:
        from agent.anthropic_adapter import _is_deepseek_anthropic_endpoint

        assert _is_deepseek_anthropic_endpoint(url) is expected
