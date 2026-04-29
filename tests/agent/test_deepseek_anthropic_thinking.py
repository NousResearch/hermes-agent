"""Regression guard: preserve unsigned thinking blocks for DeepSeek /anthropic.

DeepSeek V4 thinking models served via ``api.deepseek.com/anthropic`` speak
the Anthropic Messages protocol but require model-native ``thinking`` content
blocks to round-trip on every replayed assistant tool-call message. The
generic third-party strip path in ``convert_messages_to_anthropic`` drops
all thinking blocks, so the second turn fails with HTTP 400::

    The ``content[].thinking`` in the thinking mode must be passed back
    to the API.

The fix mirrors the strategy that already handles Kimi's ``/coding`` endpoint:
strip Anthropic-signed blocks (the third party can't validate them) but keep
the unsigned ones synthesised from ``reasoning_content`` (#16748).
"""
from __future__ import annotations

import pytest


def _build_assistant_with_thinking(thinking_text: str = "let me think...") -> dict:
    """Return an assistant message that carries one unsigned thinking block."""
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": thinking_text},
            {"type": "text", "text": "Tool result analysed."},
            {
                "type": "tool_use",
                "id": "toolu_01ABC",
                "name": "lookup",
                "input": {"q": "x"},
            },
        ],
    }


def _build_signed_assistant_thinking() -> dict:
    """Anthropic-signed thinking — must be stripped on third parties."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "signed reasoning",
                "signature": "sig-from-anthropic",
            },
            {"type": "text", "text": "Done."},
        ],
    }


class TestDeepSeekAnthropicEndpointDetector:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://api.deepseek.com/anthropic", True),
            ("https://api.deepseek.com/anthropic/", True),
            ("https://api.deepseek.com/anthropic/v1/messages", True),
            ("https://api.deepseek.com/v1", False),
            ("https://api.deepseek.com", False),
            ("https://api.minimax.io/anthropic", False),
            (None, False),
            ("", False),
        ],
    )
    def test_endpoint_match(self, url, expected):
        from agent.anthropic_adapter import _is_deepseek_anthropic_endpoint
        assert _is_deepseek_anthropic_endpoint(url) is expected


class TestDeepSeekPreservesUnsignedThinking:
    """convert_messages_to_anthropic must keep unsigned thinking on history
    when speaking to DeepSeek /anthropic — otherwise the next replay fails
    with the documented 400 about missing ``content[].thinking``.
    """

    def _convert(self, messages, *, base_url):
        from agent.anthropic_adapter import convert_messages_to_anthropic
        _, converted = convert_messages_to_anthropic(messages, base_url=base_url)
        return converted

    def test_unsigned_thinking_kept_on_deepseek(self):
        msgs = [
            {"role": "user", "content": "hi"},
            _build_assistant_with_thinking("step-by-step plan"),
            {"role": "user", "content": "ok continue"},
            _build_assistant_with_thinking("part 2"),
        ]

        converted = self._convert(
            msgs, base_url="https://api.deepseek.com/anthropic"
        )

        assistants = [m for m in converted if m.get("role") == "assistant"]
        assert len(assistants) == 2
        for a in assistants:
            blocks = a.get("content", [])
            thinking = [b for b in blocks if isinstance(b, dict) and b.get("type") == "thinking"]
            assert len(thinking) == 1, (
                "DeepSeek /anthropic must keep unsigned thinking on every "
                f"assistant message; got blocks={blocks}"
            )

    def test_signed_thinking_stripped_on_deepseek(self):
        """Anthropic-signed thinking blocks must be removed — DeepSeek can't
        validate Anthropic's signatures, so leaving them would fail."""
        msgs = [
            {"role": "user", "content": "hi"},
            _build_signed_assistant_thinking(),
        ]

        converted = self._convert(
            msgs, base_url="https://api.deepseek.com/anthropic"
        )

        assistant = next(m for m in converted if m.get("role") == "assistant")
        thinking = [
            b for b in assistant["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking == [], (
            "Anthropic-signed thinking must be stripped on DeepSeek /anthropic"
        )

    def test_other_third_party_endpoint_still_strips_unsigned(self):
        """Endpoints we haven't whitelisted must keep behaving as before
        (strip thinking on every assistant message). Locks in that the
        DeepSeek branch is opt-in."""
        msgs = [
            {"role": "user", "content": "hi"},
            _build_assistant_with_thinking("plan A"),
            {"role": "user", "content": "next"},
            _build_assistant_with_thinking("plan B"),
        ]

        converted = self._convert(
            msgs, base_url="https://api.minimax.io/anthropic"
        )

        for a in [m for m in converted if m.get("role") == "assistant"]:
            thinking = [
                b for b in a["content"]
                if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            assert thinking == [], (
                "Generic third-party endpoint must continue stripping all "
                "thinking blocks — this gating is DeepSeek-specific."
            )
