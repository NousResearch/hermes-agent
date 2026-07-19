"""Regression guard: auto-detect third-party endpoints that require reasoning echo-back.

Any third-party endpoint speaking the Anthropic Messages protocol can enable
thinking/reasoning server-side.  When it does, it returns ``reasoning_content``
in responses and requires the synthesised thinking blocks to round-trip on
subsequent requests — the same contract as Kimi and DeepSeek's ``/anthropic``.

Without detection, the generic third-party path strips ALL thinking blocks,
causing the replay to fail with HTTP 400::

    The reasoning_content in the thinking mode must be passed back to the API.

The fix: if any assistant message in the conversation carries ``reasoning_content``,
the endpoint has been using thinking mode — preserve unsigned thinking blocks
synthesised from it, exactly as we do for Kimi and DeepSeek.

Reproduces the failure seen with ``mimo-v2.5-pro`` and other proxies that
front third-party models through an Anthropic-Messages-compatible gateway.
"""

from __future__ import annotations

import pytest


GENERIC_THIRD_PARTY_URLS = [
    "https://api.mimo.ai/v1",
    "https://gateway.example.com/anthropic/v1",
    "https://proxy.internal/claude",
    "https://some-provider.com/openai/v1",
]


class TestThirdPartyReasoningAutoDetect:
    """convert_messages_to_anthropic must auto-detect reasoning echo-back requirement."""

    @pytest.mark.parametrize("base_url", GENERIC_THIRD_PARTY_URLS)
    def test_unsigned_thinking_preserved_when_reasoning_content_present(
        self, base_url: str
    ) -> None:
        """Thinking blocks must survive replay when any message has reasoning_content.

        Before the fix, third-party (non-Kimi, non-DeepSeek) endpoints had all
        thinking blocks stripped.  The endpoint then saw an assistant message
        with no reasoning and returned HTTP 400.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "thinking through the problem",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "do_thing", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url=base_url
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1, (
            f"Third-party endpoint ({base_url}) with reasoning_content must "
            "preserve unsigned thinking blocks — endpoint rejects replayed "
            "messages without them."
        )
        assert thinking_blocks[0]["thinking"] == "thinking through the problem"
        assert "signature" not in thinking_blocks[0]

    @pytest.mark.parametrize("base_url", GENERIC_THIRD_PARTY_URLS)
    def test_no_thinking_blocks_without_reasoning_content(
        self, base_url: str
    ) -> None:
        """Third-party endpoints with no reasoning_content must not get fake blocks.

        The auto-detect only activates when reasoning_content evidence is
        present.  Non-thinking sessions must not have thinking blocks injected.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url=base_url
        )

        for m in converted:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                assert not (
                    isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
                ), f"Unexpected thinking block in non-reasoning session ({base_url})"

    @pytest.mark.parametrize("base_url", GENERIC_THIRD_PARTY_URLS)
    def test_signed_anthropic_blocks_still_stripped(self, base_url: str) -> None:
        """Anthropic-signed thinking blocks must still be stripped on third-party endpoints.

        Even when auto-detect is active, signed blocks from prior native
        Anthropic sessions must not be forwarded — third-party endpoints
        cannot validate Anthropic-proprietary signatures.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "r1",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "anthropic-signed reasoning",
                        "signature": "SIG-anthropic-xyz",
                    },
                    {"type": "text", "text": "answer"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url=base_url
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        signed_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict)
            and b.get("type") == "thinking"
            and b.get("signature")
        ]
        assert signed_blocks == [], (
            f"Signed Anthropic thinking blocks must be stripped on third-party "
            f"endpoint ({base_url}) — upstream cannot validate them."
        )

    def test_multi_turn_all_reasoning_content_preserved(self) -> None:
        """All assistant turns with reasoning_content must keep their thinking blocks."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "reasoning_content": "turn 1 reasoning",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "reasoning_content": "turn 2 reasoning",
                "content": "final answer",
            },
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.mimo.ai/v1"
        )

        assistants = [m for m in converted if m["role"] == "assistant"]
        assert len(assistants) == 2
        for assistant, expected in zip(assistants, ("turn 1 reasoning", "turn 2 reasoning")):
            thinking = [
                b for b in assistant["content"]
                if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            assert len(thinking) == 1, (
                f"Expected thinking block with '{expected}' — got {thinking!r}"
            )
            assert thinking[0]["thinking"] == expected
