"""Regression guard: preserve thinking content on MiniMax's /anthropic endpoint.

MiniMax's ``api.minimax.io/anthropic`` route returns **signed** thinking blocks.
The generic third-party path in ``_manage_thinking_signatures`` strips ALL
thinking blocks because it assumes third-party endpoints cannot validate
Anthropic-proprietary signatures.  But MiniMax **accepts unsigned thinking
blocks on replay** and uses them for interleaved reasoning across tool-call
turns — stripping the blocks entirely kills chain-of-thought on turn 2+.

The fix adds a MiniMax-specific branch that strips the ``signature`` (converting
the block from signed → unsigned) while preserving the ``thinking`` content.
This is distinct from DeepSeek's policy (which strips signed blocks entirely)
because MiniMax round-trips unsigned thinking content, whereas DeepSeek only
replays unsigned blocks that Hermes synthesises from ``reasoning_content``.

See hermes-agent#75725.
"""

from __future__ import annotations

import pytest


class TestMiniMaxAnthropicPreservesThinkingContent:
    """convert_messages_to_anthropic must preserve thinking content on MiniMax."""

    # ── Layer 1: signed thinking blocks are converted to unsigned ───────

    def test_signed_thinking_block_signature_stripped_on_minimax(self) -> None:
        """Signed thinking blocks must have their signature removed on MiniMax.

        MiniMax validates signatures against the originating turn context, so
        replaying a signed block from an earlier turn is rejected with HTTP 400.
        The fix strips the signature while keeping the thinking content.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me analyze this step by step.",
                        "signature": "minimax-sig-abc123",
                    },
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.minimax.io/anthropic"
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1, (
            "Thinking block content must be preserved (not stripped) on MiniMax"
        )
        assert "signature" not in thinking_blocks[0], (
            "Signature must be stripped from signed thinking blocks on MiniMax — "
            "MiniMax cannot validate Anthropic-proprietary signatures from prior turns."
        )
        assert thinking_blocks[0]["thinking"] == "Let me analyze this step by step."

    def test_signed_thinking_block_with_tool_use_preserved(self) -> None:
        """Signed thinking + tool_use must survive as unsigned thinking + tool_use.

        This is the interleaved-reasoning round-trip: turn 1's assistant message
        carries [thinking(signed), tool_use].  On turn 2 the wire request must
        still carry [thinking(unsigned), tool_use] — not [tool_use] alone.

        Uses the ``anthropic_content_blocks`` fast path (set by
        ``normalize_response`` for turns that interleave signed thinking with
        tool_use) — this is the production scenario described in #75725.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "anthropic_content_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "I need to call the weather tool.",
                        "signature": "sig-xyz",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ],
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "25C, sunny",
            },
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.minimax.io/anthropic"
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        block_types = [
            b.get("type") if isinstance(b, dict) else None
            for b in assistant_msg["content"]
        ]
        # The thinking block must still be present (unsigned)
        assert "thinking" in block_types, (
            "Thinking block must survive on MiniMax to maintain interleaved reasoning"
        )
        # The tool_use block must also be present
        assert "tool_use" in block_types
        # Thinking must come before tool_use (order preserved)
        assert block_types.index("thinking") < block_types.index("tool_use")
        # The surviving thinking block must be unsigned
        for b in assistant_msg["content"]:
            if isinstance(b, dict) and b.get("type") == "thinking":
                assert "signature" not in b

    # ── Layer 2: unsigned thinking blocks preserved as-is ───────────────

    def test_unsigned_thinking_block_preserved_on_minimax(self) -> None:
        """Unsigned thinking blocks (no signature key) must be preserved unchanged."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Reasoning without a signature.",
                    },
                    {"type": "text", "text": "response"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.minimax.io/anthropic"
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Reasoning without a signature."

    # ── Layer 3: other endpoints unaffected ─────────────────────────────

    def test_deepseek_still_strips_signed_thinking(self) -> None:
        """DeepSeek's policy (strip signed entirely) must be unchanged."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "signed content",
                        "signature": "anthropic-sig",
                    },
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == [], (
            "DeepSeek must still strip signed thinking blocks entirely."
        )

    def test_generic_third_party_still_strips_all_thinking(self) -> None:
        """Other third-party endpoints must still strip ALL thinking blocks."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "should be stripped",
                        "signature": "some-sig",
                    },
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://some-proxy.example.com/anthropic"
        )
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == [], (
            "Generic third-party endpoints must still strip all thinking blocks."
        )

    # ── Layer 4: edge cases ──────────────────────────────────────────────

    def test_cache_control_stripped_from_preserved_thinking(self) -> None:
        """cache_control must be stripped from thinking blocks on MiniMax."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning",
                        "signature": "sig",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.minimax.io/anthropic"
        )
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        for b in assistant_msg["content"]:
            if isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}:
                assert "cache_control" not in b

    def test_redacted_thinking_converted_to_text_on_minimax(self) -> None:
        """redacted_thinking blocks (data payload) are converted to text on MiniMax.

        redacted_thinking uses ``data`` as its signature-equivalent opaque payload.
        MiniMax cannot validate it, so the block is replaced with a text
        placeholder to preserve block position for interleaved reasoning.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "redacted_thinking",
                        "data": "opaque-base64-payload",
                    },
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.minimax.io/anthropic"
        )
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        # redacted_thinking must not survive with its data payload
        for b in assistant_msg["content"]:
            if isinstance(b, dict):
                assert b.get("type") != "redacted_thinking", (
                    "redacted_thinking with data payload must not survive on MiniMax"
                )

    def test_multiple_signed_thinking_blocks_all_demoted(self) -> None:
        """Multiple signed thinking blocks must all be demoted to unsigned."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "complex task"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "First thought.",
                        "signature": "sig1",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search",
                        "input": {"q": "test"},
                    },
                    {
                        "type": "thinking",
                        "thinking": "Second thought after tool.",
                        "signature": "sig2",
                    },
                    {"type": "text", "text": "Done."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "result",
                    }
                ],
            },
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.minimax.io/anthropic"
        )
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 2, (
            f"Expected 2 thinking blocks preserved, got {len(thinking_blocks)}"
        )
        for tb in thinking_blocks:
            assert "signature" not in tb
        assert thinking_blocks[0]["thinking"] == "First thought."
        assert thinking_blocks[1]["thinking"] == "Second thought after tool."

    # ── Layer 5: both MiniMax endpoint domains ──────────────────────────

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.minimax.io/anthropic",
            "https://api.minimaxi.com/anthropic",
        ],
    )
    def test_both_minimax_endpoints_preserve_signed_thinking(
        self, base_url: str
    ) -> None:
        """Both api.minimax.io and api.minimaxi.com must preserve thinking content.

        ``_is_minimax_anthropic_endpoint`` matches two domains — both must get
        the MiniMax-specific branch (signed→unsigned) rather than falling through
        to the generic third-party path which strips ALL thinking blocks.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me think about this.",
                        "signature": "minimax-sig-xyz",
                    },
                    {"type": "text", "text": "here is the answer"},
                ],
            },
            {"role": "user", "content": "continue"},
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
            f"MiniMax endpoint {base_url} must preserve thinking content, "
            f"not strip it like generic third-party endpoints"
        )
        assert "signature" not in thinking_blocks[0], (
            f"Signature must be stripped from signed thinking block on {base_url}"
        )
        assert thinking_blocks[0]["thinking"] == "Let me think about this."
