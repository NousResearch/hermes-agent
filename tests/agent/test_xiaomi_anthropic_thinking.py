"""Regression guard: preserve thinking blocks on Xiaomi MiMo's /anthropic endpoint.

Xiaomi MiMo's ``token-plan-cn.xiaomimimo.com/anthropic`` route speaks the
Anthropic Messages protocol but, in thinking mode, requires the
``reasoning_content`` / unsigned ``thinking`` blocks from prior assistant
turns to round-trip on subsequent requests — same contract as Kimi
``/coding`` and DeepSeek ``/anthropic``. Without an MiMo-specific
carve-out the generic third-party path strips the unsigned thinking
blocks and the next tool-call turn fails with HTTP 400::

    The reasoning_content in the thinking mode must be passed back to
    the API.

Handling mirrors the DeepSeek policy: strip Anthropic-signed blocks
(MiMo cannot validate Anthropic signatures) but preserve unsigned
blocks that Hermes synthesises from ``reasoning_content``.
"""

from __future__ import annotations

import pytest


class TestXiaomiAnthropicPreservesThinking:
    """convert_messages_to_anthropic must replay Xiaomi MiMo thinking blocks."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://token-plan-cn.xiaomimimo.com/anthropic",
            "https://token-plan-cn.xiaomimimo.com/anthropic/",
            "https://token-plan-cn.xiaomimimo.com/anthropic/v1",
            "https://Token-Plan-CN.XiaoMiMimo.com/anthropic",
        ],
    )
    def test_unsigned_thinking_block_survives_replay(self, base_url: str) -> None:
        """Unsigned thinking (synthesised from reasoning_content) must be preserved."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "planning the tool call",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "skill_view", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
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
            f"Xiaomi MiMo /anthropic ({base_url}) must preserve unsigned "
            "thinking blocks synthesised from reasoning_content — upstream "
            "rejects replayed tool-call messages without them."
        )
        assert thinking_blocks[0]["thinking"] == "planning the tool call"
        # Synthesised block — never has a signature
        assert "signature" not in thinking_blocks[0]

    def test_openai_compat_xiaomi_base_is_not_matched(self) -> None:
        """Only the ``/anthropic`` path triggers the preservation policy.

        Other Xiaomi base URLs (OpenAI-compat / bare host) reach this adapter
        only via misconfiguration; the helper must not misclassify them.
        """
        from agent.anthropic_adapter import _is_xiaomi_anthropic_endpoint

        assert _is_xiaomi_anthropic_endpoint("https://token-plan-cn.xiaomimimo.com") is False
        assert _is_xiaomi_anthropic_endpoint("https://token-plan-cn.xiaomimimo.com/v1") is False
        assert _is_xiaomi_anthropic_endpoint("https://token-plan-cn.xiaomimimo.com/anthropic") is True
        assert _is_xiaomi_anthropic_endpoint("https://token-plan-cn.xiaomimimo.com/anthropic/v1") is True
