"""Regression guard: preserve MiMo thinking history on Xiaomi's Anthropic endpoint.

Xiaomi MiMo's Anthropic-compatible endpoint recently tightened thinking-mode
validation.  Once thinking mode is enabled, prior assistant turns that contain
``tool_use`` must replay their full ``reasoning_content`` on subsequent
requests.  If Hermes treats MiMo like a generic third-party Anthropic endpoint,
the signature-stripping pass removes the unsigned thinking block synthesized
from ``reasoning_content`` and MiMo returns HTTP 400::

    The reasoning_content in the thinking mode must be passed back to the API.

MiMo follows the same replay contract as Kimi / DeepSeek: strip
Anthropic-signed thinking blocks that Xiaomi cannot validate, but preserve
unsigned blocks that Hermes synthesized from ``reasoning_content``.
"""

from __future__ import annotations

import pytest


class TestXiaomiMiMoAnthropicPreservesThinking:
    """convert_messages_to_anthropic must replay MiMo thinking blocks."""

    @pytest.mark.parametrize(
        "base_url,model",
        [
            ("https://token-plan-sgp.xiaomimimo.com/anthropic", "mimo-v2.5-pro"),
            ("https://token-plan-sgp.xiaomimimo.com/anthropic/v1", "mimo-v2.5-pro"),
            ("https://TOKEN-PLAN-SGP.XIAOMIMIMO.COM/anthropic", "mimo-v2.5-pro"),
            ("https://llm.example.com/anthropic", "xiaomi-mimo-v2.5-pro"),
            ("https://llm.example.com/anthropic", "xiaomi/mimo-v2.5-pro"),
        ],
    )
    def test_unsigned_thinking_block_survives_replay(
        self, base_url: str, model: str
    ) -> None:
        """Unsigned thinking synthesized from reasoning_content must be preserved."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "planning the MiMo tool call",
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
            messages, base_url=base_url, model=model
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b
            for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "planning the MiMo tool call"
        assert "signature" not in thinking_blocks[0]

    def test_signed_anthropic_thinking_block_is_stripped(self) -> None:
        """Anthropic-signed thinking leaked from another provider must be stripped."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "anthropic signed payload",
                        "signature": "anthropic-sig-xyz",
                    },
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]

        _system, converted = convert_messages_to_anthropic(
            messages,
            base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
            model="mimo-v2.5-pro",
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b
            for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == []

    def test_non_mimo_third_party_still_strips_unsigned_thinking(self) -> None:
        """Generic third-party Anthropic endpoints keep strip-all behavior."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "r1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]

        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.minimax.io/anthropic", model="MiniMax-M2.7"
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b
            for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == []

    def test_mimo_keeps_anthropic_thinking_request_parameter(self) -> None:
        """MiMo still receives Anthropic thinking when reasoning is enabled."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="mimo-v2.5-pro",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
        )

        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["temperature"] == 1
