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

    def test_mimo_anthropic_tool_use_without_thinking_gets_placeholder(self) -> None:
        """Anthropic-format assistant turns with tool_use but no thinking must
        receive a placeholder thinking block on MiMo replay.

        This reproduces akaDRJ's live failure (#24726): 29 assistant tool-use
        turns in Anthropic content-block form, zero with thinking/reasoning
        data, causing HTTP 400 on MiMo's /anthropic endpoint.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "run a command"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_abc",
                        "name": "terminal",
                        "input": {"command": "echo hello"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "hello"},
            {"role": "user", "content": "result"},
        ]

        _system, converted = convert_messages_to_anthropic(
            messages,
            base_url="https://token-plan-cn.xiaomimimo.com/anthropic",
            model="mimo-v2.5-pro",
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        blocks = assistant_msg["content"]
        thinking_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "thinking"]
        tool_use_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]

        assert len(thinking_blocks) == 1, "Must have exactly one placeholder thinking block"
        assert thinking_blocks[0]["thinking"] == " "
        assert "signature" not in thinking_blocks[0]
        assert len(tool_use_blocks) == 1
        # Thinking block must come before tool_use (Anthropic protocol)
        assert blocks.index(thinking_blocks[0]) < blocks.index(tool_use_blocks[0])

    def test_non_mimo_tool_use_without_thinking_no_placeholder(self) -> None:
        """Generic third-party endpoints should NOT inject thinking placeholders."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_xyz",
                        "name": "terminal",
                        "input": {"command": "ls"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_xyz", "content": "file.txt"},
            {"role": "user", "content": "ok"},
        ]

        _system, converted = convert_messages_to_anthropic(
            messages,
            base_url="https://api.minimax.io/anthropic",
            model="MiniMax-M2.7",
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == []


class TestLookalikeModelNamesDoNotMatchMiMo:
    """Ensure adjacent model families are not falsely enrolled into MiMo enforcement."""

    @pytest.mark.parametrize(
        "model",
        [
            "minimax-text-01",
            "MiniMax-M2.7",
            "mistralai/mistral-medium",
            "microsoft/phi-4",
            "phi-4-mimo-style",
            "openrouter/mistralai/mistral-medium",
        ],
    )
    def test_lookalike_models_are_not_mimo(self, model: str) -> None:
        from agent.anthropic_adapter import _model_name_is_xiaomi_mimo

        assert _model_name_is_xiaomi_mimo(model) is False, (
            f"{model!r} should NOT be detected as Xiaomi MiMo"
        )

    def test_mimowave_is_not_mimo(self) -> None:
        """mimowave-7b does not match any MiMo prefix (no dash after 'mimo')."""
        from agent.anthropic_adapter import _model_name_is_xiaomi_mimo

        assert _model_name_is_xiaomi_mimo("mimowave-7b") is False


class TestDeepNamespacedModelNames:
    """Deep-namespaced model names like vendor/sub/mimo-v3 must still match."""

    @pytest.mark.parametrize(
        "model",
        [
            "vendor/sub/mimo-v3",
            "a/b/c/xiaomi-mimo-v2.5-pro",
            "openrouter/xiaomi/mimo-v2.5-pro",
        ],
    )
    def test_deep_namespace_matches(self, model: str) -> None:
        from agent.anthropic_adapter import _model_name_is_xiaomi_mimo

        assert _model_name_is_xiaomi_mimo(model) is True, (
            f"{model!r} should be detected as Xiaomi MiMo"
        )
