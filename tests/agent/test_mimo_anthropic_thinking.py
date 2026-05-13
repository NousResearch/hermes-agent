"""Regression guard: preserve thinking blocks for Xiaomi MiMo (Anthropic path).

MiMo's API requires the prior ``reasoning_content`` to round-trip on every
replayed assistant turn when thinking mode is enabled (same contract as
Kimi / Moonshot and DeepSeek's ``/anthropic`` route).  Without preservation
the next request fails with HTTP 400::

    The reasoning_content in the thinking mode must be passed back to the
    API.

Detection in the Anthropic adapter is intentionally narrow: ``xiaomi/`` /
``mimo-`` model prefixes plus the embedded ``/mimo-`` third-party catalog
shape and the ``xiaomimimo.com`` host.  Lookalike model families
(MiniMax, Mistral, Microsoft Phi-4, ``mimowave`` substrings) must NOT
trigger the carve-out — they are negative-guarded below.

See hermes-agent#24443.
"""

from __future__ import annotations

import pytest


class TestModelNameIsXiaomiMimo:
    """``_model_name_is_xiaomi_mimo`` — three-shape positive matrix + negatives."""

    @pytest.mark.parametrize(
        "model",
        [
            "xiaomi/mimo-v2.5-pro",
            "xiaomi/mimo-v2.5-pro-thinking",
            "XIAOMI/MIMO-V2.5",
            "mimo-v2.5-pro",
            "mimo-v2",
            "MIMO-V2",
            "openrouter/mimo-v2.5-pro",
            "nous-portal/mimo-v2",
            "vendor/sub/mimo-v3",
            "  mimo-v2.5-pro  ",
        ],
    )
    def test_positive_signals(self, model: str) -> None:
        from agent.anthropic_adapter import _model_name_is_xiaomi_mimo

        assert _model_name_is_xiaomi_mimo(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            # Lookalike families — must NOT match
            "minimax-text-01",
            "minimax/abab-7-chat",
            "mistral-large",
            "mistralai/mistral-medium",
            "microsoft/phi-4",
            "phi-4-mimo-style",  # contains 'mimo' but not '/mimo-' or prefix
            "mimowave-7b",       # substring; must not match
            "openrouter/mimowave",
            # Underscore variants explicitly excluded (no published catalog)
            "mimo_v2",
            "xiaomi-mimo-v2",
            # Empty / non-string
            "",
            "   ",
            None,
            123,
        ],
    )
    def test_negative_signals(self, model) -> None:
        from agent.anthropic_adapter import _model_name_is_xiaomi_mimo

        assert _model_name_is_xiaomi_mimo(model) is False


class TestIsXiaomiMimoAnthropicEndpoint:
    """``_is_xiaomi_mimo_anthropic_endpoint`` — host + model combinatorics."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.xiaomimimo.com",
            "https://api.xiaomimimo.com/anthropic",
            "https://API.XiaomiMiMo.com/v1",
        ],
    )
    def test_host_signal(self, base_url: str) -> None:
        from agent.anthropic_adapter import _is_xiaomi_mimo_anthropic_endpoint

        assert _is_xiaomi_mimo_anthropic_endpoint(base_url) is True

    def test_model_signal_without_host(self) -> None:
        from agent.anthropic_adapter import _is_xiaomi_mimo_anthropic_endpoint

        assert _is_xiaomi_mimo_anthropic_endpoint(
            "https://openrouter.ai/api/v1/messages",
            "openrouter/mimo-v2.5-pro",
        ) is True

    def test_neither_host_nor_model(self) -> None:
        from agent.anthropic_adapter import _is_xiaomi_mimo_anthropic_endpoint

        assert _is_xiaomi_mimo_anthropic_endpoint(
            "https://api.minimax.io/anthropic", "minimax-text-01"
        ) is False
        assert _is_xiaomi_mimo_anthropic_endpoint(None, None) is False


class TestMimoAnthropicPreservesThinking:
    """convert_messages_to_anthropic must replay MiMo thinking blocks."""

    @pytest.mark.parametrize(
        "base_url, model",
        [
            ("https://api.xiaomimimo.com/anthropic", None),
            (None, "xiaomi/mimo-v2.5-pro"),
            (None, "mimo-v2.5-pro"),
            (None, "openrouter/mimo-v2.5-pro"),
        ],
    )
    def test_unsigned_thinking_block_survives_replay(
        self, base_url: str | None, model: str | None
    ) -> None:
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
            messages, base_url=base_url, model=model
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1, (
            f"MiMo (base_url={base_url!r}, model={model!r}) must preserve "
            "unsigned thinking blocks synthesised from reasoning_content — "
            "upstream rejects replayed tool-call messages without them."
        )
        assert thinking_blocks[0]["thinking"] == "planning the tool call"
        assert "signature" not in thinking_blocks[0]

    def test_unsigned_thinking_preserved_on_non_latest_assistant_turn(self) -> None:
        """MiMo validates every prior assistant turn, not just the last."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "q1"},
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
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "reasoning_content": "r2",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages,
            base_url="https://api.xiaomimimo.com/anthropic",
            model="mimo-v2.5-pro",
        )

        assistants = [m for m in converted if m["role"] == "assistant"]
        assert len(assistants) == 2
        for assistant, expected in zip(assistants, ("r1", "r2")):
            thinking = [
                b for b in assistant["content"]
                if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            assert len(thinking) == 1
            assert thinking[0]["thinking"] == expected

    def test_signed_anthropic_thinking_block_is_stripped(self) -> None:
        """Anthropic-signed blocks must still be stripped on MiMo.

        MiMo cannot validate Anthropic's proprietary signatures — strip
        signed blocks, preserve unsigned ones (Hermes-synthesised).
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "anthropic-signed payload",
                        "signature": "anthropic-sig-xyz",
                    },
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages,
            base_url="https://api.xiaomimimo.com/anthropic",
            model="mimo-v2.5-pro",
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == [], (
            "Signed Anthropic thinking blocks must be stripped on MiMo — "
            "MiMo cannot validate Anthropic-proprietary signatures."
        )

    def test_cache_control_stripped_from_thinking_block(self) -> None:
        """cache_control on thinking blocks must still be stripped on MiMo."""
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
            messages,
            base_url="https://api.xiaomimimo.com/anthropic",
            model="mimo-v2.5-pro",
        )
        for m in converted:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"):
                    assert "cache_control" not in b

    def test_minimax_lookalike_still_strips_all_thinking(self) -> None:
        """MiniMax (lookalike provider) must keep the generic strip-all path.

        Regression guard: a too-liberal MiMo matcher could accidentally
        enable MiMo's preserve-unsigned-thinking branch for MiniMax, which
        rejects unsigned blocks outright.
        """
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
            messages,
            base_url="https://api.minimax.io/anthropic",
            model="minimax-text-01",
        )
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == [], (
            "MiniMax must keep the generic strip-all-thinking behaviour — "
            "a too-liberal MiMo matcher would break MiniMax replays."
        )
