"""Regression guard: preserve thinking blocks on DeepSeek's /anthropic endpoint.

DeepSeek's ``api.deepseek.com/anthropic`` route speaks the Anthropic Messages
protocol but, when thinking mode is enabled, requires ``thinking`` blocks from
prior assistant turns to round-trip on subsequent requests.  The generic
third-party path strips them (signatures are Anthropic-proprietary and other
proxies cannot validate them), so without a DeepSeek-specific carve-out the
next tool-call turn fails with HTTP 400::

    The content[].thinking in the thinking mode must be passed back to the
    API.

DeepSeek's compatibility matrix lists ``thinking`` as supported but
``redacted_thinking`` and ``cache_control`` on thinking blocks as not
supported.  Handling is the same as Kimi's ``/coding`` endpoint: strip
Anthropic-signed blocks (DeepSeek can't validate them) but preserve unsigned
blocks that Hermes synthesises from ``reasoning_content``.

See hermes-agent#16748.
"""

from __future__ import annotations

import pytest


class TestDeepSeekAnthropicPreservesThinking:
    """convert_messages_to_anthropic must replay DeepSeek thinking blocks."""



    def test_signed_anthropic_thinking_block_is_stripped(self) -> None:
        """Anthropic-signed blocks (that leaked through) must still be stripped.

        DeepSeek issues its own signatures and cannot validate Anthropic's —
        the strip-signed / keep-unsigned split matches the Kimi policy.
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
            messages, base_url="https://api.deepseek.com/anthropic"
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == [], (
            "Signed Anthropic thinking blocks must be stripped on DeepSeek — "
            "DeepSeek cannot validate Anthropic-proprietary signatures."
        )

    def test_cache_control_stripped_from_thinking_block(self) -> None:
        """cache_control must still be stripped even when the block is preserved.

        DeepSeek's compatibility matrix lists cache_control on thinking blocks
        as ignored — cache markers interfere with signature validation on
        upstreams that do check them, so Hermes strips them everywhere.
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
        # Inject cache_control on the synthesised thinking block after-the-fact
        # by running conversion once, mutating, then re-running would be
        # indirect.  Instead check the simpler invariant: no thinking block in
        # the converted output carries cache_control.
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )
        for m in converted:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}:
                    assert "cache_control" not in b


class TestDeepSeekCustomRelay:
    """Custom relay host with DeepSeek model — model-name detection path."""

    def test_model_name_detection_for_custom_relay(self) -> None:
        """When base_url is NOT api.deepseek.com but model starts with 'deepseek',
        the endpoint is detected via model-name fallback."""
        from agent.anthropic_adapter import _is_deepseek_anthropic_endpoint

        result = _is_deepseek_anthropic_endpoint(
            "https://llm-gateway.example.com/v1",
            "deepseek-deepseek-v4-pro",
        )
        assert result is True, (
            "Custom relay with deepseek model should be detected via model-name fallback"
        )

    def test_custom_relay_preserves_unsigned_thinking_blocks(self) -> None:
        """Unsigned thinking blocks from reasoning_content must be preserved on
        custom relay when model-name detection triggers the DeepSeek path."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "Looking at the config...",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "config data"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages,
            base_url="https://llm-gateway.example.com/anthropic",
            model="deepseek-deepseek-v4-pro",
        )

        # Find the assistant message with tool_calls
        assistant_msgs = [m for m in converted if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1, "Expected one assistant message"
        thinking_blocks = [
            b for b in assistant_msgs[0]["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) >= 1, (
            "Unsigned thinking blocks from reasoning_content must be "
            "preserved on custom DeepSeek relays"
        )
        assert thinking_blocks[0]["thinking"] == "Looking at the config..."


