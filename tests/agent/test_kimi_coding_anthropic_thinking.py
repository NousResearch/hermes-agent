"""Tests for Kimi /coding reasoning via anthropic_messages transport.

Kimi's ``api.kimi.com/coding`` endpoint speaks the Anthropic Messages protocol
and supports Anthropic-style ``thinking`` with signed blocks.  When
``thinking.enabled`` is present in the request, Kimi validates the message
history and requires every prior assistant tool-call message to carry
OpenAI-style ``reasoning_content``.

This was previously broken because the adapter stripped all signed thinking
blocks on third-party endpoints, causing HTTP 400 after one turn with tool
calls.  Upstream commit ``76edc40ab`` added ``_copy_reasoning_content_for_api()``
to inject ``reasoning_content: " "`` on replay, and the adapter now preserves
Kimi's own signed blocks for ``api.kimi.com/coding`` endpoints.
"""

from __future__ import annotations

import pytest


class TestKimiCodingSkipsAnthropicThinking:
    """build_anthropic_kwargs must inject ``thinking`` for Kimi /coding
    when reasoning is enabled, because Kimi validates its own signatures
    server-side."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.kimi.com/coding",
            "https://api.kimi.com/coding/v1",
            "https://api.kimi.com/coding/anthropic",
            "https://api.kimi.com/coding/",
        ],
    )
    def test_kimi_coding_endpoint_includes_thinking(self, base_url: str) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=base_url,
        )
        assert "thinking" in kwargs, (
            "Kimi /coding supports Anthropic thinking parameter; "
            "upstream run_agent.py now preserves reasoning_content on replay."
        )
        assert kwargs["thinking"]["type"] == "enabled"
        assert "output_config" not in kwargs

    def test_kimi_coding_with_explicit_disabled_also_omits(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
            base_url="https://api.kimi.com/coding",
        )
        assert "thinking" not in kwargs

    def test_non_kimi_third_party_still_gets_thinking(self) -> None:
        """MiniMax and other third-party Anthropic endpoints must retain thinking."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.minimax.io/anthropic",
        )
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"

    def test_native_anthropic_still_gets_thinking(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=None,
        )
        assert "thinking" in kwargs

    def test_kimi_root_endpoint_via_anthropic_transport_omits_thinking(self) -> None:
        """Plain ``api.kimi.com`` hit via the Anthropic transport also omits thinking.

        Auto-detection routes ``api.kimi.com/v1`` to ``chat_completions`` by
        default, but users can explicitly configure
        ``api_mode: anthropic_messages`` against any Kimi host.  The upstream
        validation (reasoning_content required on replayed tool-call
        messages) is the same regardless of URL path, so the thinking
        suppression must apply to every Kimi host, not just ``/coding``.
        See #17057.
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.kimi.com/v1",
        )
        assert "thinking" not in kwargs

    # ── #17057: custom / proxied Kimi-compatible endpoints ──────────
    @pytest.mark.parametrize(
        "base_url,model",
        [
            # Custom host with Kimi-family model — the reporter's case
            ("http://my-kimi-proxy.internal", "kimi-2.6"),
            ("https://llm.example.com/anthropic", "kimi-k2.5"),
            ("https://llm.example.com/anthropic", "moonshot-v1-8k"),
            ("https://llm.example.com/anthropic", "kimi_thinking"),
            ("https://llm.example.com/anthropic", "moonshotai/kimi-k2.5"),
            # Official Moonshot host (previously uncovered)
            ("https://api.moonshot.ai/anthropic", "moonshot-v1-32k"),
            ("https://api.moonshot.cn/anthropic", "moonshot-v1-32k"),
        ],
    )
    def test_kimi_family_custom_endpoint_omits_thinking(
        self, base_url: str, model: str
    ) -> None:
        """Custom / proxied Kimi endpoints must also strip Anthropic thinking."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=base_url,
        )
        assert "thinking" not in kwargs, (
            f"Kimi-family endpoint ({base_url}, {model}) must not receive "
            f"Anthropic thinking — upstream validates reasoning_content on "
            f"replayed tool-call history we don't preserve."
        )
        assert "output_config" not in kwargs

    def test_custom_endpoint_non_kimi_model_keeps_thinking(self) -> None:
        """Custom endpoint with a non-Kimi model must keep thinking intact.

        Guards against over-broad model-family matching — only model names
        starting with a Kimi/Moonshot prefix should trigger suppression.
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://my-llm-proxy.example.com/anthropic",
        )
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"

    def test_kimi_family_replay_preserves_unsigned_thinking(self) -> None:
        """On a custom Kimi endpoint, unsigned reasoning_content thinking
        blocks must survive the third-party signature-stripping pass so
        the upstream's message-history validation passes.
        """
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
        _, converted = convert_messages_to_anthropic(
            messages,
            base_url="http://my-kimi-proxy.internal",
            model="kimi-2.6",
        )
        # The assistant message still carries the unsigned thinking block
        # synthesised from reasoning_content (required by Kimi's history
        # validation).  A plain third-party endpoint would have stripped it.
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        assistant_blocks = assistant_msg["content"]
        thinking_blocks = [
            b for b in assistant_blocks
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "planning the tool call"

    def test_kimi_coding_preserves_signed_thinking_blocks(self) -> None:
        """Signed thinking blocks from Kimi itself must survive round-trip."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I will call a tool",
                        "signature": "EpUCCkYICxgCKkABh4..." + "x" * 100,
                    },
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "skill_view",
                        "input": {},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        _, converted = convert_messages_to_anthropic(
            messages,
            base_url="https://api.kimi.com/coding",
            model="kimi-k2.5",
        )
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        assistant_blocks = assistant_msg["content"]
        thinking_blocks = [
            b for b in assistant_blocks
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].get("signature") is not None

    def test_kimi_coding_thinking_with_tool_call_replay(self) -> None:
        """Full round-trip: reasoning + tool call preserves reasoning_content."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "find btc price"},
            {
                "role": "assistant",
                "reasoning_content": "I need to search for current BTC price",
                "tool_calls": [
                    {
                        "id": "call_btc",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"q": "BTC price"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_btc", "content": "$87,000"},
        ]
        _, converted = convert_messages_to_anthropic(
            messages,
            base_url="https://api.kimi.com/coding",
            model="kimi-k2.5",
        )
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        # reasoning_content should be synthesised into unsigned thinking block
        # AND preserved because Kimi /coding is whitelisted
        assert "content" in assistant_msg
        has_thinking = any(
            isinstance(b, dict) and b.get("type") == "thinking"
            for b in assistant_msg["content"]
        )
        assert has_thinking, "reasoning_content must survive as thinking block for Kimi /coding"
