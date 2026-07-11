"""Regression test: send Anthropic ``thinking`` to all Kimi endpoints.

The Kimi /coding endpoint speaks the Anthropic Messages protocol and supports
the ``thinking`` parameter.  When omitted, Kimi enables extended thinking
server-side by default.  The guard that previously suppressed the parameter
was removed after validating that the Kimi endpoint handles thinking correctly
without triggering ``reasoning_content`` validation errors.  See #56730, #56727.
"""

from __future__ import annotations

import pytest


class TestKimiCodingReceivesAnthropicThinking:
    """build_anthropic_kwargs must inject ``thinking`` for all Kimi endpoints."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.kimi.com/coding",
            "https://api.kimi.com/coding/v1",
            "https://api.kimi.com/coding/anthropic",
            "https://api.kimi.com/coding/",
        ],
    )
    def test_kimi_coding_endpoint_gets_thinking(self, base_url: str) -> None:
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
            "Anthropic thinking must be sent to Kimi /coding — "
            "endpoint handles the parameter correctly."
        )

    def test_kimi_coding_with_explicit_disabled_omits_thinking(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
            base_url="https://api.kimi.com/coding",
        )
        # When reasoning_config.enabled is False, thinking is omitted entirely
        # (not sent as thinking.disabled).  This matches Anthropic's behavior
        # and allows Kimi to use its default server-side reasoning.
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

    def test_kimi_root_endpoint_via_anthropic_transport_gets_thinking(self) -> None:
        """Plain ``api.kimi.com`` non-/coding endpoint keeps thinking.

        The guard was removed — all Kimi endpoints now receive the thinking
        parameter.  Kimi's /coding endpoint handles thinking correctly without
        triggering ``reasoning_content`` validation errors.  See #56730, #56727.
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
        assert "thinking" in kwargs

    # ── #56727: custom / proxied Kimi-compatible endpoints now get thinking ──
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
    def test_kimi_family_custom_endpoint_gets_thinking(
        self, base_url: str, model: str
    ) -> None:
        """Custom / proxied Kimi endpoints must now receive Anthropic thinking.

        The guard was removed after validating that Kimi endpoints handle
        thinking correctly without triggering ``reasoning_content`` validation
        errors.  See #56730, #56727.
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=base_url,
        )
        assert "thinking" in kwargs, (
            f"Kimi-family endpoint ({base_url}, {model}) must receive "
            f"Anthropic thinking — upstream validates reasoning_content on "
            f"replayed tool-call history but the parameter is no longer suppressed."
        )

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
