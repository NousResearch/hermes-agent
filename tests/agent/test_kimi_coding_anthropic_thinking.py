"""Kimi / Moonshot thinking behavior on the Anthropic-Messages wire.

Contract (changed from the prior adaptive-thinking approach):

- Kimi's /coding endpoint no longer accepts Anthropic ``thinking`` parameters.
  Sending ``thinking.enabled`` (or ``thinking.type="adaptive"``) triggers HTTP
  400 when prior assistant tool-call messages lack OpenAI-style
  ``reasoning_content`` — a field the Anthropic path never populates. Kimi
  drives reasoning server-side on the /coding route, so the Anthropic thinking
  parameter must be omitted entirely.

- ``convert_messages_to_anthropic`` still preserves unsigned
  reasoning_content-derived thinking blocks on replay for this family (now
  unified with DeepSeek's ``_preserve_unsigned_thinking`` path), so
  multi-turn tool-call history round-trips.

Kimi on the chat_completions route handles ``thinking`` via ``extra_body``
in ``ChatCompletionsTransport`` (#13503).
"""

from __future__ import annotations

import pytest


class TestKimiCodingSkipsThinking:
    """Kimi /coding endpoint omits Anthropic thinking parameters."""

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
        assert "thinking" not in kwargs
        assert "output_config" not in kwargs

    def test_kimi_coding_with_enabled_still_omits_thinking(self) -> None:
        """Even with reasoning_config.enabled=True, Kimi /coding must NOT
        receive an Anthropic thinking parameter — the endpoint now rejects it
        with HTTP 400 when the message history lacks reasoning_content."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://api.kimi.com/coding",
        )
        assert "thinking" not in kwargs, (
            f"Kimi /coding must NOT receive thinking parameter, "
            f"got {kwargs.get('thinking')!r}"
        )
        assert "output_config" not in kwargs

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


class TestKimiFamilySkipsThinking:
    """Kimi-family endpoints must NOT receive Anthropic thinking parameters."""

    @pytest.mark.parametrize(
        "base_url,model",
        [
            # Official Kimi / Moonshot hosts (all URL shapes)
            ("https://api.kimi.com/coding", "kimi-k2.5"),
            ("https://api.kimi.com/coding/v1", "kimi-k2.5"),
            ("https://api.kimi.com/coding/anthropic", "kimi-k2.5"),
            ("https://api.kimi.com/v1", "kimi-k2.5"),
            ("https://api.moonshot.ai/anthropic", "moonshot-v1-32k"),
            ("https://api.moonshot.cn/anthropic", "moonshot-v1-32k"),
            ("https://api.moonshot.cn/anthropic/v1", "kimi-0714-preview"),
            # Custom / proxied hosts with a Kimi-family model (#17057)
            ("http://my-kimi-proxy.internal", "kimi-2.6"),
            ("https://llm.example.com/anthropic", "moonshotai/kimi-k2.5"),
        ],
    )
    def test_kimi_family_endpoint_skips_thinking(
        self, base_url: str, model: str
    ) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url=base_url,
        )
        assert "thinking" not in kwargs, (
            f"Kimi-family endpoint ({base_url}, {model}) must NOT receive "
            f"thinking parameter, got {kwargs.get('thinking')!r}"
        )
        assert "output_config" not in kwargs

    def test_kimi_thinking_disabled_omits_parameter(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-0714-preview",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
            base_url="https://api.moonshot.cn/anthropic/v1",
        )
        assert "thinking" not in kwargs
        assert "output_config" not in kwargs

    def test_custom_endpoint_non_kimi_model_keeps_thinking(self) -> None:
        """Custom endpoint with a non-Kimi model must keep thinking intact.

        Guards against over-broad model-family matching — only model names
        starting with a Kimi/Moonshot prefix should route to adaptive.
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
        the upstream's message-history validation passes."""
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
