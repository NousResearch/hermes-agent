"""Regression guard: don't send Anthropic ``thinking`` to Kimi's /coding endpoint.

Kimi's ``api.kimi.com/coding`` endpoint speaks the Anthropic Messages protocol
but has its own thinking semantics.  When ``thinking.enabled`` is present in
the request, Kimi validates the message history and requires every prior
assistant tool-call message to carry OpenAI-style ``reasoning_content``.

The Anthropic path never populates that field, and
``convert_messages_to_anthropic`` strips Anthropic thinking blocks on
third-party endpoints — so after one turn with tool calls the next request
fails with HTTP 400::

    thinking is enabled but reasoning_content is missing in assistant
    tool call message at index N

Kimi on the chat_completions route handles ``thinking`` via ``extra_body`` in
``ChatCompletionsTransport`` (#13503).  On the Anthropic route the right
thing to do is drop the parameter entirely and let Kimi drive reasoning
server-side.
"""

from __future__ import annotations

import pytest


class TestKimiCodingSkipsAnthropicThinking:
    """build_anthropic_kwargs must not inject ``thinking`` for Kimi /coding."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.kimi.com/coding",
            "https://api.kimi.com/coding/v1",
            "https://api.kimi.com/coding/anthropic",
            "https://api.kimi.com/coding/",
        ],
    )
    def test_kimi_coding_endpoint_omits_thinking(self, base_url: str) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=base_url,
        )
        assert "thinking" not in kwargs, (
            "Anthropic thinking must not be sent to Kimi /coding — "
            "endpoint requires reasoning_content on history we don't preserve."
        )
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

    def test_kimi_root_endpoint_unaffected(self) -> None:
        """Only the /coding route is special-cased — plain api.kimi.com is not.

        ``api.kimi.com`` without ``/coding`` uses the chat_completions transport
        (see runtime_provider._detect_api_mode_for_url); build_anthropic_kwargs
        should never see it, but if it somehow does we should not suppress
        thinking there — that path has different semantics.
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
