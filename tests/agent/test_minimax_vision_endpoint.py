"""#69287: minimax-cn with /v1 base URL must use OpenAI-format image blocks,
not Anthropic format. Silent hallucination occurs when the wrong format
is sent to the Chat Completions endpoint.
"""

from __future__ import annotations

import pytest

from agent.auxiliary_client import _is_anthropic_compat_endpoint


class TestAnthropicCompatDetection:
    def test_minimax_cn_default_url_is_anthropic_compat(self):
        """minimax-cn with default base URL (None) is Anthropic-compat."""
        assert _is_anthropic_compat_endpoint("minimax-cn", "") is True

    def test_minimax_cn_with_anthropic_url_is_compat(self):
        """minimax-cn with /anthropic URL is Anthropic-compat."""
        assert _is_anthropic_compat_endpoint("minimax-cn", "https://api.minimaxi.com/anthropic") is True

    def test_minimax_cn_with_v1_url_is_not_anthropic_compat(self):
        """minimax-cn with /v1 URL must NOT be treated as Anthropic-compat (#69287)."""
        assert _is_anthropic_compat_endpoint("minimax-cn", "https://api.minimaxi.com/v1") is False

    def test_minimax_with_v1_url_is_not_anthropic_compat(self):
        """minimax (global) with /v1 URL must NOT be treated as Anthropic-compat."""
        assert _is_anthropic_compat_endpoint("minimax", "https://api.minimax.io/v1") is False

    def test_minimax_oauth_with_v1_url_is_not_anthropic_compat(self):
        """minimax-oauth with /v1 URL must NOT be treated as Anthropic-compat."""
        assert _is_anthropic_compat_endpoint("minimax-oauth", "https://api.minimax.io/v1") is False

    def test_non_minimax_provider_with_anthropic_url_is_compat(self):
        """Non-MiniMax providers with /anthropic URL are still compat."""
        assert _is_anthropic_compat_endpoint("custom", "https://example.com/anthropic") is True

    def test_non_minimax_provider_without_anthropic_url_is_not_compat(self):
        """Non-MiniMax providers without /anthropic URL are not compat."""
        assert _is_anthropic_compat_endpoint("custom", "https://api.openai.com/v1") is False

    def test_empty_provider_and_url_is_not_compat(self):
        assert _is_anthropic_compat_endpoint("", "") is False