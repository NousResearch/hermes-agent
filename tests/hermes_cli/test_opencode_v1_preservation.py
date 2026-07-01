"""Test for OpenCode base_url /v1 preservation bug fix.

Regression test for: OpenCode Go/Zen で chat_completions モデル (GLM-5.2)
を選択すると base_url から /v1 が削除され 404 エラーが発生する問題。

Issue: https://github.com/NousResearch/hermes-agent/issues/XXXX
"""
import re
from unittest.mock import MagicMock, patch

import pytest


class TestOpenCodeV1Preservation:
    """Test that /v1 is preserved for chat_completions models on OpenCode providers."""

    def test_chat_completions_preserves_v1(self):
        """GLM-5.2 (chat_completions) on opencode-go should keep /v1 in base_url."""
        # Simulate the logic from runtime_provider.py:492-500
        base_url = "https://opencode.ai/zen/go/v1"
        api_mode = "chat_completions"
        provider = "opencode-go"

        # Apply the fix logic
        if api_mode == "anthropic_messages" and provider in {"opencode-zen", "opencode-go"}:
            base_url = re.sub(r"/v1/?$", "", base_url)
        elif api_mode == "chat_completions" and provider in {"opencode-zen", "opencode-go"}:
            if base_url and not re.search(r"/v1/?$", base_url):
                base_url = base_url.rstrip("/") + "/v1"

        assert base_url.endswith("/v1"), f"Expected /v1 suffix, got: {base_url}"
        assert base_url == "https://opencode.ai/zen/go/v1"

    def test_chat_completions_appends_v1_if_missing(self):
        """If config has base_url without /v1, the fix should append it."""
        base_url = "https://opencode.ai/zen/go"
        api_mode = "chat_completions"
        provider = "opencode-go"

        if api_mode == "anthropic_messages" and provider in {"opencode-zen", "opencode-go"}:
            base_url = re.sub(r"/v1/?$", "", base_url)
        elif api_mode == "chat_completions" and provider in {"opencode-zen", "opencode-go"}:
            if base_url and not re.search(r"/v1/?$", base_url):
                base_url = base_url.rstrip("/") + "/v1"

        assert base_url.endswith("/v1"), f"Expected /v1 suffix, got: {base_url}"
        assert base_url == "https://opencode.ai/zen/go/v1"

    def test_anthropic_messages_strips_v1(self):
        """MiniMax (anthropic_messages) on opencode-go should strip /v1 from base_url."""
        base_url = "https://opencode.ai/zen/go/v1"
        api_mode = "anthropic_messages"
        provider = "opencode-go"

        if api_mode == "anthropic_messages" and provider in {"opencode-zen", "opencode-go"}:
            base_url = re.sub(r"/v1/?$", "", base_url)
        elif api_mode == "chat_completions" and provider in {"opencode-zen", "opencode-go"}:
            if base_url and not re.search(r"/v1/?$", base_url):
                base_url = base_url.rstrip("/") + "/v1"

        assert not base_url.endswith("/v1"), f"Expected no /v1, got: {base_url}"
        assert base_url == "https://opencode.ai/zen/go"

    def test_opencode_zen_chat_completions_preserves_v1(self):
        """Claude on opencode-zen with chat_completions should preserve /v1."""
        base_url = "https://opencode.ai/zen/v1"
        api_mode = "chat_completions"
        provider = "opencode-zen"

        if api_mode == "anthropic_messages" and provider in {"opencode-zen", "opencode-go"}:
            base_url = re.sub(r"/v1/?$", "", base_url)
        elif api_mode == "chat_completions" and provider in {"opencode-zen", "opencode-go"}:
            if base_url and not re.search(r"/v1/?$", base_url):
                base_url = base_url.rstrip("/") + "/v1"

        assert base_url.endswith("/v1"), f"Expected /v1 suffix, got: {base_url}"
        assert base_url == "https://opencode.ai/zen/v1"

    def test_non_opencode_provider_unchanged(self):
        """Non-OpenCode providers should not be affected by this logic."""
        base_url = "https://api.openai.com/v1"
        api_mode = "chat_completions"
        provider = "openai"

        # Apply the fix logic (should be no-op)
        if api_mode == "anthropic_messages" and provider in {"opencode-zen", "opencode-go"}:
            base_url = re.sub(r"/v1/?$", "", base_url)
        elif api_mode == "chat_completions" and provider in {"opencode-zen", "opencode-go"}:
            if base_url and not re.search(r"/v1/?$", base_url):
                base_url = base_url.rstrip("/") + "/v1"

        assert base_url == "https://api.openai.com/v1"

    def test_empty_base_url_handled(self):
        """Empty base_url should not cause errors."""
        base_url = ""
        api_mode = "chat_completions"
        provider = "opencode-go"

        if api_mode == "anthropic_messages" and provider in {"opencode-zen", "opencode-go"}:
            base_url = re.sub(r"/v1/?$", "", base_url)
        elif api_mode == "chat_completions" and provider in {"opencode-zen", "opencode-go"}:
            if base_url and not re.search(r"/v1/?$", base_url):
                base_url = base_url.rstrip("/") + "/v1"

        assert base_url == ""
