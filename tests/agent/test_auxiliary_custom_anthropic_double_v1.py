"""Tests for the anonymous custom provider branch's anthropic_messages handling.

When ``resolve_provider_client('custom', ...)` is called with an
``explicit_base_url`` and ``api_mode='anthropic_messages'``, the base URL
must NOT be rewritten with ``_to_openai_base_url`` (which appends ``/v1``).
The Anthropic SDK constructs its own ``/v1/messages`` path, so a pre-appended
``/v1`` produces a double-``/v1`` URL (e.g. ``/coding/v1/v1/messages`` → 404).

Regression test for #19753.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _mock_build_anthropic_client():
    """Return a (patch, fake_client) pair for build_anthropic_client."""
    fake_client = MagicMock(name="anthropic_client")
    return patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_client,
    ), fake_client


class TestAnonymousCustomAnthropicBaseUrl:
    """Anonymous custom branch must preserve raw base_url for anthropic_messages."""

    def test_anthropic_messages_no_double_v1(self):
        """api_mode=anthropic_messages must NOT append /v1 to the base URL."""
        from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

        raw_url = "https://api.kimi.com/coding"

        adapter_patch, fake_client = _mock_build_anthropic_client()
        with adapter_patch as mock_build:
            client, model = resolve_provider_client(
                "custom",
                model="kimi-for-coding",
                explicit_base_url=raw_url,
                explicit_api_key="test-key",
                api_mode="anthropic_messages",
            )

        assert isinstance(client, AnthropicAuxiliaryClient), (
            f"Expected AnthropicAuxiliaryClient, got {type(client).__name__}"
        )
        # The key assertion: build_anthropic_client must receive the RAW URL,
        # not the /v1-rewritten one.
        mock_build.assert_called_once_with("test-key", raw_url)

    def test_anthropic_messages_with_trailing_slash(self):
        """Trailing slash on base_url should be stripped, not cause //v1."""
        from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

        raw_url = "https://api.kimi.com/coding/"

        adapter_patch, fake_client = _mock_build_anthropic_client()
        with adapter_patch as mock_build:
            client, model = resolve_provider_client(
                "custom",
                model="kimi-for-coding",
                explicit_base_url=raw_url,
                explicit_api_key="test-key",
                api_mode="anthropic_messages",
            )

        assert isinstance(client, AnthropicAuxiliaryClient)
        # Should be called with trailing-slash-stripped URL
        mock_build.assert_called_once_with("test-key", "https://api.kimi.com/coding")

    def test_chat_completions_still_appends_v1(self):
        """Non-anthropic mode should still get /v1 appended (existing behavior)."""
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI

        raw_url = "https://api.example.com/v1"

        client, model = resolve_provider_client(
            "custom",
            model="gpt-4o-mini",
            explicit_base_url=raw_url,
            explicit_api_key="test-key",
            api_mode="chat_completions",
        )

        # For chat_completions, the OpenAI client should get the /v1 URL
        assert isinstance(client, OpenAI)
        assert "/v1" in str(client.base_url)

    def test_explicit_anthropic_mode_wraps_correctly(self):
        """When api_mode is explicitly anthropic_messages, _wrap_if_needed
        receives the raw URL and _maybe_wrap_anthropic wraps it correctly."""
        from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient

        raw_url = "https://litellm.internal/anthropic"

        adapter_patch, fake_client = _mock_build_anthropic_client()
        with adapter_patch:
            client, model = resolve_provider_client(
                "custom",
                model="claude-sonnet-4-6",
                explicit_base_url=raw_url,
                explicit_api_key="litellm-key",
                api_mode="anthropic_messages",
            )

        assert isinstance(client, AnthropicAuxiliaryClient)
        assert client.base_url == raw_url
        assert client.api_key == "litellm-key"
