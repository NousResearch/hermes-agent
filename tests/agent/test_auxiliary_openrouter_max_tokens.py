"""Test that _build_call_kwargs preserves max_tokens for OpenRouter endpoints.

Regression test for #41035: OpenRouter free-tier credits exhausted when
max_tokens was stripped, causing HTTP 402 and fallback to text-only model.
"""

import pytest

from agent.auxiliary_client import _build_call_kwargs


class TestOpenRouterMaxTokens:
    """max_tokens must be included for OpenRouter to prevent free-tier 402."""

    def test_openrouter_provider_includes_max_tokens(self):
        """Direct openrouter provider keeps max_tokens."""
        kwargs = _build_call_kwargs(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2000,
        )
        assert kwargs["max_tokens"] == 2000

    def test_openrouter_base_url_includes_max_tokens(self):
        """Custom endpoint with openrouter.ai base_url keeps max_tokens."""
        kwargs = _build_call_kwargs(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2000,
            base_url="https://openrouter.ai/api/v1",
        )
        assert kwargs["max_tokens"] == 2000

    def test_generic_provider_omits_max_tokens(self):
        """Generic OpenAI-compatible provider still omits max_tokens."""
        kwargs = _build_call_kwargs(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2000,
        )
        assert "max_tokens" not in kwargs

    def test_anthropic_compat_still_includes_max_tokens(self):
        """Anthropic-compatible endpoints still include max_tokens."""
        kwargs = _build_call_kwargs(
            provider="minimax",
            model="MiniMax-Text-01",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=4000,
        )
        assert kwargs["max_tokens"] == 4000

    def test_none_max_tokens_never_included(self):
        """max_tokens=None is never added regardless of provider."""
        for provider, base_url in [
            ("openrouter", None),
            ("openai", "https://openrouter.ai/api/v1"),
            ("minimax", None),
        ]:
            kwargs = _build_call_kwargs(
                provider=provider,
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=None,
                base_url=base_url,
            )
            assert "max_tokens" not in kwargs, (
                f"max_tokens should not be set when None for {provider}"
            )
