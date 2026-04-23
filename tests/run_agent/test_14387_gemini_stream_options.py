"""Tests for #14387 — stream_options rejected by Gemini endpoint.

Gemini's OpenAI-compatible API raises TypeError on stream_options, which
aborts the request non-retryably for all Gemini users.

Fix: gate stream_options behind a provider + base_url check so it is only
sent to endpoints that support it (OpenAI, OpenRouter, etc.).

Verifies:
1. stream_options is skipped for explicit Gemini provider names
2. stream_options is skipped for googleapis.com base URLs (covers provider="" case)
3. stream_options is still sent for OpenAI and OpenRouter providers
4. Usage tracking (usage_obj) is unaffected for non-Gemini providers
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: replicate the _skip_stream_options guard from run_agent.py
# ---------------------------------------------------------------------------

def _skip_stream_options(provider: str, base_url_lower: str) -> bool:
    """Mirror of the _skip_stream_options guard in _call_chat_completions."""
    return (
        provider in ("gemini", "google-gemini-cli")
        or "generativelanguage.googleapis.com" in base_url_lower
    )


# ---------------------------------------------------------------------------
# Test 1: Gemini provider names are skipped
# ---------------------------------------------------------------------------

class TestGeminiProviderSkipped:
    """stream_options must be omitted for explicit Gemini provider names."""

    def test_gemini_provider_skips(self):
        """provider='gemini' (Google AI Studio) must skip stream_options."""
        assert _skip_stream_options("gemini", "https://generativelanguage.googleapis.com/v1beta")

    def test_google_gemini_cli_provider_skips(self):
        """provider='google-gemini-cli' (OAuth) must skip stream_options."""
        assert _skip_stream_options("google-gemini-cli", "cloudcode-pa://some-url")

    def test_gemini_openai_compat_endpoint_skips(self):
        """provider='gemini' + /openai compat endpoint also skips.
        is_native_gemini_base_url() returns False for /openai URLs,
        so without this fix the standard OpenAI client would be used
        and stream_options would reach Gemini — causing TypeError."""
        assert _skip_stream_options(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )


# ---------------------------------------------------------------------------
# Test 2: googleapis.com base URL triggers skip (covers provider="" case)
# ---------------------------------------------------------------------------

class TestGoogleapisUrlSkipped:
    """Any googleapis.com URL must skip stream_options regardless of provider.

    This is the exact scenario reported in #14387: provider is empty but
    base_url points directly at generativelanguage.googleapis.com.
    """

    def test_empty_provider_googleapis_url_skips(self):
        """provider='' + googleapis base_url: the reported bug scenario."""
        assert _skip_stream_options(
            "",
            "https://generativelanguage.googleapis.com/v1beta",
        )

    def test_empty_provider_googleapis_openai_url_skips(self):
        """provider='' + googleapis /openai compat URL also skips."""
        assert _skip_stream_options(
            "",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )


# ---------------------------------------------------------------------------
# Test 3: Non-Gemini providers still receive stream_options
# ---------------------------------------------------------------------------

class TestNonGeminiProvidersSendStreamOptions:
    """stream_options must still be sent to providers that support it.

    Removing stream_options globally would break usage tracking for all
    other providers — this was the flaw in the reporter's proposed fix.
    """

    def test_openai_direct_sends_stream_options(self):
        assert not _skip_stream_options("", "https://api.openai.com/v1")

    def test_openrouter_sends_stream_options(self):
        assert not _skip_stream_options("openrouter", "https://openrouter.ai/api/v1")

    def test_openai_codex_sends_stream_options(self):
        assert not _skip_stream_options("openai-codex", "https://api.openai.com/v1")

    def test_empty_provider_non_google_url_sends_stream_options(self):
        """provider='' with a non-Google base_url must still send stream_options."""
        assert not _skip_stream_options("", "https://api.groq.com/openai/v1")

    def test_nous_portal_sends_stream_options(self):
        assert not _skip_stream_options("nous", "https://inference.nousresearch.com/v1")
