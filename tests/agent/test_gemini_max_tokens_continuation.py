"""Tests for Gemini max output tokens defaults and text continuation token boosting."""

from unittest.mock import MagicMock
import pytest

from providers import get_provider_profile
from agent.chat_completion_helpers import _get_gemini_max_output


class TestGeminiMaxTokensDefaults:
    def test_gemini_profile_get_max_tokens(self):
        profile = get_provider_profile("gemini")
        assert profile is not None
        assert profile.get_max_tokens("gemini-3.6-flash") == 65535
        assert profile.get_max_tokens("google/gemini-2.5-pro") == 65535

    def test_openrouter_profile_gemini_max_tokens(self):
        profile = get_provider_profile("openrouter")
        assert profile is not None
        assert profile.get_max_tokens("google/gemini-3-flash-preview") == 65535
        assert profile.get_max_tokens("anthropic/claude-3-5-sonnet") is None

    def test_nous_profile_gemini_max_tokens(self):
        profile = get_provider_profile("nous")
        assert profile is not None
        assert profile.get_max_tokens("google/gemini-3.6-flash") == 65535
        assert profile.get_max_tokens("hermes-3-70b") is None

    def test_chat_completion_helpers_gemini_max_output(self):
        assert _get_gemini_max_output("gemini-3.6-flash") == 65535
        assert _get_gemini_max_output("google/gemini-3-flash-preview") == 65535
        assert _get_gemini_max_output("claude-sonnet-4") is None
