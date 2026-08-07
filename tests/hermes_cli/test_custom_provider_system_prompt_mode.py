"""Regression tests for custom_providers system_prompt_mode resolution.

Covers the fix for #76783 — an opt-in, provider-scoped compatibility mode
that moves Hermes's full runtime prompt out of the ``system`` message into
the first ``user`` message for OpenAI-compatible relays backed by Gemini
that reject long system content with HTTP 429 RESOURCE_EXHAUSTED.
"""
from __future__ import annotations

from hermes_cli.config import get_custom_provider_system_prompt_mode


class TestGetCustomProviderSystemPromptMode:

    def test_default_is_system(self):
        assert (
            get_custom_provider_system_prompt_mode(
                "gemini-3.1-pro-low", "https://example.invalid/v1", []
            )
            == "system"
        )
        assert (
            get_custom_provider_system_prompt_mode(
                "gemini-3.1-pro-low", "https://example.invalid/v1", None
            )
            == "system"
        )
        assert get_custom_provider_system_prompt_mode("m", "", []) == "system"

    def test_provider_level_mode(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "system_prompt_mode": "user",
            }
        ]
        assert (
            get_custom_provider_system_prompt_mode(
                "gemini-3.1-pro-low", "https://example.invalid/v1", custom
            )
            == "user"
        )

    def test_model_scoped_mode_wins_over_provider_level(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "system_prompt_mode": "user",
                "models": {
                    "gemini-3.1-pro-low": {"system_prompt_mode": "system"},
                    "other-model": {"system_prompt_mode": "developer"},
                },
            }
        ]
        # model-scoped "system" overrides the provider-level "user"
        assert (
            get_custom_provider_system_prompt_mode(
                "gemini-3.1-pro-low", "https://example.invalid/v1", custom
            )
            == "system"
        )
        # other model keeps its own scoped mode
        assert (
            get_custom_provider_system_prompt_mode(
                "other-model", "https://example.invalid/v1", custom
            )
            == "developer"
        )
        # unscoped model falls back to provider-level mode
        assert (
            get_custom_provider_system_prompt_mode(
                "unlisted-model", "https://example.invalid/v1", custom
            )
            == "user"
        )

    def test_trailing_slash_insensitive(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1/",
                "system_prompt_mode": "user",
            }
        ]
        assert (
            get_custom_provider_system_prompt_mode(
                "m", "https://example.invalid/v1", custom
            )
            == "user"
        )
        custom2 = [
            {
                "base_url": "https://example.invalid/v1",
                "system_prompt_mode": "user",
            }
        ]
        assert (
            get_custom_provider_system_prompt_mode(
                "m", "https://example.invalid/v1/", custom2
            )
            == "user"
        )

    def test_invalid_mode_values_fall_back_to_system(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "system_prompt_mode": "bogus",
            }
        ]
        assert (
            get_custom_provider_system_prompt_mode(
                "m", "https://example.invalid/v1", custom
            )
            == "system"
        )

    def test_wrong_base_url_does_not_match(self):
        custom = [
            {
                "base_url": "https://other.invalid/v1",
                "system_prompt_mode": "user",
            }
        ]
        assert (
            get_custom_provider_system_prompt_mode(
                "m", "https://example.invalid/v1", custom
            )
            == "system"
        )

    def test_mode_is_lowercased(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "system_prompt_mode": "User",
            }
        ]
        assert (
            get_custom_provider_system_prompt_mode(
                "m", "https://example.invalid/v1", custom
            )
            == "user"
        )
