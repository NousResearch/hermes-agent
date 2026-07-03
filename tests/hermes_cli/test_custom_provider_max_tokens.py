"""Regression tests for custom_providers per-model max_tokens resolution.

Covers the fix for #28046 — startup must honor
``custom_providers[].models.<id>.max_tokens`` the same way it already does for
``context_length``.
"""
from __future__ import annotations

from hermes_cli.config import get_custom_provider_max_tokens


class TestGetCustomProviderMaxTokens:
    def test_returns_override_for_matching_entry(self):
        custom = [
            {
                "name": "xfyun",
                "base_url": "https://example.invalid/v2",
                "models": {"astron-code-latest": {"max_tokens": 32_000}},
            }
        ]
        assert (
            get_custom_provider_max_tokens(
                "astron-code-latest", "https://example.invalid/v2", custom
            )
            == 32_000
        )

    def test_trailing_slash_insensitive(self):
        custom = [
            {
                "base_url": "https://example.invalid/v2/",
                "models": {"m": {"max_tokens": 16_000}},
            }
        ]
        assert (
            get_custom_provider_max_tokens(
                "m", "https://example.invalid/v2", custom
            )
            == 16_000
        )

    def test_returns_none_when_url_does_not_match(self):
        custom = [
            {
                "base_url": "https://example.invalid/v2",
                "models": {"m": {"max_tokens": 8_000}},
            }
        ]
        assert (
            get_custom_provider_max_tokens(
                "m", "https://other.invalid/v2", custom
            )
            is None
        )

    def test_returns_none_for_bool_or_non_positive(self):
        for bad in (True, False, 0, -1, "32K"):
            custom = [
                {
                    "base_url": "https://example.invalid/v2",
                    "models": {"m": {"max_tokens": bad}},
                }
            ]
            assert (
                get_custom_provider_max_tokens(
                    "m", "https://example.invalid/v2", custom
                )
                is None
            ), f"value {bad!r} should be rejected"

    def test_empty_inputs_return_none(self):
        assert get_custom_provider_max_tokens("", "http://x", [{"base_url": "http://x", "models": {"": {"max_tokens": 1}}}]) is None
        assert get_custom_provider_max_tokens("m", "", [{"base_url": "", "models": {"m": {"max_tokens": 1}}}]) is None
        assert get_custom_provider_max_tokens("m", "http://x", None) is None
        assert get_custom_provider_max_tokens("m", "http://x", []) is None
