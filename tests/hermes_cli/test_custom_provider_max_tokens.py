"""Regression tests for custom_providers per-model max_tokens resolution.

Covers the fix for #28046 — custom_providers per-model max_tokens was silently
ignored, always defaulting to 4096.  The fix adds a symmetric lookup to the
existing context_length mechanism via get_custom_provider_model_field.
"""
from __future__ import annotations

from hermes_cli.config import get_custom_provider_max_tokens


class TestGetCustomProviderMaxTokens:
    def test_returns_override_for_matching_entry(self):
        custom = [
            {
                "name": "my-endpoint",
                "base_url": "https://example.invalid/v1",
                "models": {"gpt-5.5": {"max_tokens": 16384}},
            }
        ]
        assert (
            get_custom_provider_max_tokens(
                "gpt-5.5", "https://example.invalid/v1", custom
            )
            == 16384
        )

    def test_trailing_slash_insensitive(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1/",
                "models": {"m": {"max_tokens": 8192}},
            }
        ]
        assert (
            get_custom_provider_max_tokens(
                "m", "https://example.invalid/v1", custom
            )
            == 8192
        )

    def test_returns_none_when_no_override(self):
        custom = [
            {
                "base_url": "https://other.invalid/v1",
                "models": {"m": {"max_tokens": 4096}},
            }
        ]
        assert (
            get_custom_provider_max_tokens(
                "m", "https://example.invalid/v1", custom
            )
            is None
        )

    def test_returns_none_for_zero_value(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"max_tokens": 0}},
            }
        ]
        assert (
            get_custom_provider_max_tokens(
                "m", "https://example.invalid/v1", custom
            )
            is None
        )

    def test_returns_none_for_negative_value(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"max_tokens": -100}},
            }
        ]
        assert (
            get_custom_provider_max_tokens(
                "m", "https://example.invalid/v1", custom
            )
            is None
        )

    def test_returns_none_for_string_value(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"max_tokens": "16384"}},
            }
        ]
        # int("16384") succeeds so this should work
        assert (
            get_custom_provider_max_tokens(
                "m", "https://example.invalid/v1", custom
            )
            == 16384
        )

    def test_returns_none_for_non_int_string(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"max_tokens": "16K"}},
            }
        ]
        assert (
            get_custom_provider_max_tokens(
                "m", "https://example.invalid/v1", custom
            )
            is None
        )

    def test_coexists_with_context_length(self):
        """Both fields can be present in the same model config."""
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {
                    "m": {
                        "context_length": 256_000,
                        "max_tokens": 8192,
                    }
                },
            }
        ]
        from hermes_cli.config import get_custom_provider_context_length

        assert (
            get_custom_provider_context_length("m", "https://example.invalid/v1", custom)
            == 256_000
        )
        assert (
            get_custom_provider_max_tokens("m", "https://example.invalid/v1", custom)
            == 8192
        )

    def test_first_matching_entry_wins(self):
        custom = [
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"max_tokens": 4096}},
            },
            {
                "base_url": "https://example.invalid/v1",
                "models": {"m": {"max_tokens": 8192}},
            },
        ]
        assert (
            get_custom_provider_max_tokens(
                "m", "https://example.invalid/v1", custom
            )
            == 4096
        )

    def test_none_inputs(self):
        assert get_custom_provider_max_tokens(None, "url", []) is None
        assert get_custom_provider_max_tokens("m", None, []) is None
        assert get_custom_provider_max_tokens(None, None, []) is None
