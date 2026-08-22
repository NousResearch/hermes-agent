"""Tests for providers config entry validation and normalization.

Covers Issue #9332: camelCase keys silently ignored, non-URL strings
accepted as base_url, and unknown keys go unreported.
"""

import logging

import pytest

from hermes_cli.config import (
    _PROVIDER_NORMALIZE_WARNED,
    _normalize_custom_provider_entry,
)


class TestNormalizeCustomProviderEntry:
    """Tests for _normalize_custom_provider_entry validation."""

    @pytest.fixture(autouse=True)
    def _reset_warn_cache(self):
        """The normalizer deduplicates its warnings via a process-lifetime
        cache; clear it around each test so warning assertions are independent
        of test order."""
        _PROVIDER_NORMALIZE_WARNED.clear()
        yield
        _PROVIDER_NORMALIZE_WARNED.clear()





    def test_unknown_keys_warned_once_per_signature(self, caplog):
        """Repeated normalization of the same entry (as happens on every
        picker/inventory load) must warn only once — otherwise the warning
        storms the log handler. Fix B."""
        entry = {
            "base_url": "https://api.example.com/v1",
            "api_key": "***",
            "unknownField": "value",
        }
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                _normalize_custom_provider_entry(
                    dict(entry), provider_key="test"
                )
        unknown_warnings = [
            r for r in caplog.records
            if "unknown config keys" in r.message.lower()
        ]
        assert len(unknown_warnings) == 1




    def test_env_var_placeholder_in_base_url_not_rejected(self):
        """A base_url that is an un-expanded ${ENV_VAR} placeholder must not be
        rejected as an invalid URL — it is expanded at runtime, so a caller
        reaching this normalizer with raw config would otherwise see the
        provider silently dropped. Regression test for #14457."""
        entry = {
            "name": "PROVIDER_A",
            "base_url": "${PROVIDER_A_BASE_URL}",
            "key_env": "PROVIDER_A_API_KEY",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="PROVIDER_A")
        assert result is not None
        assert result["base_url"] == "${PROVIDER_A_BASE_URL}"

    @pytest.mark.parametrize("field", ["max_output_tokens", "context_length", "rate_limit_delay"])
    def test_a_yaml_true_is_not_taken_as_a_number(self, field):
        """bool is an int subclass, so ``field: true`` used to persist as 1.

        A token ceiling of 1 from what reads like a feature-flag typo is
        worse than the field being absent — the provider answers with one
        token and nothing explains why.
        """
        entry = {"name": "local", "base_url": "http://x/v1", field: True}
        result = _normalize_custom_provider_entry(entry, provider_key="local")
        assert field not in result, (
            f"{field}: true was accepted as a number ({result.get(field)!r})"
        )

    @pytest.mark.parametrize(
        "field,good",
        [("max_output_tokens", 8192), ("context_length", 32000), ("rate_limit_delay", 0.5)],
    )
    def test_real_numbers_still_pass(self, field, good):
        entry = {"name": "local", "base_url": "http://x/v1", field: good}
        assert _normalize_custom_provider_entry(entry, provider_key="local")[field] == good

    def test_max_output_tokens_preserved_without_warning(self, caplog):
        """max_output_tokens and max_tokens alias must be preserved in normalized
        entry without triggering unknown key warning (#88997)."""
        entry = {
            "name": "ollama-local",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": "gpt-oss:20b-64k",
            "max_output_tokens": 8192,
        }
        with caplog.at_level(logging.WARNING):
            result = _normalize_custom_provider_entry(entry, provider_key="ollama-local")
        assert result is not None
        assert result["max_output_tokens"] == 8192
        assert not [r for r in caplog.records if "unknown config keys" in r.message.lower()]

    def test_max_tokens_alias_and_camelcase_preserved(self, caplog):
        """max_tokens and camelCase maxOutputTokens/maxTokens are preserved."""
        entry1 = {
            "name": "vllm-local",
            "base_url": "http://localhost:8000/v1",
            "api_key": "vllm",
            "max_tokens": 4096,
        }
        result1 = _normalize_custom_provider_entry(entry1, provider_key="vllm-local")
        assert result1 is not None
        assert result1["max_output_tokens"] == 4096

        entry2 = {
            "name": "lmstudio-local",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lmstudio",
            "maxOutputTokens": 2048,
        }
        result2 = _normalize_custom_provider_entry(entry2, provider_key="lmstudio-local")
        assert result2 is not None
        assert result2["max_output_tokens"] == 2048
