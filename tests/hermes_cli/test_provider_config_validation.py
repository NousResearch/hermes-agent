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

    def test_valid_entry_snake_case(self):
        """Standard snake_case entry should normalize correctly."""
        entry = {
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-test-key",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="myhost")
        assert result is not None
        assert result["name"] == "myhost"
        assert result["base_url"] == "https://api.example.com/v1"
        assert result["api_key"] == "sk-test-key"


    def test_unknown_keys_logged(self, caplog):
        """Unknown config keys should produce a warning."""
        entry = {
            "base_url": "https://api.example.com/v1",
            "api_key": "***",
            "unknownField": "value",
            "anotherBad": 42,
        }
        with caplog.at_level(logging.WARNING):
            result = _normalize_custom_provider_entry(entry, provider_key="test")
        assert result is not None
        assert any("unknown config keys" in r.message.lower() for r in caplog.records)

    def test_provider_key_not_flagged_unknown(self, caplog):
        """A redundant ``provider`` key (written by Hermes' own config writer)
        must be accepted silently — not reported as an unknown key. Regression
        for the config warn-storm that deadlocked Windows logging."""
        entry = {
            "provider": "",
            "base_url": "https://api.example.com/v1",
            "api_key": "***",
        }
        with caplog.at_level(logging.WARNING):
            result = _normalize_custom_provider_entry(entry, provider_key="onyx-6000")
        assert result is not None
        assert not any("unknown config keys" in r.message.lower() for r in caplog.records)

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


    def test_non_dict_returns_none(self):
        """Non-dict entry should return None."""
        assert _normalize_custom_provider_entry("not-a-dict") is None
        assert _normalize_custom_provider_entry(42) is None
        assert _normalize_custom_provider_entry(None) is None


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


    def test_invalid_url_without_placeholder_still_rejected(self):
        """A malformed URL with no scheme/host AND no placeholder token is
        still rejected — the placeholder bypass must not weaken validation of
        ordinary literal URLs."""
        entry = {
            "name": "bad",
            "base_url": "not-a-url",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="bad")
        assert result is None
