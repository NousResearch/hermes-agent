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


    def test_numeric_yaml_name_and_key_become_strings(self):
        """Unquoted YAML `name: 2070` / key 2070 must not be dropped as non-str."""
        from hermes_cli.config import find_provider_entry, stringify_provider_map

        result = _normalize_custom_provider_entry(
            {"name": 2070, "base_url": "http://192.168.1.10:8082/v1"},
            provider_key=2070,
        )
        assert result is not None
        assert result["name"] == "2070"
        assert result["provider_key"] == "2070"

        mapped = stringify_provider_map({2070: {"base_url": "http://x"}})
        assert list(mapped) == ["2070"]

        stored, entry = find_provider_entry({2070: {"base_url": "http://x"}}, "2070")
        assert stored == 2070
        assert entry == {"base_url": "http://x"}


