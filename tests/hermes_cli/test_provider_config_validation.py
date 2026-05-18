"""Tests for providers config entry validation and normalization.

Covers Issue #9332: camelCase keys silently ignored, non-URL strings
accepted as base_url, and unknown keys go unreported.
"""

import logging
from unittest.mock import patch

import pytest

from hermes_cli.config import _normalize_custom_provider_entry


class TestNormalizeCustomProviderEntry:
    """Tests for _normalize_custom_provider_entry validation."""

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

    def test_camel_case_api_key_mapped(self):
        """camelCase apiKey should be auto-mapped to api_key."""
        entry = {
            "base_url": "https://api.example.com/v1",
            "apiKey": "sk-test-key",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="myhost")
        assert result is not None
        assert result["api_key"] == "sk-test-key"

    def test_camel_case_base_url_mapped(self):
        """camelCase baseUrl should be auto-mapped to base_url."""
        entry = {
            "baseUrl": "https://api.example.com/v1",
            "api_key": "sk-test-key",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="myhost")
        assert result is not None
        assert result["base_url"] == "https://api.example.com/v1"

    def test_non_url_api_field_rejected(self):
        """Non-URL string in 'api' field should be skipped with a warning."""
        entry = {
            "api": "openai-reverse-proxy",
            "api_key": "sk-test-key",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="nvidia")
        # Should return None because no valid URL was found
        assert result is None

    def test_valid_url_in_api_field_accepted(self):
        """Valid URL in 'api' field should still be accepted."""
        entry = {
            "api": "https://integrate.api.nvidia.com/v1",
            "api_key": "sk-test-key",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="nvidia")
        assert result is not None
        assert result["base_url"] == "https://integrate.api.nvidia.com/v1"

    def test_base_url_preferred_over_api(self):
        """base_url should be checked before api field."""
        entry = {
            "base_url": "https://correct.example.com/v1",
            "api": "https://wrong.example.com/v1",
            "api_key": "sk-test-key",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="test")
        assert result is not None
        assert result["base_url"] == "https://correct.example.com/v1"

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

    def test_timeout_keys_not_flagged_unknown(self, caplog):
        """request_timeout_seconds and stale_timeout_seconds should not produce warnings."""
        entry = {
            "base_url": "https://api.example.com/v1",
            "api_key": "***",
            "request_timeout_seconds": 300,
            "stale_timeout_seconds": 900,
        }
        with caplog.at_level(logging.WARNING):
            result = _normalize_custom_provider_entry(entry, provider_key="test")
        assert result is not None
        assert not any("unknown config keys" in r.message.lower() for r in caplog.records)

    def test_camel_case_warning_logged(self, caplog):
        """camelCase alias mapping should produce a warning."""
        entry = {
            "baseUrl": "https://api.example.com/v1",
            "apiKey": "sk-test-key",
        }
        with caplog.at_level(logging.WARNING):
            result = _normalize_custom_provider_entry(entry, provider_key="test")
        assert result is not None
        camel_warnings = [r for r in caplog.records if "camelcase" in r.message.lower() or "auto-mapped" in r.message.lower()]
        assert len(camel_warnings) >= 1

    def test_snake_case_takes_precedence_over_camel(self):
        """If both snake_case and camelCase exist, snake_case wins."""
        entry = {
            "api_key": "snake-key",
            "apiKey": "camel-key",
            "base_url": "https://api.example.com/v1",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="test")
        assert result is not None
        assert result["api_key"] == "snake-key"

    def test_non_dict_returns_none(self):
        """Non-dict entry should return None."""
        assert _normalize_custom_provider_entry("not-a-dict") is None
        assert _normalize_custom_provider_entry(42) is None
        assert _normalize_custom_provider_entry(None) is None

    def test_no_url_returns_none(self):
        """Entry with no valid URL in any field should return None."""
        entry = {
            "api_key": "sk-test-key",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="test")
        assert result is None

    def test_no_name_returns_none(self):
        """Entry with no name and no provider_key should return None."""
        entry = {
            "base_url": "https://api.example.com/v1",
        }
        result = _normalize_custom_provider_entry(entry, provider_key="")
        assert result is None

    def test_models_list_converted_to_dict(self):
        """List-format models should be preserved as an empty-value dict so
        /model picks them up instead of showing the provider with (0) models."""
        entry = {
            "name": "tencent-coding-plan",
            "base_url": "https://api.lkeap.cloud.tencent.com/coding/v3",
            "models": ["glm-5", "kimi-k2.5", "minimax-m2.5"],
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert result["models"] == {"glm-5": {}, "kimi-k2.5": {}, "minimax-m2.5": {}}

    def test_models_dict_preserved(self):
        """Dict-format models should pass through unchanged."""
        entry = {
            "name": "acme",
            "base_url": "https://api.example.com/v1",
            "models": {"gpt-foo": {"context_length": 32000}},
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert result["models"] == {"gpt-foo": {"context_length": 32000}}

    def test_models_list_filters_empty_and_non_string(self):
        """List entries that are empty strings or non-strings are skipped."""
        entry = {
            "name": "acme",
            "base_url": "https://api.example.com/v1",
            "models": ["valid", "", None, 42, "  ", "also-valid"],
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert result["models"] == {"valid": {}, "also-valid": {}}

    def test_models_empty_list_omitted(self):
        """Empty list (falsy) should not produce a models key."""
        entry = {
            "name": "acme",
            "base_url": "https://api.example.com/v1",
            "models": [],
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert "models" not in result

    def test_id_preserved_for_stable_slug_lookup(self):
        """The ``id`` field must be forwarded so renaming ``name`` doesn't
        break ``model.provider: custom:<id>`` resolution downstream.

        The runtime resolver (``_get_named_custom_provider`` in
        ``hermes_cli/runtime_provider.py``) matches the requested slug
        against an id-derived slug, but that lookup only sees what this
        normalizer forwards. Previously ``id`` was not in ``_KNOWN_KEYS``
        and never copied into the normalized result, so a cosmetic
        ``name`` change silently invalidated every persisted
        ``custom:<old-slug>`` reference (main model, gateway, auxiliary
        tasks, cron jobs).
        """
        entry = {
            "id": "amd-llm-gateway",
            "name": "AMD LLM Gateway (Anthropic)",
            "base_url": "https://gateway.example.com:8443/Anthropic",
            "api_key": "sk-test",
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert result["id"] == "amd-llm-gateway"
        # name and id are independent — name follows the cosmetic label,
        # id stays put across renames.
        assert result["name"] == "AMD LLM Gateway (Anthropic)"

    def test_id_stripped_of_surrounding_whitespace(self):
        """``id`` values are stripped to match the same normalization the
        runtime resolver applies before slug comparison."""
        entry = {
            "id": "  my-stable-slug  ",
            "name": "My Gateway",
            "base_url": "https://api.example.com/v1",
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert result["id"] == "my-stable-slug"

    def test_id_omitted_when_missing_or_empty(self):
        """Entries without an ``id`` (or with empty/whitespace-only ``id``)
        must not synthesize one — leaving the field absent preserves the
        legacy "match by name only" behaviour for configs that never opted
        into the stable-slug pattern."""
        for missing_id in (None, "", "   "):
            entry = {
                "name": "Cosmetic Only",
                "base_url": "https://api.example.com/v1",
            }
            if missing_id is not None:
                entry["id"] = missing_id
            result = _normalize_custom_provider_entry(entry)
            assert result is not None
            assert "id" not in result, (
                f"id should be absent when value is {missing_id!r}"
            )

    def test_id_non_string_ignored(self):
        """Non-string ``id`` values are silently ignored rather than
        coerced — matches the defensive style of the other field
        normalizers."""
        for bad_id in (123, ["list"], {"dict": 1}, True):
            entry = {
                "id": bad_id,
                "name": "Bad ID Type",
                "base_url": "https://api.example.com/v1",
            }
            result = _normalize_custom_provider_entry(entry)
            assert result is not None
            assert "id" not in result, (
                f"id should be absent for non-string value {bad_id!r}"
            )

    def test_id_is_a_known_key_not_warned_as_unknown(self, caplog):
        """``id`` belongs in ``_KNOWN_KEYS`` so configs that set it
        don't trigger spurious ``unknown config keys ignored`` warnings."""
        import logging
        entry = {
            "id": "stable-slug",
            "name": "Some Provider",
            "base_url": "https://api.example.com/v1",
        }
        with caplog.at_level(logging.WARNING):
            _normalize_custom_provider_entry(entry, provider_key="acme")
        assert not any(
            "unknown config keys" in rec.message and "id" in rec.message
            for rec in caplog.records
        ), "id field should not trigger an unknown-keys warning"
