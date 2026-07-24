"""Tests for the per-provider ``needs_reasoning_content`` config knob.

These only cover config parsing and provider-matching by base_url. For the
test that actually proves reasoning_content survives onto the outgoing
request, see tests/run_agent/test_custom_provider_reasoning_content_echo.py.
"""

from hermes_cli.config import (
    _normalize_custom_provider_entry,
    get_custom_provider_needs_reasoning_content,
)


def test_normalize_entry_reads_needs_reasoning_content():
    """needs_reasoning_content: true survives config normalization."""
    normalized = _normalize_custom_provider_entry({
        "name": "local-llamacpp",
        "base_url": "http://localhost:8080/v1",
        "needs_reasoning_content": True,
    })
    assert normalized["needs_reasoning_content"] is True


def test_normalize_entry_omits_field_when_not_set_in_yaml():
    """No key in YAML means no key in the normalized entry (not False)."""
    normalized = _normalize_custom_provider_entry({
        "name": "local-llamacpp",
        "base_url": "http://localhost:8080/v1",
    })
    assert "needs_reasoning_content" not in normalized


def test_normalize_entry_ignores_non_boolean_values():
    """Only real YAML booleans count; junk values are dropped rather than
    silently coerced (mirrors discover_models handling)."""
    normalized = _normalize_custom_provider_entry({
        "name": "local-llamacpp",
        "base_url": "http://localhost:8080/v1",
        "needs_reasoning_content": "yes",
    })
    assert "needs_reasoning_content" not in normalized


def test_lookup_matches_provider_by_base_url():
    """The runtime lookup helper finds the flag by matching base_url."""
    providers = [{
        "name": "local-llamacpp",
        "base_url": "http://localhost:8080/v1",
        "needs_reasoning_content": True,
    }]
    assert get_custom_provider_needs_reasoning_content(
        "http://localhost:8080/v1", custom_providers=providers,
    ) is True


def test_lookup_returns_false_for_a_different_provider():
    """A base_url with no matching entry defaults to False, not an error."""
    providers = [{
        "name": "local-llamacpp",
        "base_url": "http://localhost:8080/v1",
        "needs_reasoning_content": True,
    }]
    assert get_custom_provider_needs_reasoning_content(
        "http://other-host:8080/v1", custom_providers=providers,
    ) is False
