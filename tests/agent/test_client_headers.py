"""Tests for shared client header helpers."""

from agent.client_headers import get_model_custom_headers, merge_default_headers


def test_get_model_custom_headers_sanitizes_entries():
    headers = get_model_custom_headers(
        {
            "model": {
                "custom_headers": {
                    " X-Source ": " hermes-agent ",
                    "": "ignored",
                    "X-Empty": "   ",
                    "X-Number": 42,
                    7: "ignored",
                }
            }
        }
    )

    assert headers == {
        "X-Source": "hermes-agent",
        "X-Number": "42",
    }


def test_merge_default_headers_prefers_later_values():
    merged = merge_default_headers(
        {"X-Source": "base", "X-Base": "1"},
        {"X-Source": "override", "X-Custom": "2"},
    )

    assert merged == {
        "X-Source": "override",
        "X-Base": "1",
        "X-Custom": "2",
    }


def test_merge_default_headers_preserves_all_when_multiple_sets():
    """When merging config headers with provider headers, both are kept."""
    config_headers = {"X-Custom": "from-config"}
    provider_headers = {"X-Provider": "from-provider"}
    merged = merge_default_headers(config_headers, provider_headers)
    assert merged == {
        "X-Custom": "from-config",
        "X-Provider": "from-provider",
    }


def test_merge_default_headers_provider_overrides_config():
    """Provider headers override config headers when keys conflict."""
    config_headers = {"X-Source": "config-value"}
    provider_headers = {"X-Source": "provider-value"}
    merged = merge_default_headers(config_headers, provider_headers)
    assert merged == {"X-Source": "provider-value"}


def test_merge_default_headers_skips_none():
    """None inputs are skipped gracefully."""
    merged = merge_default_headers(None, {"X-Only": "one"}, None)
    assert merged == {"X-Only": "one"}


def test_merge_default_headers_empty_inputs():
    """Empty dicts are skipped gracefully."""
    merged = merge_default_headers({}, {"X-Only": "one"}, {})
    assert merged == {"X-Only": "one"}
