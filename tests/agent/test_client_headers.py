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
