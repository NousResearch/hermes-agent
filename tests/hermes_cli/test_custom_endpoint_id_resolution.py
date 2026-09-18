"""Regression tests for custom endpoint identifier resolution."""

from hermes_cli.web_routers.config_env import _resolve_custom_endpoint_entry


def test_custom_endpoint_resolution_prefers_stored_key_with_punctuation():
    providers = {
        "local-127.0.0.1:8283": {
            "name": "Local",
            "base_url": "http://127.0.0.1:8283/v1",
            "model": "local-model",
        }
    }

    stored_key, entry = _resolve_custom_endpoint_entry(
        providers, "local-127.0.0.1:8283"
    )

    assert stored_key == "local-127.0.0.1:8283"
    assert entry["name"] == "Local"


def test_custom_endpoint_resolution_keeps_slug_compatibility():
    providers = {
        "local-127-0-0-1-8283": {
            "name": "Local",
            "base_url": "http://127.0.0.1:8283/v1",
            "model": "local-model",
        }
    }

    stored_key, entry = _resolve_custom_endpoint_entry(
        providers, "local-127.0.0.1:8283"
    )

    assert stored_key == "local-127-0-0-1-8283"
    assert entry["name"] == "Local"


def test_custom_endpoint_resolution_returns_none_for_unknown_id():
    stored_key, entry = _resolve_custom_endpoint_entry(
        {"existing": {"name": "Existing"}}, "missing"
    )

    assert stored_key is None
    assert entry is None
