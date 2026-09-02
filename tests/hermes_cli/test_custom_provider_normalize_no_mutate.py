"""Regression: _normalize_custom_provider_entry must never mutate its input.

Provider entries frequently alias cached config dicts (load_config_readonly()
shares one dict process-wide). The normalizer's alias rewrites
(api_key_env -> key_env, camelCase -> snake_case) used to write INTO the
caller's entry, corrupting the shared cache for every subsequent reader.
"""

import copy

from hermes_cli.config import (
    _normalize_custom_provider_entry,
    get_compatible_custom_providers,
    providers_dict_to_custom_providers,
)


def test_normalizer_does_not_mutate_entry_api_key_env():
    entry = {"base_url": "https://x.example/v1", "api_key_env": "MY_KEY"}
    snapshot = copy.deepcopy(entry)
    out = _normalize_custom_provider_entry(entry, provider_key="p")
    assert entry == snapshot, "input entry must not gain key_env"
    assert out is not None and out.get("key_env") == "MY_KEY"


def test_normalizer_does_not_mutate_entry_camel_aliases():
    entry = {"baseUrl": "https://x.example/v1", "apiKey": "sk-test-not-real"}
    snapshot = copy.deepcopy(entry)
    _normalize_custom_provider_entry(entry, provider_key="p")
    assert entry == snapshot, "camelCase aliasing must not write into the input"


def test_providers_dict_roundtrip_leaves_cached_config_untouched():
    config = {
        "providers": {
            "myprov": {
                "base_url": "https://x.example/v1",
                "api_key_env": "MY_KEY",
                "models": {"m1": {"context_length": 8192}},
            }
        }
    }
    snapshot = copy.deepcopy(config)
    providers_dict_to_custom_providers(config["providers"])
    assert config == snapshot

    get_compatible_custom_providers(config)
    assert config == snapshot


def test_bare_base_url_warns_about_missing_v1_prefix(caplog):
    """A scheme+host-only base_url is the silent-404 shape: OpenAI-compatible
    clients append /chat/completions to base_url, so the request misses the
    server's /v1/chat/completions (#89334). The normalizer must say so once."""
    import logging

    import hermes_cli.config as config_mod

    entry = {"base_url": "https://api.example.com", "key_env": "MY_KEY"}
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        out = config_mod._normalize_custom_provider_entry(entry, provider_key="dots")
    assert out is not None
    assert any(
        "has no path" in r.getMessage() for r in caplog.records
    ), "bare base_url must warn about the missing /v1 prefix"


def test_base_url_with_path_does_not_warn(caplog):
    import logging

    import hermes_cli.config as config_mod

    entry = {"base_url": "https://api.example.com/v1", "key_env": "MY_KEY"}
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        out = config_mod._normalize_custom_provider_entry(entry, provider_key="dots")
    assert out is not None
    assert not any(
        "has no path" in r.getMessage() for r in caplog.records
    )


def test_bare_base_url_placeholder_does_not_warn(caplog):
    """Env-ref/template URLs are expanded at runtime; the path may come from
    the expansion, so the bare check must not fire on them."""
    import logging

    import hermes_cli.config as config_mod

    entry = {"base_url": "https://api-{region}.example.com", "key_env": "MY_KEY"}
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        out = config_mod._normalize_custom_provider_entry(entry, provider_key="dots")
    assert out is not None
    assert not any(
        "has no path" in r.getMessage() for r in caplog.records
    )


def test_normalized_models_mapping_is_not_shared_with_input():
    entry = {
        "base_url": "https://x.example/v1",
        "api_key": "sk-test-not-real",
        "models": {"m1": {}},
    }
    out = _normalize_custom_provider_entry(entry, provider_key="p")
    assert out is not None
    out["models"]["injected"] = {}
    assert "injected" not in entry["models"], (
        "normalized models mapping must not alias the caller's dict"
    )
