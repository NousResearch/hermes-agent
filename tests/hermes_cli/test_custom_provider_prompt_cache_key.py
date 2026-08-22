from hermes_cli.config import (
    _VALID_CUSTOM_PROVIDER_FIELDS,
    _custom_provider_entry_to_provider_config,
    _normalize_custom_provider_entry,
)


def test_prompt_cache_capability_survives_normalization_and_migration():
    raw = {
        "name": "cliproxy",
        "base_url": "https://cliproxy.example/v1",
        "supports_prompt_cache_key": True,
    }

    normalized = _normalize_custom_provider_entry(raw, provider_key="cliproxy")
    migrated = _custom_provider_entry_to_provider_config(raw, provider_key="cliproxy")

    assert normalized is not None
    assert normalized["supports_prompt_cache_key"] is True
    assert migrated is not None
    assert migrated["supports_prompt_cache_key"] is True
    assert "supports_prompt_cache_key" in _VALID_CUSTOM_PROVIDER_FIELDS
