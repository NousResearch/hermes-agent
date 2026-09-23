"""Tests for Honcho's declared config surface."""

from plugins.memory.config_schema import (
    KIND_SECRET,
    STORAGE_HONCHO_HOST_BLOCK,
    get_provider_config_schema,
)

# The curated set shown in the compact panel; everything else lives in the modal.
INLINE_KEYS = {
    "apiKey",
    "baseUrl",
    "environment",
    "workspace",
    "peerName",
    "aiPeer",
    "sessionStrategy",
}


def test_honcho_is_declared():
    provider = get_provider_config_schema("honcho")

    assert provider is not None
    assert provider.label == "Honcho"
    assert provider.storage == STORAGE_HONCHO_HOST_BLOCK
    # Field keys are unique, and the curated inline set is present.
    keys = [field.key for field in provider.fields]
    assert len(keys) == len(set(keys))
    assert INLINE_KEYS <= set(keys)








def test_api_key_is_a_secret_bound_to_env():
    provider = get_provider_config_schema("honcho")
    assert provider is not None

    api_key = next(f for f in provider.fields if f.key == "apiKey")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HONCHO_API_KEY"


def test_root_scoped_fields_are_exactly_the_global_ones():
    provider = get_provider_config_schema("honcho")
    assert provider is not None

    scopes = {f.key: f.scope for f in provider.fields}
    root_keys = {k for k, scope in scopes.items() if scope == "root"}
    # baseUrl, timeout and sessions live at the config root in Honcho's schema;
    # everything else is per-profile host-scoped.
    assert root_keys == {"baseUrl", "timeout", "sessions"}
