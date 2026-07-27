"""Tests for Honcho's declared config surface."""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_JSON,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
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


def test_inline_fields_are_the_curated_subset():
    provider = get_provider_config_schema("honcho")
    assert provider is not None

    assert {field.key for field in provider.inline_fields()} == INLINE_KEYS
    # The modal-only fields are a non-empty remainder.
    non_inline = {f.key for f in provider.fields} - INLINE_KEYS
    assert {"writeFrequency", "recallMode", "userPeerAliases"} <= non_inline


def test_declares_the_new_field_kinds():
    provider = get_provider_config_schema("honcho")
    assert provider is not None

    by_key = {f.key: f for f in provider.fields}
    assert by_key["saveMessages"].kind == KIND_BOOL
    assert by_key["dialecticMaxChars"].kind == KIND_NUMBER
    assert by_key["userPeerAliases"].kind == KIND_JSON
    assert by_key["recallMode"].allowed_values() == {"hybrid", "context", "tools"}
    assert by_key["observationMode"].allowed_values() == {"directional", "unified"}


def test_selects_constrain_their_values():
    provider = get_provider_config_schema("honcho")
    assert provider is not None

    environment = next(f for f in provider.fields if f.key == "environment")
    assert environment.kind == KIND_SELECT
    # Honcho SDK only accepts local/production; "demo" is not a valid environment.
    assert environment.allowed_values() == {"production", "local"}

    strategy = next(f for f in provider.fields if f.key == "sessionStrategy")
    assert strategy.allowed_values() == {"per-directory", "per-repo", "per-session", "global"}


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
