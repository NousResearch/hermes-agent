"""Tests for the declarative memory-provider registry."""

from hermes_cli.memory_providers import (
    KIND_BOOL,
    KIND_JSON,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    MEMORY_PROVIDERS,
    STORAGE_HONCHO_HOST_BLOCK,
    get_memory_provider,
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


def test_registry_lists_honcho_before_hindsight():
    assert list(MEMORY_PROVIDERS) == ["honcho", "hindsight"]


def test_honcho_is_declared():
    provider = get_memory_provider("honcho")

    assert provider is not None
    assert provider.label == "Honcho"
    assert provider.storage == STORAGE_HONCHO_HOST_BLOCK
    # Field keys are unique, and the curated inline set is present.
    keys = [field.key for field in provider.fields]
    assert len(keys) == len(set(keys))
    assert INLINE_KEYS <= set(keys)


def test_honcho_inline_fields_are_the_curated_subset():
    provider = get_memory_provider("honcho")
    assert provider is not None

    assert {field.key for field in provider.inline_fields()} == INLINE_KEYS
    # The modal-only fields are a non-empty remainder.
    non_inline = {f.key for f in provider.fields} - INLINE_KEYS
    assert {"writeFrequency", "recallMode", "userPeerAliases"} <= non_inline


def test_honcho_declares_the_new_field_kinds():
    provider = get_memory_provider("honcho")
    assert provider is not None

    by_key = {f.key: f for f in provider.fields}
    assert by_key["saveMessages"].kind == KIND_BOOL
    assert by_key["dialecticMaxChars"].kind == KIND_NUMBER
    assert by_key["userPeerAliases"].kind == KIND_JSON
    assert by_key["recallMode"].allowed_values() == {"hybrid", "context", "tools"}
    assert by_key["observationMode"].allowed_values() == {"directional", "unified"}


def test_honcho_selects_constrain_their_values():
    provider = get_memory_provider("honcho")
    assert provider is not None

    environment = next(f for f in provider.fields if f.key == "environment")
    assert environment.kind == KIND_SELECT
    # Honcho SDK only accepts local/production; "demo" is not a valid environment.
    assert environment.allowed_values() == {"production", "local"}

    strategy = next(f for f in provider.fields if f.key == "sessionStrategy")
    assert strategy.allowed_values() == {"per-directory", "per-repo", "per-session", "global"}


def test_honcho_api_key_is_a_secret_bound_to_env():
    provider = get_memory_provider("honcho")
    assert provider is not None

    api_key = next(f for f in provider.fields if f.key == "apiKey")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HONCHO_API_KEY"


def test_honcho_root_scoped_fields_are_exactly_the_global_ones():
    provider = get_memory_provider("honcho")
    assert provider is not None

    scopes = {f.key: f.scope for f in provider.fields}
    root_keys = {k for k, scope in scopes.items() if scope == "root"}
    # baseUrl, timeout and sessions live at the config root in Honcho's schema;
    # everything else is per-profile host-scoped.
    assert root_keys == {"baseUrl", "timeout", "sessions"}


def test_hindsight_is_declared():
    provider = get_memory_provider("hindsight")

    assert provider is not None
    assert provider.label == "Hindsight"
    assert {field.key for field in provider.fields} == {
        "mode",
        "api_key",
        "api_url",
        "bank_id",
        "recall_budget",
    }


def test_hindsight_fields_are_all_inline():
    provider = get_memory_provider("hindsight")
    assert provider is not None

    # Hindsight is simple enough to render fully in the compact panel, so it
    # never grows a Full config… modal.
    assert all(field.inline for field in provider.fields)


def test_hindsight_mode_gating_is_expressed_as_select_options():
    provider = get_memory_provider("hindsight")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.allowed_values() == {"cloud", "local_external"}
    # local_embedded is intentionally unsupported on desktop.
    assert "local_embedded" not in mode.allowed_values()


def test_hindsight_api_key_is_a_secret_bound_to_env():
    provider = get_memory_provider("hindsight")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HINDSIGHT_API_KEY"


def test_unknown_provider_is_none():
    assert get_memory_provider("builtin") is None
