"""Tests for Mem0's declared config surface.

The keys are the ones ``plugins/memory/mem0`` itself reads from ``$HERMES_HOME/mem0.json`` (its
``save_config`` writes them there) plus the env keys it falls back to; a drift between the panel and
the plugin is what these assertions catch.
"""

from plugins.memory.config_schema import (
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    STORAGE_FLAT_JSON,
    get_provider_config_schema,
)

# The curated set shown in the compact panel; everything else lives in the modal.
INLINE_KEYS = {"api_key", "host", "mode", "user_id"}


def test_mem0_is_declared():
    provider = get_provider_config_schema("mem0")

    assert provider is not None
    assert provider.label == "Mem0"
    assert provider.storage == STORAGE_FLAT_JSON
    keys = [field.key for field in provider.fields]
    assert len(keys) == len(set(keys))
    assert INLINE_KEYS == {field.key for field in provider.fields if field.inline}
    # Every key the plugin reads from mem0.json / the env store is on the panel.
    assert {"mode", "host", "user_id", "agent_id", "rerank", "sync_max_chars"} <= set(keys)


def test_api_key_is_a_secret_bound_to_env():
    provider = get_provider_config_schema("mem0")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "MEM0_API_KEY"


def test_connection_and_identity_fields_carry_their_env_fallbacks():
    provider = get_provider_config_schema("mem0")
    assert provider is not None
    fields = {field.key: field for field in provider.fields}

    assert fields["host"].kind == KIND_TEXT
    assert fields["host"].env_fallbacks == ("MEM0_HOST",)
    assert fields["user_id"].env_fallbacks == ("MEM0_USER_ID",)
    assert fields["agent_id"].env_fallbacks == ("MEM0_AGENT_ID",)


def test_mode_offers_the_two_routing_values_the_plugin_understands():
    provider = get_provider_config_schema("mem0")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.default == "platform"
    # Self-hosted is platform + a host URL, not a third mode (setup writes host=... with platform).
    assert list(mode.allowed_values()) == ["platform", "oss"]


def test_message_cap_is_numeric_so_the_panel_can_raise_it():
    provider = get_provider_config_schema("mem0")
    assert provider is not None

    cap = next(field for field in provider.fields if field.key == "sync_max_chars")
    assert cap.kind == KIND_NUMBER
    # Unset in mem0.json keeps the plugin's own 450-char default.
    assert not cap.default
