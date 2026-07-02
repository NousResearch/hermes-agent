"""Tests for config-schema loading from memory provider plugin dirs."""

from plugins.memory.config_schema import get_provider_config_schema


def test_unknown_provider_is_none():
    assert get_provider_config_schema("builtin") is None


def test_plugin_without_schema_is_none():
    # mem0 is a real plugin dir that declares no config_schema.py.
    assert get_provider_config_schema("mem0") is None


def test_schemas_are_cached_per_provider():
    assert get_provider_config_schema("honcho") is get_provider_config_schema("honcho")
