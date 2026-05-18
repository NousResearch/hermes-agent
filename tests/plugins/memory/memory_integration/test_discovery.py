from __future__ import annotations

from plugins.memory import discover_memory_providers


def test_provider_is_discoverable_and_loadable(load_provider):
    discovered = {name: (description, available) for name, description, available in discover_memory_providers()}

    assert "memory-integration" in discovered
    assert "Vault" in discovered["memory-integration"][0]

    provider = load_provider()
    assert provider is not None
    assert provider.name == "memory-integration"


def test_status_tool_schema_available_before_initialize(load_provider):
    provider = load_provider()

    schemas = provider.get_tool_schemas()

    assert [schema["name"] for schema in schemas] == ["memory_integration_status"]
    assert schemas[0]["parameters"] == {"type": "object", "properties": {}, "required": []}
