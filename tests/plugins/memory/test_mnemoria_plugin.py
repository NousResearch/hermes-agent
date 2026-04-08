import json

from plugins.memory import discover_memory_providers, load_memory_provider
from plugins.memory.mnemoria import provider as mnemoria_provider_module
from plugins.memory.mnemoria.provider import MnemoriaMemoryProvider


def test_mnemoria_provider_loads_even_when_dependency_missing():
    provider = load_memory_provider("mnemoria")
    assert provider is not None
    assert isinstance(provider, MnemoriaMemoryProvider)
    assert provider.name == "mnemoria"


def test_mnemoria_provider_exposes_expected_tool_schemas():
    provider = MnemoriaMemoryProvider()
    schemas = provider.get_tool_schemas()

    assert len(schemas) == 8
    assert {schema["name"] for schema in schemas} == {
        "mcp_umemory_write",
        "mcp_umemory_recall",
        "mcp_umemory_search",
        "mcp_umemory_reflect",
        "mcp_umemory_reward",
        "mcp_umemory_explore",
        "mcp_umemory_stats",
        "mcp_umemory_consolidate",
    }


def test_mnemoria_provider_returns_json_error_when_dependency_missing(monkeypatch):
    provider = MnemoriaMemoryProvider()
    monkeypatch.setattr(mnemoria_provider_module, "_UM_AVAILABLE", False)
    result = provider.handle_tool_call("mcp_umemory_stats", {})

    payload = json.loads(result)
    assert payload == {"error": "mnemoria package not available"}


def test_mnemoria_is_discoverable_in_memory_provider_list():
    providers = {name: (desc, available) for name, desc, available in discover_memory_providers()}

    assert "mnemoria" in providers
    desc, available = providers["mnemoria"]
    assert "Mnemoria cognitive memory system" in desc
    assert isinstance(available, bool)
