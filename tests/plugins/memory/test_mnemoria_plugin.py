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
        "mnemoria_write",
        "mnemoria_recall",
        "mnemoria_search",
        "mnemoria_reflect",
        "mnemoria_reward",
        "mnemoria_explore",
        "mnemoria_stats",
        "mnemoria_consolidate",
    }


def test_mnemoria_provider_returns_json_error_when_dependency_missing(monkeypatch):
    provider = MnemoriaMemoryProvider()
    monkeypatch.setattr(mnemoria_provider_module, "_UM_AVAILABLE", False)
    result = provider.handle_tool_call("mnemoria_stats", {})

    payload = json.loads(result)
    assert payload == {"error": "mnemoria package not available"}


def test_mnemoria_is_discoverable_in_memory_provider_list():
    providers = {name: (desc, available) for name, desc, available in discover_memory_providers()}

    assert "mnemoria" in providers
    desc, available = providers["mnemoria"]
    assert "Mnemoria" in desc
    assert isinstance(available, bool)


def test_initialize_sets_read_only_for_cron_context():
    provider = MnemoriaMemoryProvider()
    provider.initialize("test-session", agent_context="cron", hermes_home="/tmp")
    assert provider._read_only is True


def test_initialize_sets_read_only_for_flush_context():
    provider = MnemoriaMemoryProvider()
    provider.initialize("test-session", agent_context="flush", hermes_home="/tmp")
    assert provider._read_only is True


def test_initialize_not_read_only_for_primary_context():
    provider = MnemoriaMemoryProvider()
    provider.initialize("test-session", agent_context="primary", hermes_home="/tmp")
    assert provider._read_only is False


def test_system_prompt_block_includes_usage_hint():
    provider = MnemoriaMemoryProvider()
    block = provider.system_prompt_block()
    assert "[MNEMORIA MEMORY]" in block
    assert "mnemoria_write" in block
    assert "mnemoria_recall" in block


def test_initialize_stores_profile_and_user_id():
    provider = MnemoriaMemoryProvider()
    provider.initialize("test-session", agent_identity="coder", user_id="user-abc", platform="telegram", hermes_home="/tmp")
    assert provider._profile == "coder"
    assert provider._user_id == "user-abc"
    assert provider._platform == "telegram"
