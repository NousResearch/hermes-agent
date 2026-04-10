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


def test_prefetch_returns_cached_result_from_queue(monkeypatch):
    provider = MnemoriaMemoryProvider()
    from unittest.mock import MagicMock
    mock_fact = MagicMock()
    mock_fact.fact.fact_type = "V"
    mock_fact.fact.target = "test"
    mock_fact.fact.content = "cached content"
    provider._prefetch_result = [mock_fact]
    result = provider.prefetch("any query")
    assert "cached content" in result
    assert provider._prefetch_result is None


def test_get_config_schema_returns_valid_fields():
    provider = MnemoriaMemoryProvider()
    schema = provider.get_config_schema()
    assert isinstance(schema, list)
    assert len(schema) >= 1
    keys = {field["key"] for field in schema}
    assert "db_path" in keys


def test_save_config_writes_json(tmp_path):
    provider = MnemoriaMemoryProvider()
    provider.save_config({"db_path": "/custom/path.db"}, str(tmp_path))
    config_path = tmp_path / "mnemoria.json"
    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["db_path"] == "/custom/path.db"


def test_save_config_merges_with_existing(tmp_path):
    config_path = tmp_path / "mnemoria.json"
    config_path.write_text(json.dumps({"existing_key": "value"}))
    provider = MnemoriaMemoryProvider()
    provider.save_config({"db_path": "/new/path.db"}, str(tmp_path))
    data = json.loads(config_path.read_text())
    assert data["existing_key"] == "value"
    assert data["db_path"] == "/new/path.db"


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


# Task 8: on_memory_write

def test_on_memory_write_is_noop_when_read_only():
    provider = MnemoriaMemoryProvider()
    provider._read_only = True
    provider.on_memory_write("add", "user", "some content")


def test_on_memory_write_skips_remove_action():
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    provider.on_memory_write("remove", "user", "some content")


# Task 9: on_delegation

def test_on_delegation_is_noop_when_read_only():
    provider = MnemoriaMemoryProvider()
    provider._read_only = True
    provider.on_delegation("do research", "found nothing", child_session_id="child-1")


def test_on_delegation_does_not_raise_without_store(monkeypatch):
    monkeypatch.setattr(mnemoria_provider_module, "_UM_AVAILABLE", False)
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    provider.on_delegation("do research", "found nothing", child_session_id="child-1")


# Task 10: on_pre_compress

def test_on_pre_compress_is_noop_when_read_only():
    provider = MnemoriaMemoryProvider()
    provider._read_only = True
    result = provider.on_pre_compress([{"role": "tool", "content": "Error: broke"}])
    assert result == ""


def test_on_pre_compress_returns_empty_string():
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    assert provider.on_pre_compress([]) == ""


def test_on_pre_compress_advances_message_index():
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    provider._last_extracted_msg_index = 0
    provider.on_pre_compress([{"role": "user", "content": "hello"}, {"role": "tool", "content": "ok"}])
    assert provider._last_extracted_msg_index == 2


# Task 11: on_session_end

def test_on_session_end_does_not_raise_without_store(monkeypatch):
    monkeypatch.setattr(mnemoria_provider_module, "_UM_AVAILABLE", False)
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    provider.on_session_end([{"role": "user", "content": "bye"}])
