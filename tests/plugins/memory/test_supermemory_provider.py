import json
import threading

import pytest

from plugins.memory.supermemory import (
    SupermemoryMemoryProvider,
    _clean_text_for_capture,
    _format_prefetch_context,
    _load_supermemory_config,
    _save_supermemory_config,
)


class FakeClient:
    def __init__(self, api_key: str, timeout: float, container_tag: str, search_mode: str = "hybrid"):
        self.api_key = api_key
        self.timeout = timeout
        self.container_tag = container_tag
        self.search_mode = search_mode
        self.add_calls = []
        self.search_results = []
        self.profile_response = {"static": [], "dynamic": [], "search_results": []}
        self.ingest_calls = []
        self.forgotten_ids = []
        self.forget_by_query_response = {"success": True, "message": "Forgot"}

    def add_memory(self, content, metadata=None, *, entity_context="",
                   container_tag=None, custom_id=None):
        self.add_calls.append({
            "content": content,
            "metadata": metadata,
            "entity_context": entity_context,
            "container_tag": container_tag,
            "custom_id": custom_id,
        })
        return {"id": "mem_123"}

    def search_memories(self, query, *, limit=5, container_tag=None, search_mode=None):
        return self.search_results

    def get_profile(self, query=None, *, container_tag=None):
        return self.profile_response

    def forget_memory(self, memory_id, *, container_tag=None):
        self.forgotten_ids.append(memory_id)

    def forget_by_query(self, query, *, container_tag=None):
        return self.forget_by_query_response

    def ingest_conversation(self, session_id, messages):
        self.ingest_calls.append({"session_id": session_id, "messages": messages})


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    p = SupermemoryMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    return p


def test_is_available_false_without_api_key(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    p = SupermemoryMemoryProvider()
    assert p.is_available() is False


def test_is_available_false_when_import_missing(monkeypatch):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "supermemory":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = SupermemoryMemoryProvider()
    assert p.is_available() is False


def test_load_and_save_config_round_trip(tmp_path):
    _save_supermemory_config({"container_tag": "demo-tag", "auto_capture": False}, str(tmp_path))
    cfg = _load_supermemory_config(str(tmp_path))
    # container_tag is kept raw — sanitization happens in initialize() after template resolution
    assert cfg["container_tag"] == "demo-tag"
    assert cfg["auto_capture"] is False
    assert cfg["auto_recall"] is True


def test_clean_text_for_capture_strips_injected_context():
    text = "hello\n<supermemory-context>ignore me</supermemory-context>\nworld"
    assert _clean_text_for_capture(text) == "hello\nworld"


def test_format_prefetch_context_deduplicates_overlap():
    result = _format_prefetch_context(
        static_facts=["Jordan prefers short answers"],
        dynamic_facts=["Jordan prefers short answers", "Uses Hermes"],
        search_results=[{"memory": "Uses Hermes", "similarity": 0.9}],
        max_results=10,
    )
    assert result.count("Jordan prefers short answers") == 1
    assert result.count("Uses Hermes") == 1
    assert "<supermemory-context>" in result


def test_prefetch_includes_profile_on_first_turn(provider):
    provider._client.profile_response = {
        "static": ["Jordan prefers short answers"],
        "dynamic": ["Current project is Supermemory provider"],
        "search_results": [{"memory": "Working on Hermes memory provider", "similarity": 0.88}],
    }
    provider.on_turn_start(1, "start")
    result = provider.prefetch("what am I working on?")
    assert "User Profile (Persistent)" in result
    assert "Recent Context" in result
    assert "Relevant Memories" in result


def test_prefetch_skips_profile_between_frequency(provider):
    provider._client.profile_response = {
        "static": ["Jordan prefers short answers"],
        "dynamic": ["Current project is Supermemory provider"],
        "search_results": [{"memory": "Working on Hermes memory provider", "similarity": 0.88}],
    }
    provider.on_turn_start(2, "next")
    result = provider.prefetch("what am I working on?")
    assert "Relevant Memories" in result
    assert "User Profile (Persistent)" not in result


def test_sync_turn_skips_trivial_message(provider):
    provider.sync_turn("ok", "sure", session_id="session-1")
    assert provider._client.add_calls == []


def test_sync_turn_persists_cleaned_exchange(provider):
    provider.sync_turn(
        "Please remember this\n<supermemory-context>ignore</supermemory-context>",
        "Got it, storing the context",
        session_id="session-1",
    )
    provider._sync_thread.join(timeout=1)
    assert len(provider._client.add_calls) == 1
    content = provider._client.add_calls[0]["content"]
    assert "ignore" not in content
    assert "[role: user]" in content
    assert "[role: assistant]" in content


def test_on_session_end_ingests_clean_messages(provider):
    messages = [
        {"role": "system", "content": "skip"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    provider.on_session_end(messages)
    assert len(provider._client.ingest_calls) == 1
    payload = provider._client.ingest_calls[0]
    assert payload["session_id"] == "session-1"
    assert payload["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_on_memory_write_tracks_thread(provider):
    provider.on_memory_write("add", "memory", "Jordan likes concise docs")
    assert provider._write_thread is not None
    provider._write_thread.join(timeout=1)
    assert len(provider._client.add_calls) == 1
    assert provider._client.add_calls[0]["metadata"]["type"] == "explicit_memory"


def test_shutdown_joins_and_clears_threads(provider, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_add_memory(content, metadata=None, *, entity_context="",
                        container_tag=None, custom_id=None):
        started.set()
        release.wait(timeout=1)
        provider._client.add_calls.append({
            "content": content,
            "metadata": metadata,
            "entity_context": entity_context,
        })
        return {"id": "mem_slow"}

    monkeypatch.setattr(provider._client, "add_memory", slow_add_memory)

    provider.sync_turn(
        "Please remember this request in long-term memory",
        "Absolutely, I will keep that in long-term memory.",
        session_id="session-1",
    )
    assert started.wait(timeout=1)
    assert provider._sync_thread is not None

    started.clear()
    provider.on_memory_write("add", "memory", "Jordan likes concise docs")
    assert started.wait(timeout=1)
    assert provider._write_thread is not None

    release.set()
    provider.shutdown()

    assert provider._sync_thread is None
    assert provider._write_thread is None
    assert provider._prefetch_thread is None
    assert len(provider._client.add_calls) == 2


def test_store_tool_returns_saved_payload(provider):
    result = json.loads(provider.handle_tool_call("supermemory_store", {"content": "Jordan likes concise docs"}))
    assert result["saved"] is True
    assert result["id"] == "mem_123"


def test_search_tool_formats_results(provider):
    provider._client.search_results = [
        {"id": "m1", "memory": "Jordan likes concise docs", "similarity": 0.92}
    ]
    result = json.loads(provider.handle_tool_call("supermemory_search", {"query": "concise docs"}))
    assert result["count"] == 1
    assert result["results"][0]["similarity"] == 92


def test_forget_tool_by_id(provider):
    result = json.loads(provider.handle_tool_call("supermemory_forget", {"id": "m1"}))
    assert result == {"forgotten": True, "id": "m1"}
    assert provider._client.forgotten_ids == ["m1"]


def test_forget_tool_by_query(provider):
    provider._client.forget_by_query_response = {"success": True, "message": "Forgot one", "id": "m7"}
    result = json.loads(provider.handle_tool_call("supermemory_forget", {"query": "that thing"}))
    assert result["success"] is True
    assert result["id"] == "m7"


def test_profile_tool_formats_sections(provider):
    provider._client.profile_response = {
        "static": ["Jordan prefers concise docs"],
        "dynamic": ["Working on Supermemory provider"],
        "search_results": [],
    }
    result = json.loads(provider.handle_tool_call("supermemory_profile", {}))
    assert result["static_count"] == 1
    assert result["dynamic_count"] == 1
    assert "User Profile (Persistent)" in result["profile"]


def test_handle_tool_call_returns_error_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    p = SupermemoryMemoryProvider()
    result = json.loads(p.handle_tool_call("supermemory_search", {"query": "x"}))
    assert "error" in result


# -- Identity template tests --------------------------------------------------


def test_identity_template_resolved_in_container_tag(monkeypatch, tmp_path):
    """container_tag with {identity} resolves to profile-scoped tag."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"container_tag": "hermes-{identity}"}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli", agent_identity="coder")
    assert p._container_tag == "hermes_coder"


def test_identity_template_default_profile(monkeypatch, tmp_path):
    """Without agent_identity kwarg, {identity} resolves to 'default'."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"container_tag": "hermes-{identity}"}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._container_tag == "hermes_default"


def test_container_tag_env_var_override(monkeypatch, tmp_path):
    """SUPERMEMORY_CONTAINER_TAG env var overrides config."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setenv("SUPERMEMORY_CONTAINER_TAG", "env-override")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._container_tag == "env_override"


# -- Search mode tests --------------------------------------------------------


def test_search_mode_config_passed_to_client(monkeypatch, tmp_path):
    """search_mode from config is passed to _SupermemoryClient."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"search_mode": "memories"}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._search_mode == "memories"
    assert p._client.search_mode == "memories"


def test_invalid_search_mode_falls_back_to_default(monkeypatch, tmp_path):
    """Invalid search_mode falls back to 'hybrid'."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"search_mode": "invalid_mode"}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._search_mode == "hybrid"


# -- Multi-container tests ----------------------------------------------------


def test_multi_container_disabled_by_default(provider):
    """Multi-container is off by default; schemas have no container_tag param."""
    assert provider._enable_custom_containers is False
    schemas = provider.get_tool_schemas()
    for s in schemas:
        assert "container_tag" not in s["parameters"]["properties"]


def test_multi_container_enabled_adds_schema_param(monkeypatch, tmp_path):
    """When enabled, tool schemas include container_tag parameter."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({
        "enable_custom_container_tags": True,
        "custom_containers": ["project-alpha", "shared"],
    }, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._enable_custom_containers is True
    assert p._allowed_containers == ["hermes", "project_alpha", "shared"]
    schemas = p.get_tool_schemas()
    for s in schemas:
        assert "container_tag" in s["parameters"]["properties"]


def test_multi_container_tool_store_with_custom_tag(monkeypatch, tmp_path):
    """supermemory_store uses the resolved container_tag when multi-container is enabled."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({
        "enable_custom_container_tags": True,
        "custom_containers": ["project-alpha"],
    }, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    result = json.loads(p.handle_tool_call("supermemory_store", {
        "content": "test memory",
        "container_tag": "project-alpha",
    }))
    assert result["saved"] is True
    assert result["container_tag"] == "project_alpha"
    assert p._client.add_calls[-1]["container_tag"] == "project_alpha"


def test_multi_container_rejects_unlisted_tag(monkeypatch, tmp_path):
    """Tool calls with a non-whitelisted container_tag return an error."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({
        "enable_custom_container_tags": True,
        "custom_containers": ["allowed-tag"],
    }, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    result = json.loads(p.handle_tool_call("supermemory_store", {
        "content": "test",
        "container_tag": "forbidden-tag",
    }))
    assert "error" in result
    assert "not allowed" in result["error"]


def test_multi_container_system_prompt_includes_instructions(monkeypatch, tmp_path):
    """system_prompt_block includes container list and instructions when multi-container is enabled."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({
        "enable_custom_container_tags": True,
        "custom_containers": ["docs"],
        "custom_container_instructions": "Use docs for documentation context.",
    }, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    block = p.system_prompt_block()
    assert "Multi-container mode enabled" in block
    assert "docs" in block
    assert "Use docs for documentation context." in block


def test_get_config_schema_minimal():
    """get_config_schema only returns the API key field."""
    p = SupermemoryMemoryProvider()
    schema = p.get_config_schema()
    assert len(schema) == 1
    assert schema[0]["key"] == "api_key"
    assert schema[0]["secret"] is True
