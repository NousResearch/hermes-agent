import json
import os
import stat
import threading

import pytest

from plugins.memory.supermemory import (
    SupermemoryMemoryProvider,
    _clean_text_for_capture,
    _format_connection_summary,
    _format_prefetch_context,
    _load_supermemory_config,
    _probe_supermemory_connection,
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

    def ingest_conversation(self, session_id, messages, metadata=None):
        self.ingest_calls.append({"session_id": session_id, "messages": messages, "metadata": metadata})


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


def test_is_available_true_when_import_missing_but_key_set(monkeypatch):
    # Regression: is_available() must NOT gate on the supermemory SDK being
    # importable. The SDK is lazy-installed at client construction (see
    # _SupermemoryClient.__init__ -> tools.lazy_deps.ensure). Gating here is a
    # chicken-and-egg trap: on a sealed Docker venv the package isn't present
    # until ensure() runs, but ensure() only runs once the provider loads —
    # which this gates. So with the key set and the SDK absent, the provider
    # must still report available. Mirrors honcho/mem0 (config-presence only).
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "supermemory" or name.startswith("supermemory."):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = SupermemoryMemoryProvider()
    assert p.is_available() is True


def test_is_available_false_without_key(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
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


def test_sync_turn_buffers_short_messages(provider):
    # Trivial filtering is no longer applied at sync time — every non-empty turn
    # is buffered and only the full session is written at session boundaries.
    provider.sync_turn("ok", "sure", session_id="session-1")
    assert provider._session_turns == [{"user": "ok", "assistant": "sure"}]
    assert provider._client.add_calls == []


def test_sync_turn_buffers_cleaned_exchange(provider):
    provider.sync_turn(
        "Please remember this\n<supermemory-context>ignore</supermemory-context>",
        "Got it, storing the context",
        session_id="session-1",
    )
    assert len(provider._session_turns) == 1
    turn = provider._session_turns[0]
    assert "ignore" not in turn["user"]
    assert turn["user"].startswith("Please remember this")
    assert turn["assistant"] == "Got it, storing the context"
    # Buffering only — no per-turn writes to the client
    assert provider._client.add_calls == []
    assert provider._client.ingest_calls == []


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
    assert payload["metadata"]["type"] == "full_session"
    assert payload["metadata"]["session_id"] == "session-1"
    assert payload["metadata"]["message_count"] == 2
    # Buffer is cleared after a normal session-end ingest.
    assert provider._session_turns == []


def test_merge_metadata_stamps_sm_source():
    # sm_source routes Hermes writes into the "Hermes" Space in the Supermemory
    # app (functional routing, not telemetry) — must always be present.
    from plugins.memory.supermemory import _SupermemoryClient

    client = _SupermemoryClient.__new__(_SupermemoryClient)
    merged = client._merge_metadata({"type": "explicit_memory"})
    assert merged["sm_source"] == "hermes"
    assert merged["type"] == "explicit_memory"

    # Legacy "source" is migrated into "type" when type is absent.
    merged2 = client._merge_metadata({"source": "conversation_turn"})
    assert merged2["sm_source"] == "hermes"
    assert merged2["type"] == "conversation_turn"
    assert "source" not in merged2


def test_on_memory_write_tracks_thread(provider):
    provider.on_memory_write("add", "memory", "Jordan likes concise docs")
    assert provider._write_thread is not None
    provider._write_thread.join(timeout=1)
    assert len(provider._client.add_calls) == 1
    assert provider._client.add_calls[0]["metadata"]["type"] == "explicit_memory"


def test_shutdown_joins_threads_and_flushes_buffer(provider, monkeypatch):
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

    # sync_turn now only buffers — no thread is spawned.
    provider.sync_turn(
        "Please remember this request in long-term memory",
        "Absolutely, I will keep that in long-term memory.",
        session_id="session-1",
    )
    assert provider._sync_thread is None
    assert len(provider._session_turns) == 1

    # on_memory_write still runs on a background thread.
    provider.on_memory_write("add", "memory", "Jordan likes concise docs")
    assert started.wait(timeout=1)
    assert provider._write_thread is not None

    release.set()
    provider.shutdown()

    # All tracked threads joined and cleared.
    assert provider._sync_thread is None
    assert provider._write_thread is None
    assert provider._prefetch_thread is None
    # Explicit memory write went through.
    assert len(provider._client.add_calls) == 1
    # Buffered turn was flushed as a partial full-session ingest.
    assert len(provider._client.ingest_calls) == 1
    payload = provider._client.ingest_calls[0]
    assert payload["session_id"] == "session-1"
    assert payload["metadata"]["partial"] is True
    assert payload["metadata"]["type"] == "full_session"


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


def test_format_connection_summary_ok():
    summary = _format_connection_summary({
        "ok": True,
        "container_tag": "hermes_coder",
        "profile_facts": 12,
        "auto_recall": True,
        "auto_capture": False,
    })
    assert "✓ Connected" in summary
    assert "container: hermes_coder" in summary
    assert "12 profile facts" in summary
    assert "auto_recall on" in summary
    assert "auto_capture off" in summary


def test_format_connection_summary_single_fact_and_error():
    one = _format_connection_summary({
        "ok": True,
        "container_tag": "hermes",
        "profile_facts": 1,
        "auto_recall": True,
        "auto_capture": True,
    })
    assert "1 profile fact" in one
    assert "1 profile facts" not in one

    err = _format_connection_summary({
        "ok": False,
        "error": "invalid API key",
        "container_tag": "hermes",
        "auto_recall": True,
        "auto_capture": True,
    })
    assert "✗ invalid API key" in err
    assert "container: hermes" in err


def test_probe_supermemory_connection_missing_key(tmp_path):
    status = _probe_supermemory_connection("", str(tmp_path))
    assert status["ok"] is False
    assert status["error"] == "SUPERMEMORY_API_KEY not set"
    assert status["container_tag"] == "hermes"


def _stub_supermemory_importable(monkeypatch):
    """Make ``__import__("supermemory")`` succeed without the real package.

    ``_probe_supermemory_connection`` guards on ``__import__("supermemory")``
    before using the (mocked) client, so tests that mock ``_SupermemoryClient``
    must also satisfy that import guard — otherwise they only pass in an
    environment where the optional ``supermemory`` package happens to be
    installed (and fail on a clean checkout / CI). Mirrors the inverse stub in
    ``test_is_available_false_when_import_missing``.
    """
    import builtins
    import types

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "supermemory":
            return types.ModuleType("supermemory")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_probe_supermemory_connection_success(monkeypatch, tmp_path):
    _stub_supermemory_importable(monkeypatch)
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)

    class CountingClient(FakeClient):
        def get_profile(self, query=None, *, container_tag=None):
            return {
                "static": ["Prefers TypeScript"],
                "dynamic": ["", "Working on Hermes"],
                "search_results": [],
            }

    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", CountingClient)
    status = _probe_supermemory_connection("test-key", str(tmp_path))
    assert status["ok"] is True
    assert status["profile_facts"] == 2
    assert status["auto_recall"] is True


def test_probe_supermemory_connection_client_error(monkeypatch, tmp_path):
    _stub_supermemory_importable(monkeypatch)

    class BrokenClient(FakeClient):
        def get_profile(self, query=None, *, container_tag=None):
            raise RuntimeError("API unavailable")

    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", BrokenClient)
    status = _probe_supermemory_connection("test-key", str(tmp_path))
    assert status["ok"] is False
    assert "API unavailable" in status["error"]


def test_get_status_config_returns_summary(monkeypatch, tmp_path):
    _stub_supermemory_importable(monkeypatch)
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: tmp_path,
    )
    result = SupermemoryMemoryProvider().get_status_config({})
    assert "summary" in result
    assert "✓ Connected" in result["summary"]
    assert "container: hermes" in result["summary"]


def test_post_setup_writes_config_and_prints_summary(monkeypatch, tmp_path, capsys):
    config: dict = {"memory": {}}
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "")
    monkeypatch.setattr(
        "hermes_cli.memory_setup._prompt",
        lambda label, secret=True, default=None: "new-api-key",
    )
    monkeypatch.setattr(
        "plugins.memory.supermemory._probe_supermemory_connection",
        lambda api_key, hermes_home, **kwargs: {
            "ok": True,
            "container_tag": "hermes",
            "profile_facts": 3,
            "auto_recall": True,
            "auto_capture": True,
        },
    )

    saved: dict = {}

    def fake_save_config(cfg):
        saved.update(cfg)

    monkeypatch.setattr("hermes_cli.config.save_config", fake_save_config)

    SupermemoryMemoryProvider().post_setup(str(tmp_path), config)

    assert config["memory"]["provider"] == "supermemory"
    assert saved["memory"]["provider"] == "supermemory"
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "SUPERMEMORY_API_KEY=new-api-key" in env_text

    out = capsys.readouterr().out
    assert "✓ Connected" in out
    assert "3 profile facts" in out
    assert "Memory provider: supermemory" in out


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
def test_save_config_sets_owner_only_permissions(tmp_path):
    """supermemory.json must be written with 0o600 so API key is not world-readable."""
    _save_supermemory_config({"api_key": "sm-test-key"}, str(tmp_path))
    config_file = tmp_path / "supermemory.json"
    assert config_file.exists()
    mode = stat.S_IMODE(config_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"
