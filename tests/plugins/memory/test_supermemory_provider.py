import json
import os
import stat
import threading

import pytest

from plugins.memory.supermemory import (
    SupermemoryMemoryProvider,
    _capture_custom_id,
    _clean_text_for_capture,
    _format_connection_summary,
    _format_prefetch_context,
    _load_supermemory_config,
    _probe_supermemory_connection,
    _save_supermemory_config,
)


class FakeClient:
    def __init__(self, api_key: str, timeout: float, container_tag: str, search_mode: str = "hybrid",
                 base_url: str = ""):
        self.api_key = api_key
        self.timeout = timeout
        self.container_tag = container_tag
        self.search_mode = search_mode
        self.base_url = base_url
        self.add_calls = []
        self.search_results = []
        self.profile_response = {"static": [], "dynamic": [], "search_results": []}
        self.fail_add = False
        self.forgotten_ids = []
        self.forget_by_query_response = {"success": True, "message": "Forgot"}

    def add_memory(self, content, metadata=None, *, entity_context="",
                   container_tag=None, custom_id=None):
        if self.fail_add:
            raise RuntimeError("boom")
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


def test_clean_text_for_capture_strips_inline_data_uri():
    text = "look: data:image/png;base64,iVBORw0KGgoAAAANSUhEUg== ok"
    assert _clean_text_for_capture(text) == "look: [image] ok"


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


def test_capture_custom_id_buckets_by_four_hours():
    from datetime import datetime, timezone
    a = _capture_custom_id("session-1", datetime(2026, 9, 12, 3, 59, tzinfo=timezone.utc))
    b = _capture_custom_id("session-1", datetime(2026, 9, 12, 4, 0, tzinfo=timezone.utc))
    assert a == "session_1_2026-09-12_b0"
    assert b == "session_1_2026-09-12_b1"
    assert _capture_custom_id("", datetime(2026, 9, 12, 23, 0, tzinfo=timezone.utc)) == "hermes_2026-09-12_b5"


def test_sync_turn_writes_turn_to_session_document(provider):
    # Every completed turn is appended to one document per session per 4h window.
    provider.sync_turn("hello", "hi there", session_id="session-1")
    assert len(provider._client.add_calls) == 1
    call = provider._client.add_calls[0]
    assert call["custom_id"] == _capture_custom_id("session-1")
    assert call["content"] == "[role: user]\nhello\n[user:end]\n[role: assistant]\nhi there\n[assistant:end]"
    assert call["metadata"]["type"] == "conversation"
    assert call["metadata"]["session_id"] == "session-1"
    assert call["entity_context"]
    assert provider._pending_turns == []


def test_sync_turn_skips_empty_turn(provider):
    provider.sync_turn("", "<supermemory-context>x</supermemory-context>", session_id="session-1")
    assert provider._client.add_calls == []


def test_failed_turn_write_is_retried_at_session_end(provider):
    provider._client.fail_add = True
    provider.sync_turn("hello", "hi there", session_id="session-1")
    assert provider._client.add_calls == []
    assert provider._pending_turns == [{"user": "hello", "assistant": "hi there", "session_id": "session-1"}]

    provider._client.fail_add = False
    provider.on_session_end([])
    assert len(provider._client.add_calls) == 1
    call = provider._client.add_calls[0]
    assert call["custom_id"] == _capture_custom_id("session-1")
    assert "hello" in call["content"]
    assert provider._pending_turns == []


def test_pending_turns_are_batched_with_next_turn(provider):
    provider._client.fail_add = True
    provider.sync_turn("one", "uno", session_id="session-1")
    provider._client.fail_add = False
    provider.sync_turn("two", "dos", session_id="session-1")
    assert len(provider._client.add_calls) == 1
    assert provider._client.add_calls[0]["content"].index("one") < provider._client.add_calls[0]["content"].index("two")
    assert provider._pending_turns == []


def test_session_switch_flushes_pending_to_old_session(provider):
    provider._client.fail_add = True
    provider.sync_turn("hello", "hi", session_id="session-1")
    provider._client.fail_add = False
    provider.on_session_switch("session-2", reset=True)
    assert provider._client.add_calls[0]["custom_id"] == _capture_custom_id("session-1")
    assert provider._session_id == "session-2"
    assert provider._pending_turns == []


def test_failed_switch_flush_keeps_old_session_turns_for_later_retry(provider):
    provider._client.fail_add = True
    provider.sync_turn("old turn", "old reply", session_id="session-1")
    provider.on_session_switch("session-2", reset=True)  # flush fails: service unavailable at the boundary
    assert provider._session_id == "session-2"
    assert provider._pending_turns == [{"user": "old turn", "assistant": "old reply", "session_id": "session-1"}]

    provider._client.fail_add = False
    provider.sync_turn("new turn", "new reply", session_id="session-2")
    calls = provider._client.add_calls
    assert [c["custom_id"] for c in calls] == [_capture_custom_id("session-1"), _capture_custom_id("session-2")]
    assert calls[0]["metadata"]["session_id"] == "session-1" and "old turn" in calls[0]["content"]
    assert calls[1]["metadata"]["session_id"] == "session-2" and "new turn" in calls[1]["content"]
    assert provider._pending_turns == []


def test_concurrent_sync_turn_and_session_switch_do_not_duplicate_pending(provider, monkeypatch):
    # Worker thread: sync_turn(B) with pending [A] snapshots [A, B] and blocks inside add_memory.
    # Caller thread: on_session_switch must wait for that write, not re-send A from a stale snapshot.
    provider._client.fail_add = True
    provider.sync_turn("A", "a", session_id="session-1")
    provider._client.fail_add = False
    entered, release = threading.Event(), threading.Event()
    real_add = provider._client.add_memory

    def slow_add(content, metadata=None, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return real_add(content, metadata=metadata, **kwargs)

    monkeypatch.setattr(provider._client, "add_memory", slow_add)
    worker = threading.Thread(target=provider.sync_turn, args=("B", "b"), kwargs={"session_id": "session-1"})
    worker.start()
    assert entered.wait(timeout=2)
    switcher = threading.Thread(target=provider.on_session_switch, args=("session-2",), kwargs={"reset": True})
    switcher.start()
    switcher.join(timeout=0.2)
    assert switcher.is_alive()  # blocked on the capture lock while the worker's write is in flight
    release.set()
    worker.join(timeout=2); switcher.join(timeout=2)
    assert not worker.is_alive() and not switcher.is_alive()
    assert len(provider._client.add_calls) == 1
    assert provider._client.add_calls[0]["content"].count("[role: user]") == 2  # A and B, once each
    assert provider._pending_turns == []
    assert provider._session_id == "session-2"


def test_failed_switch_flush_is_retried_at_shutdown(provider):
    provider._client.fail_add = True
    provider.sync_turn("old turn", "old reply", session_id="session-1")
    provider.on_session_switch("session-2", reset=True)
    provider._client.fail_add = False
    provider.shutdown()
    assert provider._client.add_calls[0]["custom_id"] == _capture_custom_id("session-1")
    assert provider._pending_turns == []


def test_sync_turn_drops_inline_image_payloads(provider):
    blob = "A" * 4096
    provider.sync_turn(f"describe this data:image/png;base64,{blob}", "a screenshot", session_id="session-1")
    call = provider._client.add_calls[0]
    assert "describe this [image]" in call["content"]
    assert blob not in json.dumps(call)


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


def test_shutdown_joins_threads_and_flushes_buffer(provider, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_add_memory(content, metadata=None, *, entity_context="",
                        container_tag=None, custom_id=None):
        if provider._client.fail_add:
            raise RuntimeError("boom")
        started.set()
        release.wait(timeout=1)
        provider._client.add_calls.append({
            "content": content,
            "metadata": metadata,
            "entity_context": entity_context,
            "custom_id": custom_id,
        })
        return {"id": "mem_slow"}

    monkeypatch.setattr(provider._client, "add_memory", slow_add_memory)

    # A failed turn write stays pending; shutdown retries it.
    provider._client.fail_add = True
    provider.sync_turn(
        "Please remember this request in long-term memory",
        "Absolutely, I will keep that in long-term memory.",
        session_id="session-1",
    )
    provider._client.fail_add = False
    assert provider._sync_thread is None
    assert len(provider._pending_turns) == 1

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
    # Explicit memory write and the retried turn both went through.
    assert len(provider._client.add_calls) == 2
    flushed = next(c for c in provider._client.add_calls if c.get("custom_id") == _capture_custom_id("session-1"))
    assert provider._pending_turns == []


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


def test_container_tag_env_var_override(monkeypatch, tmp_path):
    """SUPERMEMORY_CONTAINER_TAG env var overrides config."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setenv("SUPERMEMORY_CONTAINER_TAG", "env-override")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._container_tag == "env_override"


# -- Search mode tests --------------------------------------------------------


def test_invalid_search_mode_falls_back_to_default(monkeypatch, tmp_path):
    """Invalid search_mode falls back to 'hybrid'."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    _save_supermemory_config({"search_mode": "invalid_mode"}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._search_mode == "hybrid"


# -- Base URL tests -------------------------------------------------------------


def test_base_url_defaults_to_cloud(monkeypatch, tmp_path):
    """Without config or env override, the client targets api.supermemory.ai."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.delenv("SUPERMEMORY_BASE_URL", raising=False)
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    p = SupermemoryMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._base_url == "https://api.supermemory.ai"
    assert p._client.base_url == "https://api.supermemory.ai"


def test_client_passes_custom_base_url_to_sdk(monkeypatch):
    """SDK client receives the normalized base URL."""
    import sys
    import types

    from plugins.memory.supermemory import _SupermemoryClient

    captured = {}

    class StubSupermemory:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = types.ModuleType("supermemory")
    module.Supermemory = StubSupermemory
    monkeypatch.setitem(sys.modules, "supermemory", module)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *args, **kwargs: None)

    client = _SupermemoryClient(
        api_key="test-key",
        timeout=1.0,
        container_tag="hermes",
        base_url="http://localhost:6767/",
    )

    assert client._base_url == "http://localhost:6767"
    assert captured["base_url"] == "http://localhost:6767"


# -- Multi-container tests ----------------------------------------------------


def test_multi_container_disabled_by_default(provider):
    """Multi-container is off by default; schemas have no container_tag param."""
    assert provider._enable_custom_containers is False
    schemas = provider.get_tool_schemas()
    for s in schemas:
        assert "container_tag" not in s["parameters"]["properties"]


def test_get_config_schema_minimal():
    """get_config_schema only returns the API key field."""
    p = SupermemoryMemoryProvider()
    schema = p.get_config_schema()
    assert len(schema) == 1
    assert schema[0]["key"] == "api_key"
    assert schema[0]["secret"] is True


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
