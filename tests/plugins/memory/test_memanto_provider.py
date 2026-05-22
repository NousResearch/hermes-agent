import json

import pytest

from plugins.memory.memanto import (
    MemantoMemoryProvider,
    _detect_memory_type,
    _format_recall_block,
    _load_memanto_config,
    _sanitize_agent_id,
    _save_memanto_config,
)


class FakeClient:
    """Stand-in for ``_MemantoClient`` — records calls, returns canned data."""

    def __init__(self, api_key, agent_id, *, pattern="tool", auto_create=True,
                 session_duration_hours=None):
        self.api_key = api_key
        self._agent_id = agent_id
        self.pattern = pattern
        self.auto_create = auto_create
        self.session_duration_hours = session_duration_hours
        self.ensure_calls = 0
        self.remember_calls = []
        self.answer_calls = []
        self.recall_results = []
        self.answer_response = {"answer": "", "sources": []}

    @property
    def agent_id(self):
        return self._agent_id

    def ensure_session(self):
        self.ensure_calls += 1

    def remember(self, *, memory_type, title, content, confidence,
                 tags=None, source="hermes", provenance="explicit_statement"):
        self.remember_calls.append({
            "memory_type": memory_type,
            "title": title,
            "content": content,
            "confidence": confidence,
            "tags": tags or [],
            "source": source,
            "provenance": provenance,
        })
        return {"memory_id": "mem_123", "agent_id": self._agent_id, "status": "queued"}

    def recall(self, query, *, limit, type=None, min_confidence=None):
        return self.recall_results

    def answer(self, question, *, limit=None):
        self.answer_calls.append({"question": question, "limit": limit})
        return self.answer_response


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("MOORCHEH_API_KEY", "test-key")
    monkeypatch.delenv("MEMANTO_AGENT_ID", raising=False)
    monkeypatch.setattr("plugins.memory.memanto._MemantoClient", FakeClient)
    p = MemantoMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    if p._warmup_thread:
        p._warmup_thread.join(timeout=1)
    return p


# -- Availability -------------------------------------------------------------


def test_is_available_false_without_api_key(monkeypatch):
    monkeypatch.delenv("MOORCHEH_API_KEY", raising=False)
    p = MemantoMemoryProvider()
    assert p.is_available() is False


def test_is_available_false_when_import_missing(monkeypatch):
    monkeypatch.setenv("MOORCHEH_API_KEY", "test-key")

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "memanto":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = MemantoMemoryProvider()
    assert p.is_available() is False


# -- Helpers ------------------------------------------------------------------


def test_sanitize_agent_id_coerces_charset():
    assert _sanitize_agent_id("Hermes Coder!@#") == "Hermes-Coder"
    assert _sanitize_agent_id("") == "hermes"
    assert _sanitize_agent_id("a" * 100) == "a" * 64


def test_detect_memory_type():
    assert _detect_memory_type("User prefers dark mode") == "preference"
    assert _detect_memory_type("We decided to use Postgres") == "decision"
    assert _detect_memory_type("The API is rate limited") == "fact"


def test_load_and_save_config_round_trip(tmp_path):
    _save_memanto_config({"agent_id": "Demo Agent", "auto_capture": False}, str(tmp_path))
    cfg = _load_memanto_config(str(tmp_path))
    # save_config is invoked via the provider; here we only check load defaults.
    assert cfg["agent_id"] == "Demo Agent"
    assert cfg["auto_capture"] is False
    assert cfg["auto_recall"] is True
    assert cfg["pattern"] == "tool"


def test_save_config_sanitizes_concrete_agent_id(tmp_path):
    p = MemantoMemoryProvider()
    p.save_config({"agent_id": "My Agent!"}, str(tmp_path))
    cfg = _load_memanto_config(str(tmp_path))
    assert cfg["agent_id"] == "My-Agent"


def test_save_config_preserves_identity_template(tmp_path):
    p = MemantoMemoryProvider()
    p.save_config({"agent_id": "hermes-{identity}"}, str(tmp_path))
    cfg = _load_memanto_config(str(tmp_path))
    assert cfg["agent_id"] == "hermes-{identity}"


def test_format_recall_block_renders_types_and_scores():
    block = _format_recall_block(
        [
            {"type": "preference", "content": "Prefers dark mode", "score": 0.91},
            {"type": "fact", "content": "Lives in Berlin"},
        ],
        max_results=10,
    )
    assert "<memanto-memory>" in block
    assert "[preference] [91%] Prefers dark mode" in block
    assert "[fact] Lives in Berlin" in block


def test_format_recall_block_empty():
    assert _format_recall_block([], max_results=10) == ""


# -- Identity / config resolution --------------------------------------------


def test_identity_template_resolved(monkeypatch, tmp_path):
    monkeypatch.setenv("MOORCHEH_API_KEY", "test-key")
    monkeypatch.delenv("MEMANTO_AGENT_ID", raising=False)
    monkeypatch.setattr("plugins.memory.memanto._MemantoClient", FakeClient)
    _save_memanto_config({"agent_id": "hermes-{identity}"}, str(tmp_path))
    p = MemantoMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli", agent_identity="coder")
    assert p._agent_id == "hermes-coder"


def test_identity_template_default_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("MOORCHEH_API_KEY", "test-key")
    monkeypatch.delenv("MEMANTO_AGENT_ID", raising=False)
    monkeypatch.setattr("plugins.memory.memanto._MemantoClient", FakeClient)
    _save_memanto_config({"agent_id": "hermes-{identity}"}, str(tmp_path))
    p = MemantoMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._agent_id == "hermes-default"


def test_agent_id_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MOORCHEH_API_KEY", "test-key")
    monkeypatch.setenv("MEMANTO_AGENT_ID", "env-agent")
    monkeypatch.setattr("plugins.memory.memanto._MemantoClient", FakeClient)
    p = MemantoMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._agent_id == "env-agent"


# -- Prefetch -----------------------------------------------------------------


def test_prefetch_formats_recall(provider):
    provider._client.recall_results = [
        {"id": "m1", "type": "preference", "content": "Prefers dark mode", "score": 0.88}
    ]
    result = provider.prefetch("ui preferences")
    assert "Relevant Memories" in result
    assert "Prefers dark mode" in result


def test_prefetch_empty_query_returns_blank(provider):
    assert provider.prefetch("   ") == ""


def test_prefetch_disabled_when_auto_recall_off(monkeypatch, tmp_path):
    monkeypatch.setenv("MOORCHEH_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.memanto._MemantoClient", FakeClient)
    _save_memanto_config({"auto_recall": False}, str(tmp_path))
    p = MemantoMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    if p._warmup_thread:
        p._warmup_thread.join(timeout=1)
    p._client.recall_results = [{"type": "fact", "content": "x", "score": 0.9}]
    assert p.prefetch("anything") == ""


# -- Turn capture -------------------------------------------------------------


def test_sync_turn_skips_trivial(provider):
    provider.sync_turn("ok", "sure", session_id="session-1")
    assert provider._client.remember_calls == []


def test_sync_turn_persists_event(provider):
    provider.sync_turn(
        "Please remember I work in Pacific time",
        "Got it — noting your timezone as Pacific.",
        session_id="session-1",
    )
    provider._sync_thread.join(timeout=1)
    assert len(provider._client.remember_calls) == 1
    call = provider._client.remember_calls[0]
    assert call["memory_type"] == "event"
    assert "User:" in call["content"] and "Assistant:" in call["content"]


def test_sync_turn_skipped_in_cron_context(monkeypatch, tmp_path):
    monkeypatch.setenv("MOORCHEH_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.memanto._MemantoClient", FakeClient)
    p = MemantoMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path), platform="cron", agent_context="cron")
    if p._warmup_thread:
        p._warmup_thread.join(timeout=1)
    p.sync_turn(
        "Remember this long enough message to pass the filter",
        "Acknowledged, storing the long enough message now.",
        session_id="s1",
    )
    assert p._client.remember_calls == []


# -- Built-in memory mirroring ------------------------------------------------


def test_on_memory_write_mirrors_to_memanto(provider):
    provider.on_memory_write("add", "memory", "The deploy pipeline lives in gh actions")
    provider._write_thread.join(timeout=1)
    assert len(provider._client.remember_calls) == 1
    assert provider._client.remember_calls[0]["source"] == "hermes-memory"


def test_on_memory_write_user_target_is_preference(provider):
    provider.on_memory_write("add", "user", "Name is Jordan")
    provider._write_thread.join(timeout=1)
    assert provider._client.remember_calls[0]["memory_type"] == "preference"


def test_on_memory_write_ignores_non_add(provider):
    provider.on_memory_write("remove", "memory", "something")
    assert provider._write_thread is None


# -- Tools --------------------------------------------------------------------


def test_get_tool_schemas_names(provider):
    names = {s["name"] for s in provider.get_tool_schemas()}
    assert names == {"memanto_remember", "memanto_recall", "memanto_answer"}


def test_remember_tool(provider):
    result = json.loads(provider.handle_tool_call(
        "memanto_remember", {"content": "Prefers TypeScript", "type": "preference"}
    ))
    assert result["saved"] is True
    assert result["memory_id"] == "mem_123"
    assert result["type"] == "preference"
    assert provider._client.remember_calls[0]["source"] == "hermes-tool"


def test_remember_tool_infers_type_when_invalid(provider):
    json.loads(provider.handle_tool_call(
        "memanto_remember", {"content": "User prefers vim", "type": "not-a-type"}
    ))
    assert provider._client.remember_calls[0]["memory_type"] == "preference"


def test_remember_tool_requires_content(provider):
    result = json.loads(provider.handle_tool_call("memanto_remember", {"content": "  "}))
    assert "error" in result


def test_recall_tool_formats_results(provider):
    provider._client.recall_results = [
        {"id": "m1", "type": "fact", "content": "Lives in Berlin", "similarity_score": 0.77}
    ]
    result = json.loads(provider.handle_tool_call("memanto_recall", {"query": "where"}))
    assert result["count"] == 1
    assert result["results"][0]["score"] == 77
    assert result["results"][0]["type"] == "fact"


def test_answer_tool(provider):
    provider._client.answer_response = {
        "answer": "They prefer dark mode.",
        "sources": [{"id": "m1"}],
    }
    result = json.loads(provider.handle_tool_call("memanto_answer", {"question": "ui pref?"}))
    assert result["answer"] == "They prefer dark mode."
    assert result["count"] == 1


def test_handle_tool_call_unknown(provider):
    result = json.loads(provider.handle_tool_call("memanto_unknown", {}))
    assert "error" in result


def test_handle_tool_call_unconfigured_returns_error(monkeypatch):
    monkeypatch.delenv("MOORCHEH_API_KEY", raising=False)
    p = MemantoMemoryProvider()
    result = json.loads(p.handle_tool_call("memanto_recall", {"query": "x"}))
    assert "error" in result


def test_get_config_schema(provider):
    schema = provider.get_config_schema()
    keys = {f["key"] for f in schema}
    assert "api_key" in keys and "agent_id" in keys
    api_field = next(f for f in schema if f["key"] == "api_key")
    assert api_field["secret"] is True
    assert api_field["env_var"] == "MOORCHEH_API_KEY"


def test_system_prompt_block_mentions_agent(provider):
    block = provider.system_prompt_block()
    assert "Memanto Memory Agent" in block
    assert provider._agent_id in block
