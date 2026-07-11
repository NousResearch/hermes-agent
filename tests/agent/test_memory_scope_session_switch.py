"""End-to-end memory-scope rebinding across session transitions."""
from unittest.mock import patch

class _RecordingProvider:
    name = "recording"
    supported_memory_scopes = frozenset({"identity", "user", "conversation", "session"})
    def __init__(self): self.switches = []
    def is_available(self): return True
    def initialize(self, session_id, **kwargs): self.initial_key = kwargs.get("memory_scope_key")
    def get_tool_schemas(self): return []
    def on_session_switch(self, session_id, **kwargs): self.switches.append((session_id, kwargs.get("memory_scope_key")))
    def shutdown(self): pass

def _agent(tmp_path, config, provider=None, **kwargs):
    patches = [
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_constants.get_hermes_home", return_value=tmp_path),
        patch("tools.memory_tool.get_hermes_home", return_value=tmp_path),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ]
    if provider is not None:
        patches.append(patch("plugins.memory.load_memory_provider", return_value=provider))
    for item in patches: item.start()
    try:
        from run_agent import AIAgent
        return AIAgent(api_key="test-key", base_url="https://example.invalid/v1", quiet_mode=True, skip_context_files=True, skip_memory=False, **kwargs)
    finally:
        for item in reversed(patches): item.stop()

def test_session_scope_rebinds_builtin_store_and_prompt_snapshot(tmp_path, monkeypatch):
    from agent.memory_scope import resolve_scope_key
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent(tmp_path, {"memory": {"scope": "session", "memory_enabled": True}, "agent": {}}, session_id="one")
    old_dir = agent._memory_store._get_mem_dir()
    agent._memory_store.add(target="memory", content="old-only")
    next_key = resolve_scope_key("session", session_id="two")
    next_dir = tmp_path / "memories" / "scopes" / next_key
    next_dir.mkdir(parents=True)
    (next_dir / "MEMORY.md").write_text("new-only", encoding="utf-8")
    agent._cached_system_prompt = "stale prompt"
    agent.session_id = "two"
    agent.reset_session_state(old_session_id="one")
    assert agent._memory_store._get_mem_dir() != old_dir
    assert agent._memory_store._get_mem_dir().name == agent._memory_scope_key
    assert not any("old-only" in e for e in agent._memory_store.memory_entries)
    assert any("new-only" in e for e in agent._memory_store.memory_entries)
    assert agent._cached_system_prompt is None

def test_local_conversation_scope_rebinds_on_new_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent(tmp_path, {"memory": {"scope": "conversation", "memory_enabled": True}, "agent": {}}, session_id="one", platform="cli")
    old_dir = agent._memory_store._get_mem_dir()
    agent.session_id = "two"
    agent.reset_session_state(old_session_id="one")
    assert agent._memory_store._get_mem_dir() != old_dir

def test_user_scope_does_not_rebind_for_same_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent(tmp_path, {"memory": {"scope": "user", "memory_enabled": True}, "agent": {}}, session_id="one", platform="whatsapp", user_id="same-user")
    old_store = agent._memory_store
    agent.session_id = "two"
    agent.reset_session_state(old_session_id="one")
    assert agent._memory_store is old_store

def test_provider_receives_exact_new_session_scope_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _RecordingProvider()
    agent = _agent(tmp_path, {"memory": {"scope": "session", "provider": "recording", "memory_enabled": True}, "agent": {}}, provider=provider, session_id="one")
    agent.session_id = "two"
    agent.reset_session_state(old_session_id="one")
    agent._memory_manager.on_session_switch("two", reset=True)
    assert provider.switches == [("two", agent._memory_scope_key)]
