"""Commit/recovery invariants for a model-only runtime override (#99053)."""
from __future__ import annotations
from types import SimpleNamespace

from agent.runtime_override import apply_runtime_override
from tests.agent.test_runtime_override import _build_model_owned_agent, _patch_model_owned_resolution


def _setup_session_agent(monkeypatch, tmp_path):
    from hermes_state import SessionDB
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent()
    agent.session_id = "parent"
    agent.platform = "cli"
    agent._build_system_prompt = lambda system_message=None: f"Model: {agent.model}\\nProvider: {agent.provider}"
    db = SessionDB(db_path=tmp_path / "scope.db")
    db.create_session("parent", source="cli", model=agent.model)
    agent._session_db = db
    agent.context_compressor.bind_session_state(db, agent.session_id)
    return agent, db


def test_rotated_compaction_rebinds_restored_compressor(monkeypatch, tmp_path):
    agent, db = _setup_session_agent(monkeypatch, tmp_path)
    source_model = agent.model
    with apply_runtime_override(agent, {"model": "override-model"}) as scope:
        db.create_session("child", source="cli", model=source_model)
        db.update_system_prompt("child", "Model: override-model")
        agent.session_id = "child"
        agent._cached_system_prompt = "Model: override-model"
        scope._durable_compaction_committed = True
        scope._durable_compaction_rotated = True
    assert agent.model == source_model
    assert agent.context_compressor._session_id == "child"
    assert db.get_session("child")["model"] == source_model
    assert agent._cached_system_prompt == db.get_session("child")["system_prompt"]
    assert agent._cached_system_prompt.startswith(f"Model: {source_model}")


def test_in_place_compaction_keeps_new_committed_prompt(monkeypatch, tmp_path):
    agent, db = _setup_session_agent(monkeypatch, tmp_path)
    source_model = agent.model
    with apply_runtime_override(agent, {"model": "override-model"}) as scope:
        db.update_system_prompt("parent", "Model: override-model")
        agent._cached_system_prompt = "Model: override-model"
        scope._durable_compaction_committed = True
        scope._durable_compaction_rotated = False
    assert agent.model == source_model
    assert db.get_session("parent")["model"] == source_model
    assert agent._cached_system_prompt == db.get_session("parent")["system_prompt"]
    assert agent._cached_system_prompt.startswith(f"Model: {source_model}")


def test_anthropic_oauth_refresh_updates_restored_compressor(monkeypatch):
    from agent.client_lifecycle import ClientLifecycleMixin

    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent()
    old_token = "sk-ant-old-token"
    new_token = "sk-ant-new-token"
    agent.provider = "anthropic"
    agent.api_mode = "anthropic_messages"
    agent.base_url = "https://api.anthropic.com"
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._anthropic_api_key = old_token
    agent._is_anthropic_oauth = True
    agent.api_key = old_token
    agent.context_compressor.api_key = old_token
    agent._anthropic_client = SimpleNamespace(close=lambda: None)
    agent._build_direct_anthropic_client = lambda token, base_url: SimpleNamespace(close=lambda: None)
    agent._anthropic_oauth_flag = lambda token: True
    agent._primary_runtime = {"api_key": old_token, "anthropic_api_key": old_token, "compressor_api_key": old_token}
    monkeypatch.setattr("agent.anthropic_credentials.resolve_anthropic_token", lambda model: new_token)
    monkeypatch.setattr("agent.auxiliary_key_rotation.rotate_runtime_main_api_key", lambda old, new: None)
    base_compressor = agent.context_compressor
    with apply_runtime_override(agent, {"model": "override-model"}):
        assert ClientLifecycleMixin._try_refresh_anthropic_client_credentials(agent)
        assert agent.context_compressor.api_key == new_token
    assert agent.context_compressor is base_compressor
    assert base_compressor.api_key == new_token
    assert agent.api_key == new_token
    assert agent._primary_runtime["api_key"] == new_token

def test_real_rotation_publisher_uses_durable_session_model(monkeypatch, tmp_path):
    """Call the production child-publication path, with only persistence I/O stubbed."""
    from agent.conversation_compression import _publish_rotated_compaction
    from agent import conversation_compression as cc

    agent, db = _setup_session_agent(monkeypatch, tmp_path)
    original_model = agent.model
    agent._session_init_model_config = {}
    agent._flush_messages_to_session_db = lambda *a, **kw: None
    publication = []
    monkeypatch.setattr(db, "publish_compression_child", lambda **kw: publication.append(kw))
    monkeypatch.setattr(cc, "mint_session_id", lambda: "child")
    monkeypatch.setattr(cc, "_rebind_session_context", lambda *_: None)
    monkeypatch.setattr(cc, "_carry_session_state_to_child", lambda *_: None)
    lease = SimpleNamespace(holder=None, ttl=60, watermark=None)
    with apply_runtime_override(agent, {"model": "override-model"}):
        assert agent.model == "override-model"
        _publish_rotated_compaction(
            agent, [], [], new_system_prompt="Model: override-model",
            lease=lease, old_session_id="parent", compressed_user_turn_outcome="none",
        )
        assert publication and publication[0]["model"] == original_model
        agent.session_id = "parent"  # direct publisher test does not commit a child row
