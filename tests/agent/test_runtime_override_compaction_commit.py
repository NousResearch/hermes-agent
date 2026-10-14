"""Production commit integration for model-only ephemeral override (#99053).

These tests exercise _commit_compaction(), not manually-set scope commit flags.
"""
from __future__ import annotations

from types import SimpleNamespace
import time

from agent.runtime_override import apply_runtime_override
from tests.agent.test_runtime_override_boundary_handoffs import _setup_session_agent


def test_real_rotation_commit_hands_off_session_model_and_compressor(monkeypatch, tmp_path):
    from agent import conversation_compression as cc

    agent, db = _setup_session_agent(monkeypatch, tmp_path)
    original_model = agent.model
    agent._session_init_model_config = {}
    agent._session_db_created = True
    agent._flush_messages_to_session_db = lambda *a, **k: None
    agent.commit_memory_session = lambda *_: None
    agent._emit_warning = lambda *_: None
    monkeypatch.setattr(cc, "mint_session_id", lambda: "child")
    monkeypatch.setattr(cc, "_rebind_session_context", lambda *_: None)
    monkeypatch.setattr(cc, "_carry_session_state_to_child", lambda *_: None)

    messages = [{"role": "user", "content": "Important archived context. " * 100}]
    compressed = [{"role": "user", "content": "Summary of previous conversation."}]
    lease = SimpleNamespace(holder=None, watermark=None, ttl=60)
    attempt = SimpleNamespace(started_at=time.monotonic(), seed={}, snapshot={})

    with apply_runtime_override(agent, {"model": "override-model"}) as scope:
        outcome = cc._commit_compaction(
            agent, messages, compressed, in_place=False, lease=lease,
            new_system_prompt="Model: override-model", system_message="",
            compressed_user_turn_outcome="none",
            messages_before_compression=list(messages),
            made_progress=True, attempt=attempt,
        )
        assert outcome.session_commit_succeeded, outcome
        assert scope._durable_compaction_committed
        assert scope._durable_compaction_rotated
        assert db.get_session("child")["model"] == original_model

    assert agent.model == original_model
    assert agent.session_id == "child"
    assert agent.context_compressor._session_id == "child"
    assert db.get_session("child")["system_prompt"] == agent._cached_system_prompt
    assert "Model: " + original_model in agent._cached_system_prompt


def test_real_in_place_commit_hands_off_prompt_without_model_leak(monkeypatch, tmp_path):
    from agent import conversation_compression as cc

    agent, db = _setup_session_agent(monkeypatch, tmp_path)
    original_model = agent.model
    agent._session_db_created = True
    agent.commit_memory_session = lambda *_: None
    agent._emit_warning = lambda *_: None
    messages = [{"role": "user", "content": "Important archived context. " * 100}]
    compressed = [{"role": "user", "content": "Short summary."}]
    lease = SimpleNamespace(holder=None, watermark=None, ttl=60)
    attempt = SimpleNamespace(started_at=time.monotonic(), seed={}, snapshot={})

    with apply_runtime_override(agent, {"model": "override-model"}) as scope:
        outcome = cc._commit_compaction(
            agent, messages, compressed, in_place=True, lease=lease,
            new_system_prompt="Model: override-model", system_message="",
            compressed_user_turn_outcome="none",
            messages_before_compression=list(messages),
            made_progress=True, attempt=attempt,
        )
        assert outcome.session_commit_succeeded, outcome
        assert scope._durable_compaction_committed
        assert not scope._durable_compaction_rotated

    assert agent.model == original_model
    assert agent.session_id == "parent"
    assert agent.context_compressor._session_id == agent.session_id
    assert db.get_session("parent")["model"] == original_model
    assert db.get_session("parent")["system_prompt"] == agent._cached_system_prompt
    assert "Model: " + original_model in agent._cached_system_prompt
