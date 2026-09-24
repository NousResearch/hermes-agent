"""A gateway turn served by the fallback chain must not persist the fallback's live route as
the session row's route — resume restores the model column and ``gateway_runtime`` as the
session's primary, so the fallback would outlive the single turn that needed it (#121506)."""

import json
from types import SimpleNamespace

from gateway.run_turn import GatewayTurnMixin
from hermes_state import SessionDB


def _agent(fallback: bool):
    agent = SimpleNamespace(
        model="fallback-model", provider="custom:local",
        base_url="http://127.0.0.1:8000/v1", api_mode="chat_completions",
        _fallback_activated=fallback,
    )
    if fallback:
        agent._primary_runtime = {
            "model": "primary-model", "provider": "anthropic",
            "base_url": "https://anthropic.example/v1", "api_mode": "anthropic_messages",
        }
    return agent


def _sync(tmp_path, agent):
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="s1", source="telegram", model="primary-model")
    runner = GatewayTurnMixin()
    runner._session_db = SimpleNamespace(_db=db)
    runner._sync_session_model_from_agent("s1", agent)
    return db.get_session("s1")


def test_fallback_turn_persists_requested_route(tmp_path):
    row = _sync(tmp_path, _agent(fallback=True))
    assert row["model"] == "primary-model"
    runtime = json.loads(row["model_config"])["gateway_runtime"]
    assert runtime["provider"] == "anthropic"
    assert runtime["base_url"] == "https://anthropic.example/v1"
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["fallback_active"] is True


def test_healthy_turn_persists_live_route(tmp_path):
    row = _sync(tmp_path, _agent(fallback=False))
    assert row["model"] == "fallback-model"
    runtime = json.loads(row["model_config"])["gateway_runtime"]
    assert runtime["provider"] == "custom:local"
    assert runtime["fallback_active"] is False


def test_fallback_without_snapshot_keeps_live_route(tmp_path):
    """Defensive: a fallback flag without a primary snapshot (plugin engines) falls back to the
    live route rather than persisting a half-empty one."""
    agent = _agent(fallback=True)
    agent._primary_runtime = {}
    row = _sync(tmp_path, agent)
    assert row["model"] == "fallback-model"
    runtime = json.loads(row["model_config"])["gateway_runtime"]
    assert runtime["provider"] == "custom:local"
