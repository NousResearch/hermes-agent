"""Regression contract for PR #99053 review finding "temporary model becomes saved identity".

A turn-scoped ``runtime_override`` model must not replace the durable session route in
``sessions.model``. Its tokens belong in the per-model breakdown (``session_model_usage``)
under a per-turn task, exactly like the existing ``voice_chat`` route does.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from agent import turn_usage
from agent.runtime_override import apply_runtime_override
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_runtime_override import _patch_model_owned_resolution

SESSION_MODEL = "session-model"
OVERRIDE_MODEL = "override-model"
SESSION_ID = "s-route"


def _response(model: str):
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=5,
        total_tokens=105,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return SimpleNamespace(usage=usage, id="gen-99053", provider="openai", model=model)


def _build_agent() -> AIAgent:
    return AIAgent(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        provider="openai",
        api_mode="chat_completions",
        model=SESSION_MODEL,
        session_id=SESSION_ID,
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
    )


def test_override_turn_does_not_become_the_session_route(monkeypatch, tmp_path):
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_agent()
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(SESSION_ID, source="cli", model=SESSION_MODEL)
    agent._session_db = db
    agent._session_db_created = True
    try:
        assert db.get_session(SESSION_ID)["model"] == SESSION_MODEL

        with apply_runtime_override(agent, {"model": OVERRIDE_MODEL}):
            turn_usage.record_response_usage(
                agent,
                _response(OVERRIDE_MODEL),
                messages=[{"role": "user", "content": "hi"}],
                api_call_count=1,
                api_duration=0.1,
                compression_attempts=0,
                max_compression_attempts=3,
            )
        db.flush_token_counts()

        row = db.get_session(SESSION_ID)
        assert row["model"] == SESSION_MODEL, (
            f"the ephemeral override model became the durable session route: {row['model']!r}"
        )

        with sqlite3.connect(db.db_path) as conn:
            models = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT model FROM session_model_usage WHERE session_id = ?",
                    (SESSION_ID,),
                )
            ]
        assert OVERRIDE_MODEL in models, (
            f"the override turn's usage is missing from the per-model breakdown: {models}"
        )
    finally:
        agent.close()
        db.close()


def test_fallback_supersede_restores_the_turn_route_task(monkeypatch):
    """The fallback handoff supersedes the scope; the temporary route marker must
    still be dropped or every later turn is attributed to the override."""
    from agent.runtime_override import consume_runtime_override

    _patch_model_owned_resolution(monkeypatch)
    agent = _build_agent()
    agent._turn_route_task = ""

    with apply_runtime_override(agent, {"model": OVERRIDE_MODEL}):
        assert agent._turn_route_task == "runtime_override"
        consume_runtime_override(agent)  # fallback took over the route

    assert agent._turn_route_task == "", (
        f"the temporary route marker leaked past the fallback handoff: {agent._turn_route_task!r}"
    )


def test_null_session_model_is_not_backfilled_from_the_override(monkeypatch, tmp_path):
    """A session row created without a model must not have the ephemeral override
    model written into `sessions.model` by the accounting path."""
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_agent()
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(SESSION_ID, source="cli")  # model left NULL
    agent._session_db = db
    agent._session_db_created = True
    try:
        assert db.get_session(SESSION_ID).get("model") is None

        with apply_runtime_override(agent, {"model": OVERRIDE_MODEL}):
            turn_usage.record_response_usage(
                agent,
                _response(OVERRIDE_MODEL),
                messages=[{"role": "user", "content": "hi"}],
                api_call_count=1,
                api_duration=0.1,
                compression_attempts=0,
                max_compression_attempts=3,
            )
        db.flush_token_counts()

        assert db.get_session(SESSION_ID).get("model") != OVERRIDE_MODEL, (
            "the ephemeral override model backfilled the NULL session route"
        )
    finally:
        agent.close()
        db.close()


def test_preexisting_turn_route_task_is_preserved(monkeypatch):
    """A route task that another feature already owns (e.g. voice_chat) must be left
    alone by the override scope, before, during and after the turn.

    Contract contributed by the Spark independent review of the G4 fix.
    """
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_agent()
    agent._turn_route_task = "voice_chat"
    try:
        with apply_runtime_override(agent, {"model": OVERRIDE_MODEL}):
            assert agent._turn_route_task == "voice_chat"
        assert agent._turn_route_task == "voice_chat"
    finally:
        agent.close()
