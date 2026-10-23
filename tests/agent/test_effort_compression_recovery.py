"""Recover effort baselines after tagged Hermes summary replacement, not native compaction."""
from types import SimpleNamespace

import pytest

from agent.effort_updates import (
    EFFORT_BASELINE_KEY,
    effort_update,
    record_effort_switch,
    resolve_effort_updates,
)
from hermes_state import SessionDB


def _agent(**kwargs):
    return SimpleNamespace(
        model="gpt-6.1-sol", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        _session_init_model_config={},
        reasoning_config={"enabled": True, "effort": "low"}, **kwargs,
    )


def test_sqlite_compaction_resume_recovers_last_effective_effort(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("effort", "cli", model="gpt-6.1-sol", model_config={})
        agent = _agent(_session_db=db, session_id="effort")
        assert not record_effort_switch(agent, [])
        history = [{"role": "user", "content": "first"},
                   {"role": "assistant", "content": "answer"}]
        agent.reasoning_config["effort"] = "high"
        assert record_effort_switch(agent, history)
        for message in history:
            db.append_message(
                "effort", message["role"], content=message.get("content"),
                display_kind=message.get("display_kind"),
                display_metadata=message.get("display_metadata"),
            )
        db.archive_and_compact("effort", [
            {"role": "assistant", "content": "compacted summary", "_compressed_summary": True},
        ])
    # Reopen the database as well as constructing a fresh agent: no process-local baseline.
    with SessionDB(db_path=tmp_path / "state.db") as db:
        history = db.get_messages_as_conversation("effort", repair_alternation=True)
        assert history[0].get("_compressed_summary")
        assert not any(effort_update(message) for message in history)
        resumed = _agent(_session_db=db, session_id="effort")
        resumed.reasoning_config["effort"] = "high"
        assert not record_effort_switch(resumed, history)
        stored = db.get_session_model_config_value("effort", EFFORT_BASELINE_KEY)
        assert stored["effort"] == "high"
        resumed.reasoning_config["effort"] = "medium"
        assert record_effort_switch(resumed, history)
        assert effort_update(history[-1])["previous"] == "high"
        assert resolve_effort_updates(history, "medium")[1] == "high"


@pytest.mark.parametrize("new_effort", ["low", "medium"])
def test_first_post_compression_change_uses_last_history_effort(new_effort):
    agent = _agent()
    assert not record_effort_switch(agent, [])
    history = [{"role": "assistant", "content": "answer"}]
    agent.reasoning_config["effort"] = "high"
    assert record_effort_switch(agent, history)
    compressed = [{"role": "assistant", "content": "new summary", "_compressed_summary": True}]
    agent.reasoning_config["effort"] = new_effort
    assert record_effort_switch(agent, compressed)
    assert effort_update(compressed[-1])["previous"] == "high"
    assert resolve_effort_updates(compressed, new_effort)[1] == "high"


@pytest.mark.parametrize("tagged", [False, True])
def test_rollback_with_unchanged_summary_preserves_original_baseline(tagged):
    agent = _agent()
    assert not record_effort_switch(agent, [])
    original = [{"role": "assistant", "content": "old summary", "_compressed_summary": tagged}]
    history = list(original)
    agent.reasoning_config["effort"] = "high"
    assert record_effort_switch(agent, history)
    agent.reasoning_config["effort"] = "low"
    assert not record_effort_switch(agent, original)
    assert agent._session_init_model_config[EFFORT_BASELINE_KEY]["effort"] == "low"
