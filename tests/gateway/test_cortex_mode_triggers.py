"""Regression tests for Cortex plaintext learning/normal mode triggers."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_state import SessionDB


def _make_runner(session_id="session-1"):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._pending_model_notes = {}
    runner._evict_cached_agent = lambda _session_key: None

    class Store:
        def get_or_create_session(self, source):
            return SimpleNamespace(
                session_key=f"{source.platform.value}:{source.chat_id}",
                session_id=session_id,
            )

    runner.session_store = Store()
    return runner


def _make_event(text="Learning Mode"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="7704114243",
            chat_id="7704114243",
            chat_type="dm",
        ),
    )


def test_cortex_trigger_matches_raw_and_voice_wrapper():
    runner = _make_runner()

    assert runner._cortex_mode_trigger_from_text("Learning Mode")[1] == "gpt-5.4-mini"
    assert runner._cortex_mode_trigger_from_text("learning mode.")[1] == "gpt-5.4-mini"
    assert (
        runner._cortex_mode_trigger_from_text(
            '[The user sent a voice message~ Here\'s what they said: "Learning Mode"]'
        )[1]
        == "gpt-5.4-mini"
    )
    assert runner._cortex_mode_trigger_from_text("Can you use learning mode?") is None


@pytest.mark.asyncio
async def test_cortex_learning_mode_switch_sets_override_and_db_marker(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", db_path)
    db = SessionDB(db_path)
    db.create_session("session-1", "telegram", model="gpt-5.5", user_id="7704114243")
    db.close()

    runner = _make_runner("session-1")
    event = _make_event("Learning Mode")

    async def fake_model_command(model_event):
        assert model_event.text == "/model gpt-5.4-mini --provider openai-codex"
        runner._session_model_overrides[runner._session_key_for_source(event.source)] = {
            "model": "gpt-5.4-mini",
            "provider": "openai-codex",
        }
        return "Model switched to `gpt-5.4-mini`"

    runner._session_key_for_source = lambda source: f"{source.platform.value}:{source.chat_id}"
    runner._handle_model_command = fake_model_command

    trigger = runner._cortex_mode_trigger_from_text(event.text)
    assert trigger is not None
    result = await runner._handle_cortex_mode_trigger(event, trigger)

    assert result.startswith("Learning mode active — using gpt-5.4-mini")
    assert runner._session_model_overrides["telegram:7704114243"]["model"] == "gpt-5.4-mini"

    db = SessionDB(db_path)
    try:
        row = db._conn.execute(
            "SELECT model, model_config FROM sessions WHERE id = ?", ("session-1",)
        ).fetchone()
    finally:
        db.close()
    assert row["model"] == "gpt-5.4-mini"
    assert '"provider": "openai-codex"' in row["model_config"]
    assert '"session_override": true' in row["model_config"]


@pytest.mark.asyncio
async def test_cortex_learning_mode_reports_model_command_errors_without_marker(tmp_path, monkeypatch):
    """Do not claim learning mode is active unless /model recorded an override."""
    db_path = tmp_path / "state.db"
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", db_path)
    db = SessionDB(db_path)
    db.create_session("session-1", "telegram", model="gpt-5.5", user_id="7704114243")
    db.close()

    runner = _make_runner("session-1")
    event = _make_event("Learning Mode")
    runner._session_key_for_source = lambda source: f"{source.platform.value}:{source.chat_id}"

    async def fake_failed_model_command(event):
        return "Error: Could not resolve credentials for provider 'OpenAI Codex'"

    runner._handle_model_command = fake_failed_model_command

    trigger = runner._cortex_mode_trigger_from_text(event.text)
    assert trigger is not None
    result = await runner._handle_cortex_mode_trigger(event, trigger)

    assert result.startswith("Learning mode active — using gpt-5.4-mini requested, but the switch failed")
    assert "Could not resolve credentials" in result
    assert runner._session_model_overrides == {}

    db = SessionDB(db_path)
    try:
        row = db._conn.execute(
            "SELECT model, model_config FROM sessions WHERE id = ?", ("session-1",)
        ).fetchone()
    finally:
        db.close()
    assert row["model"] == "gpt-5.5"
