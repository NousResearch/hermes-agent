"""session_rotation: a conversation that grew too large, or went quiet, starts fresh on the
next message (off by default)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    s = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    s._db = SessionDB(db_path=tmp_path / "state.db")
    return s


def _rotation(monkeypatch, **values):
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"session_rotation": values})


SOURCE = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="u1")


def test_off_by_default_keeps_the_conversation(store, monkeypatch):
    _rotation(monkeypatch)
    first = store.get_or_create_session(SOURCE)
    store.update_session(first.session_key, last_prompt_tokens=500_000)
    assert store.get_or_create_session(SOURCE).session_id == first.session_id


def test_a_conversation_too_large_starts_fresh(store, monkeypatch):
    _rotation(monkeypatch, max_prompt_tokens=40_000)
    first = store.get_or_create_session(SOURCE)
    store.update_session(first.session_key, last_prompt_tokens=39_000)
    assert store.get_or_create_session(SOURCE).session_id == first.session_id
    store.update_session(first.session_key, last_prompt_tokens=41_000)
    second = store.get_or_create_session(SOURCE)
    assert second.session_id != first.session_id
    assert second.was_auto_reset and second.auto_reset_reason == "rotated"


def test_a_quiet_conversation_starts_fresh(store, monkeypatch):
    _rotation(monkeypatch, idle_hours=4)
    first = store.get_or_create_session(SOURCE)
    store.update_session(first.session_key, last_prompt_tokens=12_000)
    store._entries[first.session_key].updated_at = datetime.now() - timedelta(hours=5)
    second = store.get_or_create_session(SOURCE)
    assert second.session_id != first.session_id and second.auto_reset_reason == "rotated"


def test_the_configured_note_reaches_the_agent(monkeypatch):
    from gateway.session_lifecycle import rotation_config
    _rotation(monkeypatch, note="  [closed]  ", max_prompt_tokens="oops")
    assert rotation_config() == {"max_prompt_tokens": 0.0, "idle_hours": 0.0, "note": "[closed]"}
