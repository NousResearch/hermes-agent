"""Tests for gateway /today behavior."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key



def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )



def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
    )



def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner



def _write_todo_state(base_dir):
    todo_dir = base_dir / "hermes-daily-state"
    todo_dir.mkdir(parents=True, exist_ok=True)
    todo_path = todo_dir / "todo-state.json"
    todo_path.write_text(
        """{
  \"version\": 1,
  \"updated_at\": \"2026-04-19T20:08:00+08:00\",
  \"pending\": [
    {\"id\": \"p1\", \"title\": \"Continue benchmark from checkpoint 482\", \"status\": \"pending\"}
  ],
  \"active\": [
    {\"id\": \"a1\", \"title\": \"Stabilize today's todo bridge\", \"status\": \"active\"}
  ],
  \"resolved_recent\": [],
  \"archive\": []
}
""",
        encoding="utf-8",
    )
    return todo_path


@pytest.mark.asyncio
async def test_today_command_returns_snapshot_and_queues_note(monkeypatch, tmp_path):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner = _make_runner(session_entry)
    hermes_home = tmp_path / ".hermes"
    _write_todo_state(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = await runner._handle_message(_make_event("/today"))

    assert "Today Todo" in result
    assert "Active (1)" in result
    assert "Stabilize today's todo bridge" in result
    assert "Pending (1)" in result
    note = runner._pending_model_notes[session_entry.session_key]
    assert "today todo snapshot" in note.lower()
    assert "Stabilize today's todo bridge" in note
    assert "Continue benchmark from checkpoint 482" in note
