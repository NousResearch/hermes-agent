"""Tests for silent automatic session resets in the gateway."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _make_event(text="hello", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(adapter, session_entry, history=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner.config = GatewayConfig()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._set_session_env = lambda context, event=None: None
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    runner._deliver_media_from_response = AsyncMock()
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "history_offset": 0,
            "tools": [],
            "last_prompt_tokens": 0,
        }
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history or []
    runner.session_store.has_any_sessions.return_value = True
    return runner


@pytest.mark.asyncio
async def test_auto_reset_is_silent_for_user(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("privacy:\n  redact_pii: false\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_config_path", config_path)
    monkeypatch.setattr(gateway_run, "build_session_context", lambda source, config, entry: SimpleNamespace())
    monkeypatch.setattr(gateway_run, "build_session_context_prompt", lambda context, redact_pii=False: "ctx")

    adapter = SimpleNamespace(send=AsyncMock())
    entry = SessionEntry(
        session_key="telegram:67890:12345",
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        was_auto_reset=True,
        auto_reset_reason="idle",
        reset_had_activity=True,
    )
    runner = _make_runner(adapter, entry)

    response = await runner._handle_message_with_agent(
        _make_event(),
        _make_event().source,
        "telegram:67890:12345",
    )

    assert response == "ok"
    adapter.send.assert_not_awaited()
    assert entry.was_auto_reset is False
    assert entry.auto_reset_reason is None


@pytest.mark.asyncio
async def test_manual_reset_still_returns_visible_confirmation():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = MagicMock()
    runner.session_store._entries = {
        "telegram:67890:12345": SimpleNamespace(session_id="old-session")
    }
    runner.session_store.reset_session.return_value = SimpleNamespace(session_id="new-session")
    runner._background_tasks = set()
    runner._evict_cached_agent = MagicMock()
    runner._async_flush_memories = AsyncMock()
    runner._format_session_info = lambda: ""
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    result = await runner._handle_reset_command(_make_event(text="/reset"))

    assert result == "✨ Session reset! Starting fresh."
