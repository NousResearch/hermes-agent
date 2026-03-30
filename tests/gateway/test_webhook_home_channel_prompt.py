"""Regression tests for webhook session onboarding behavior."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(platform: Platform) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(platform: Platform) -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=_make_source(platform),
        message_id="m1",
    )


def _make_runner(platform: Platform) -> tuple:
    from gateway.run import GatewayRunner

    source = _make_source(platform)
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    runner.adapters = {platform: adapter}
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
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._model = "openai/gpt-5.4"
    runner._base_url = "https://openrouter.ai/api/v1"
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._clear_session_env = lambda: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._has_setup_skill = lambda: False
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "already_sent": True,
            "model": "openai/gpt-5.4",
        }
    )
    return runner, adapter


def test_webhook_sessions_skip_home_channel_prompt(monkeypatch):
    monkeypatch.delenv("WEBHOOK_HOME_CHANNEL", raising=False)
    runner, adapter = _make_runner(Platform.WEBHOOK)

    result = asyncio.run(runner._handle_message(_make_event(Platform.WEBHOOK)))

    assert result is None
    adapter.send.assert_not_called()


def test_non_webhook_sessions_still_get_home_channel_prompt(monkeypatch):
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    runner, adapter = _make_runner(Platform.TELEGRAM)

    result = asyncio.run(runner._handle_message(_make_event(Platform.TELEGRAM)))

    assert result is None
    adapter.send.assert_called_once()
    sent_text = adapter.send.await_args.args[1]
    assert "No home channel is set for Telegram" in sent_text
