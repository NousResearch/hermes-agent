"""Tests for gateway /status behavior and token persistence."""

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
    runner._session_db = None
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


@pytest.mark.asyncio
async def test_status_command_reports_running_agent_without_interrupt(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=321,
    )
    runner = _make_runner(session_entry)
    running_agent = MagicMock()
    runner._running_agents[build_session_key(_make_source())] = running_agent

    result = await runner._handle_message(_make_event("/status"))

    assert "**Tokens:** 321" in result
    assert "**Agent Running:** Yes ⚡" in result
    running_agent.interrupt.assert_not_called()
    assert runner._pending_messages == {}


@pytest.mark.asyncio
async def test_handle_message_persists_agent_token_counts(monkeypatch):
    import gateway.run as gateway_run

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.session_store.load_transcript.return_value = [{"role": "user", "content": "earlier"}]
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 80,
            "input_tokens": 120,
            "output_tokens": 45,
            "model": "openai/test-model",
        }
    )

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("hello"))

    assert result == "ok"
    runner.session_store.update_session.assert_called_once_with(
        session_entry.session_key,
        input_tokens=120,
        output_tokens=45,
        cache_read_tokens=0,
        cache_write_tokens=0,
        last_prompt_tokens=80,
        model="openai/test-model",
        estimated_cost_usd=None,
        cost_status=None,
        cost_source=None,
        provider=None,
        base_url=None,
    )


@pytest.mark.asyncio
async def test_usage_command_includes_account_section(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    running_agent = MagicMock()
    running_agent.provider = "openai-codex"
    running_agent.base_url = "https://chatgpt.com/backend-api/codex"
    running_agent.api_key = "unused"
    running_agent.session_prompt_tokens = 100
    running_agent.session_completion_tokens = 25
    running_agent.session_total_tokens = 125
    running_agent.session_api_calls = 2
    running_agent.context_compressor = SimpleNamespace(
        last_prompt_tokens=100,
        context_length=1000,
        compression_count=0,
    )
    runner._running_agents[build_session_key(_make_source())] = running_agent

    monkeypatch.setattr(
        "gateway.run.fetch_account_usage",
        lambda provider, base_url=None, api_key=None: object(),
    )
    monkeypatch.setattr(
        "gateway.run.render_account_usage_lines",
        lambda snapshot, markdown=False: [
            "📈 **Account limits**",
            "Provider: openai-codex (Pro)",
            "Session: 85% remaining (15% used)",
        ],
    )

    result = await runner._handle_usage_command(_make_event("/usage"))

    assert "📊 **Session Token Usage**" in result
    assert "📈 **Account limits**" in result
    assert "Provider: openai-codex (Pro)" in result


@pytest.mark.asyncio
async def test_usage_command_uses_persisted_provider_when_agent_not_running(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.session_store.load_transcript.return_value = [{"role": "user", "content": "earlier"}]
    runner._session_db = MagicMock()
    runner._session_db.get_session.return_value = {
        "billing_provider": "openai-codex",
        "billing_base_url": "https://chatgpt.com/backend-api/codex",
    }

    calls = {}

    async def _fake_to_thread(fn, *args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return fn(*args, **kwargs)

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _fake_to_thread)
    monkeypatch.setattr(
        "gateway.run.fetch_account_usage",
        lambda provider, base_url=None, api_key=None: object(),
    )
    monkeypatch.setattr(
        "gateway.run.render_account_usage_lines",
        lambda snapshot, markdown=False: [
            "📈 **Account limits**",
            "Provider: openai-codex (Pro)",
        ],
    )

    result = await runner._handle_usage_command(_make_event("/usage"))

    assert calls["args"] == ("openai-codex",)
    assert calls["kwargs"]["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert "📊 **Session Info**" in result
    assert "📈 **Account limits**" in result
