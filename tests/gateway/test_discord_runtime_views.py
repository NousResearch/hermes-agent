"""Tests for Discord-native runtime status, help, commands, and whoami views."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(**overrides) -> SessionSource:
    base = {
        "platform": Platform.DISCORD,
        "user_id": "u1",
        "chat_id": "c1",
        "user_name": "alan",
        "chat_name": "Hermes / #general",
        "chat_type": "group",
        "thread_id": "thr-1",
        "chat_topic": "release room",
    }
    base.update(overrides)
    return SessionSource(**base)


def _make_event(
    text: str,
    *,
    source: SessionSource | None = None,
    session_source: SessionSource | None = None,
    target_source: SessionSource | None = None,
) -> MessageEvent:
    source = source or _make_source()
    metadata = {}
    if session_source is not None:
        metadata["session_source"] = session_source
    if target_source is not None:
        metadata["command_target_source"] = target_source
    return MessageEvent(
        text=text,
        source=source,
        message_id="m1",
        metadata=metadata,
    )


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
        }
    )
    discord_adapter = MagicMock()
    discord_adapter.get_activation_mode = MagicMock(return_value="always")
    discord_adapter.get_thread_binding = MagicMock(
        return_value=SimpleNamespace(
            chat_name="release room",
            idle_timeout_minutes=120,
            max_age_minutes=1440,
        )
    )
    runner.adapters = {
        Platform.DISCORD: discord_adapter,
        Platform.TELEGRAM: MagicMock(),
    }
    runner._voice_mode = {}
    runner._session_send_policies = {}
    runner._session_docks = {}
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
    runner._load_current_model_selection = lambda: {
        "current_model": "anthropic/claude-opus-4.6",
        "current_provider": "openrouter",
    }
    runner._resolve_model_runtime_details = lambda _provider: (
        {
            "provider": "openrouter",
            "api_mode": "responses",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "secret-key",
            "source": "OPENROUTER_API_KEY",
        },
        None,
    )
    runner._effective_model = None
    runner._effective_provider = None
    return runner


@pytest.mark.asyncio
async def test_discord_status_command_returns_rich_runtime_summary(monkeypatch):
    source = _make_source()
    slash_source = _make_source(session_namespace="slash:u1")
    session_key = build_session_key(slash_source)
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime(2026, 3, 19, 10, 0),
        updated_at=datetime(2026, 3, 19, 11, 15),
        platform=Platform.DISCORD,
        chat_type="group",
        input_tokens=220,
        output_tokens=110,
        cache_read_tokens=45,
        cache_write_tokens=12,
        total_tokens=387,
        last_prompt_tokens=144,
        estimated_cost_usd=0.0234,
        cost_status="estimated",
    )
    runner = _make_runner(session_entry)
    runner._running_agents[session_key] = MagicMock()
    runner._pending_messages[session_key] = {"text": "queued"}
    runner._pending_approvals[session_key] = {"command": "rm -rf /tmp/not-real"}
    runner._voice_mode[source.chat_id] = "all"
    runner._session_send_policies[session_key] = "off"
    runner._session_docks[session_key] = {
        "platform": "telegram",
        "chat_id": "-1001",
        "thread_id": None,
        "name": "Ops Feed",
    }

    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 200000,
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry.has_active_for_session",
        lambda key: key == session_key,
    )

    event = _make_event(
        "/status",
        source=source,
        session_source=slash_source,
        target_source=source,
    )
    result = await runner._handle_status_command(event)

    assert "📊 **Hermes Status**" in result
    assert "**Session**" in result
    assert "**Model**" in result
    assert "**Usage & Context**" in result
    assert "**Runtime**" in result
    assert "**Platforms**" in result
    assert "`slash:u1`" in result
    assert "`anthropic/claude-opus-4.6`" in result
    assert "Context Window: 200,000 tokens" in result
    assert "Pending Approval: Yes" in result
    assert "Background Processes: Yes" in result
    assert "Send Policy: `off`" in result
    assert "Dock Target: telegram:Ops Feed (`-1001`)" in result
    assert "Activation: `always`" in result
    assert "Focused Thread: yes (release room; idle 120m; max-age 1440m)" in result
    assert "Connected: discord, telegram" in result
    assert "secret-key" not in result


@pytest.mark.asyncio
async def test_discord_help_and_commands_are_distinct():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    runner = _make_runner(session_entry)
    event = _make_event("/help")

    help_result = await runner._handle_help_command(event)
    commands_result = await runner._handle_commands_command(_make_event("/commands"))

    assert help_result != commands_result
    assert "📖 **Hermes Help**" in help_result
    assert "Use `/commands` for the full command catalog." in help_result
    assert "**Quick Start**" in help_result
    assert "`/whoami` — Show the sender identity Hermes sees (alias: `/id`)" in help_result

    assert "🧭 **Hermes Command Catalog**" in commands_result
    assert "Use `/help` for the guided overview." in commands_result
    assert "**Session**" in commands_result
    assert "**Configuration**" in commands_result
    assert "`/models` — Open the Discord model picker or list models" in commands_result
    assert "`/whoami` — Show the sender identity Hermes sees (alias: `/id`)" in commands_result


@pytest.mark.asyncio
async def test_discord_whoami_includes_routing_context():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    runner = _make_runner(session_entry)
    target_source = _make_source(thread_id="thr-1", chat_topic="release room")
    session_source = _make_source(session_namespace="slash:u1")

    result = await runner._handle_whoami_command(
        _make_event(
            "/whoami",
            source=target_source,
            session_source=session_source,
            target_source=target_source,
        )
    )

    assert "👤 **Hermes Sees You As**" in result
    assert "• Platform: discord" in result
    assert "• User ID: `u1`" in result
    assert "• Chat ID: `c1`" in result
    assert "• Thread ID: `thr-1`" in result
    assert "• Chat Topic: release room" in result
    assert "• Session Namespace: `slash:u1`" in result


@pytest.mark.asyncio
async def test_id_alias_dispatches_to_discord_whoami():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    runner = _make_runner(session_entry)

    result = await runner._handle_message(_make_event("/id"))

    assert "👤 **Hermes Sees You As**" in result
