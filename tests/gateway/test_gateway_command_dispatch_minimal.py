from datetime import datetime
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.platform_actions import SourceBoundPlatformActions
from hermes_cli.plugins import PluginCommandInvocation, PluginContext, PluginManager, PluginManifest


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
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m1",
        internal=True,
    )


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = _session_entry()
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
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
    runner._update_prompt_pending = {}
    runner._busy_input_mode = "interrupt"
    runner._draining = False
    runner._session_run_generation = {}
    runner._session_sources = {}
    runner._pending_native_image_paths_by_session = {}
    runner._background_tasks = {}
    runner._background_task_counter = 0
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._service_tier = None
    runner._fast_mode_by_session = {}
    runner._goal_state_by_session = {}
    runner._goal_runs_in_progress = set()
    runner._goal_queued_by_session = set()
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._should_send_telegram_lobby_reminder = lambda _source: False
    runner._check_slash_access = lambda _source, _command, _raw_args="": None
    runner._begin_session_run_generation = lambda _key: 1
    runner._release_running_agent_state = lambda key, run_generation=None: runner._running_agents.pop(key, None)
    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("command_text", ["/queue do this next", "/q do this next"])
async def test_idle_queue_sends_payload_as_next_turn(command_text):
    runner, _adapter = _make_runner()
    captured = {}

    async def fake_handle_message_with_agent(event, source, key, generation):
        captured["text"] = event.text
        captured["command"] = event.get_command()
        captured["source"] = source
        captured["key"] = key
        captured["generation"] = generation
        return {"final_response": "", "messages": []}

    runner._handle_message_with_agent = fake_handle_message_with_agent

    result = await runner._handle_message(_make_event(command_text))

    assert result == {"final_response": "", "messages": []}
    assert captured["text"] == "do this next"
    assert captured["command"] is None
    assert captured["source"] == _make_source()
    assert captured["key"] == build_session_key(_make_source())
    assert captured["generation"] == 1
    assert runner._running_agents == {}


@pytest.mark.asyncio
async def test_contextual_plugin_command_receives_bounded_immutable_gateway_context(monkeypatch):
    runner, adapter = _make_runner()
    adapter.is_connected = True
    adapter.normalize_source_identity_candidates = lambda source: (
        str(source.user_id),
        "normalized-alt",
    )
    runner._active_profile_name = lambda: "default"
    captured = {}
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="fixture", source="user"), manager)

    async def handler(raw_args, invocation):
        captured["raw_args"] = raw_args
        captured["invocation"] = invocation
        return "context-ok"

    ctx.register_command(
        "contextual",
        handler,
        with_context=True,
        access=lambda raw_args: "user" if raw_args == "status" else "admin",
        busy_policy="reject",
    )
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: manager
    )

    result = await runner._handle_message(_make_event("/contextual status"))

    invocation = captured["invocation"]
    assert result == "context-ok"
    assert captured["raw_args"] == "status"
    assert isinstance(invocation, PluginCommandInvocation)
    assert invocation.platform == "telegram"
    assert invocation.channel_id == "c1"
    assert invocation.thread_id is None
    assert invocation.message_id == "m1"
    assert invocation.chat_type == "dm"
    assert invocation.source_identity_candidates == ("u1", "normalized-alt")
    assert invocation.routed_profile == "default"
    assert isinstance(invocation.platform_actions, SourceBoundPlatformActions)
    assert not any(
        name in vars(invocation)
        for name in ("adapter", "client", "private_key", "credentials", "profile_path")
    )
    with pytest.raises(FrozenInstanceError):
        invocation.channel_id = "other"


def test_contextual_plugin_access_is_argument_aware_and_uses_normalized_identities(monkeypatch):
    runner, adapter = _make_runner()
    source = _make_source()
    source.chat_type = "group"
    source.user_id = "ordinary"
    source.user_id_alt = "ADMIN-ALT"
    adapter.normalize_source_identity_candidates = lambda _source: (
        "ordinary",
        "admin-alt",
    )
    runner.config.platforms[Platform.TELEGRAM].extra = {
        "group_allow_admin_from": ["admin-alt"],
    }
    runner._check_slash_access = runner.__class__._check_slash_access.__get__(runner)
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="fixture", source="user"), manager)
    ctx.register_command(
        "contextual",
        lambda raw_args, invocation: "ok",
        with_context=True,
        access=lambda raw_args: "user" if raw_args == "status" else "admin",
        busy_policy="reject",
    )
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: manager
    )

    assert runner._check_slash_access(source, "contextual", "status") is None
    assert runner._check_slash_access(source, "contextual", "listen always") is None

    adapter.normalize_source_identity_candidates = lambda _source: ("ordinary",)
    denial = runner._check_slash_access(source, "contextual", "listen always")
    assert "admin-only" in denial


@pytest.mark.asyncio
async def test_contextual_plugin_reject_while_busy_is_control_response(monkeypatch):
    runner, _adapter = _make_runner()
    handler = AsyncMock(return_value="should-not-run")
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="fixture", source="user"), manager)
    ctx.register_command(
        "contextual", handler, with_context=True, busy_policy="reject"
    )
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: manager
    )
    from hermes_cli.commands import resolve_gateway_command, should_bypass_active_session

    cmd_def = resolve_gateway_command("contextual")
    assert should_bypass_active_session("contextual") is True
    response = await runner._dispatch_busy_slash_command(
        _make_event("/contextual listen always"),
        cmd_def,
        build_session_key(_make_source()),
        _make_source(),
    )

    assert "can't run mid-turn" in response
    handler.assert_not_awaited()


def test_legacy_plugin_without_busy_policy_keeps_pre_metadata_queue_behavior(
    monkeypatch,
):
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="fixture", source="user"), manager)
    ctx.register_command("legacy", lambda raw_args: raw_args)
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: manager
    )
    from hermes_cli.commands import resolve_gateway_command, should_bypass_active_session

    assert manager._plugin_commands["legacy"]["busy_policy"] is None
    assert resolve_gateway_command("legacy").busy_policy == "reject"
    assert should_bypass_active_session("legacy") is False


@pytest.mark.asyncio
async def test_plugin_interrupt_then_dispatch_runs_plugin_handler(monkeypatch):
    runner, _adapter = _make_runner()
    handler = AsyncMock(return_value="interrupted-plugin-ok")
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="fixture", source="user"), manager)
    ctx.register_command(
        "interrupting",
        handler,
        access="user",
        busy_policy="interrupt_then_dispatch",
    )
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: manager
    )
    from hermes_cli.commands import (
        is_interrupt_then_dispatch,
        resolve_gateway_command,
        should_bypass_active_session,
    )

    cmd_def = resolve_gateway_command("interrupting")
    assert should_bypass_active_session("interrupting") is True
    assert is_interrupt_then_dispatch("interrupting") is True
    response = await runner._dispatch_busy_slash_command(
        _make_event("/interrupting now"),
        cmd_def,
        build_session_key(_make_source()),
        _make_source(),
    )

    assert response == "interrupted-plugin-ok"
    handler.assert_awaited_once_with("now")
