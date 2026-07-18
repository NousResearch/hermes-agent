import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    TurnControlContext,
    TurnDirective,
    invoke_turn_controllers,
)

from tests.gateway.test_run_cleanup_progress import (
    CleanupCaptureAdapter,
    _make_runner,
)


def _plugin_manager() -> PluginManager:
    mgr = PluginManager()
    mgr._discovered = True
    return mgr


@pytest.mark.asyncio
async def test_gateway_context_aware_plugin_command_receives_session_id(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:dm:-1001"
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(session_id="gw-sess")

    mgr = _plugin_manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)
    seen = []

    def handler(command_ctx, raw_args):
        seen.append((command_ctx.surface, command_ctx.session_id, command_ctx.task_id, raw_args))
        return "started"

    ctx.register_command("ctxcmd", handler, context_aware=True)
    event = gateway_run.MessageEvent(
        text="/ctxcmd start",
        message_type=gateway_run.MessageType.TEXT,
        source=source,
    )

    with patch("hermes_cli.plugins._plugin_manager", mgr):
        result = await runner._dispatch_plugin_command_event(
            command="ctxcmd",
            event=event,
            source=source,
            session_key=session_key,
        )

    assert result == "started"
    assert seen == [("gateway", "gw-sess", session_key, "start")]


@pytest.mark.asyncio
async def test_busy_safe_plugin_control_dispatches_without_interrupt(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        user_id="user-1",
        chat_type="dm",
    )
    session_key = runner._session_key_for_source(source)
    runner._is_user_authorized = lambda _source: True
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._update_prompt_pending = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(
        session_id="busy-sess"
    )
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "seconds_since_activity": 0,
        "last_activity_desc": "tool",
        "api_call_count": 1,
        "max_iterations": 10,
    }
    runner._running_agents[session_key] = running_agent
    runner._running_agents_ts = {session_key: __import__("time").time()}

    mgr = _plugin_manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)
    seen = []

    def handler(command_ctx, raw_args):
        seen.append((command_ctx.session_id, raw_args))
        return "paused"

    ctx.register_command(
        "control",
        handler,
        context_aware=True,
        busy_safe_subcommands=("status", "pause", "clear"),
    )
    event = gateway_run.MessageEvent(
        text="/control pause",
        message_type=gateway_run.MessageType.TEXT,
        source=source,
    )

    with patch("hermes_cli.plugins._plugin_manager", mgr):
        result = await runner._handle_message(event)

    assert result == "paused"
    assert seen == [("busy-sess", "pause")]
    running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_busy_unsafe_plugin_subcommand_is_rejected_without_interrupt(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        user_id="user-1",
        chat_type="dm",
    )
    session_key = runner._session_key_for_source(source)
    runner._is_user_authorized = lambda _source: True
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._update_prompt_pending = {}
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "seconds_since_activity": 0,
        "last_activity_desc": "tool",
        "api_call_count": 1,
        "max_iterations": 10,
    }
    runner._running_agents[session_key] = running_agent
    runner._running_agents_ts = {session_key: __import__("time").time()}

    mgr = _plugin_manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)
    called = []
    ctx.register_command(
        "control",
        lambda _ctx, raw: called.append(raw) or "started",
        context_aware=True,
        busy_safe_subcommands=("status", "pause", "clear"),
    )
    event = gateway_run.MessageEvent(
        text="/control start new mission",
        message_type=gateway_run.MessageType.TEXT,
        source=source,
    )

    with patch("hermes_cli.plugins._plugin_manager", mgr):
        result = await runner._handle_message(event)

    assert "can't run mid-turn" in result
    assert called == []
    running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_plugin_command_exception_isolated(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(session_id="gw-sess")

    mgr = _plugin_manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)
    ctx.register_command("boom", lambda _raw: (_ for _ in ()).throw(RuntimeError("boom")))
    event = gateway_run.MessageEvent(
        text="/boom",
        message_type=gateway_run.MessageType.TEXT,
        source=source,
    )

    with patch("hermes_cli.plugins._plugin_manager", mgr):
        result = await runner._dispatch_plugin_command_event(
            command="boom",
            event=event,
            source=source,
            session_key="agent:main:telegram:group:-1001",
        )
    assert result == "Plugin command failed."


@pytest.mark.asyncio
async def test_stale_gateway_enqueue_followup_rejects_rotated_session(monkeypatch):
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:dm:-1001"
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(session_id="new-sess")

    ok = await runner._enqueue_plugin_followup(
        prompt="continue",
        source=source,
        session_id="old-sess",
        session_key=session_key,
    )

    assert ok is False
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_turn_directive_state_version_is_monotonic_per_controller_and_session():
    mgr = _plugin_manager()
    ctx = PluginContext(PluginManifest(name="plug"), mgr)
    current = {"version": 2, "dedupe": "key-a"}

    def controller(_ctx):
        return TurnDirective(
            action="continue",
            continuation_prompt="next",
            dedupe_key=current["dedupe"],
            state_version=current["version"],
        )

    ctx.register_turn_controller("versioned", controller)

    def turn_context(session_id):
        return TurnControlContext(
            surface="gateway",
            session_id=session_id,
            platform="telegram",
            source=None,
            task_id=session_id,
            turn_id=f"{session_id}:turn",
            user_message="hi",
            final_response="done",
            interrupted=False,
            background_processes=[],
        )

    with patch("hermes_cli.plugins._plugin_manager", mgr):
        assert await invoke_turn_controllers(turn_context("session-a")) is not None
        current.update(version=1, dedupe="key-b")
        assert await invoke_turn_controllers(turn_context("session-a")) is None
        current.update(version=3, dedupe="key-c")
        assert await invoke_turn_controllers(turn_context("session-a")) is not None
        current.update(version=1, dedupe="key-d")
        assert await invoke_turn_controllers(turn_context("session-b")) is not None
