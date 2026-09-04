import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.context_engine import _EngineCollector
from tests.gateway.test_run_cleanup_progress import CleanupCaptureAdapter, _make_runner
from tests.gateway.test_slash_access_dispatch import (
    _make_event as _make_access_event,
    _make_runner as _make_access_runner,
    _make_source as _make_access_source,
)


def _plugin_manager() -> PluginManager:
    manager = PluginManager()
    manager._discovered = True
    return manager


def _active_adapter(runner, adapter, source):
    session_key = runner._session_key_for_source(source)
    runner._is_user_authorized = lambda _source: True
    runner._check_slash_access = lambda _source, _command: None
    runner._effective_busy_input_mode = lambda _source: "interrupt"
    runner._draining = False
    adapter.set_message_handler(AsyncMock(return_value=None))
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = asyncio.current_task()
    return session_key


def test_plugin_command_records_normalized_busy_safe_subcommands():
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)

    context.register_command(
        "control",
        lambda _raw: "ok",
        busy_safe_subcommands=("", " STATUS ", "status", "Pause"),
    )

    assert manager._plugin_commands["control"]["busy_safe_subcommands"] == (
        "",
        "status",
        "pause",
    )


def test_plugin_command_rejects_multitoken_busy_safe_entry():
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)

    with pytest.raises(ValueError, match="single tokens"):
        context.register_command(
            "control",
            lambda _raw: "ok",
            busy_safe_subcommands=("not safe",),
        )


def test_busy_safe_subcommands_is_keyword_only():
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)

    with pytest.raises(TypeError):
        context.register_command(
            "control",
            lambda _raw: "ok",
            "",
            "",
            None,
            ("",),
        )


def test_context_engine_collector_forwards_busy_safe_metadata():
    manager = _plugin_manager()
    collector = _EngineCollector("memory-engine")

    with patch("hermes_cli.plugins._plugin_manager", manager):
        collector.register_command(
            "memory-control",
            lambda _raw: "ok",
            args_hint="[status]",
            argument_mode="mixed",
            busy_safe_subcommands=("", "STATUS"),
        )

    assert manager._plugin_commands["memory-control"] == {
        "handler": manager._plugin_commands["memory-control"]["handler"],
        "description": "Context engine command",
        "plugin": "context-engine:memory-engine",
        "plugin_key": "context-engine:memory-engine",
        "args_hint": "[status]",
        "argument_mode": "mixed",
        "busy_safe_subcommands": ("", "status"),
    }


@pytest.mark.asyncio
async def test_cold_plugin_command_applies_slash_access_control():
    runner = _make_access_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)
    called = []
    context.register_command(
        "control",
        lambda raw: called.append(raw) or "secret",
    )

    with patch("hermes_cli.plugins._plugin_manager", manager):
        result = await runner._handle_message(
            _make_access_event(
                "/control status",
                _make_access_source(user_id="999"),
            )
        )

    assert called == []
    assert result is not None
    assert "⛔" in result
    assert "/control is admin-only here" in result


@pytest.mark.asyncio
async def test_cold_plugin_command_keeps_unrestricted_back_compat():
    runner = _make_access_runner()
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)

    async def handler(raw_args):
        await asyncio.sleep(0)
        return f"cold:{raw_args}"

    context.register_command("control", handler)

    with patch("hermes_cli.plugins._plugin_manager", manager):
        result = await runner._handle_message(
            _make_access_event(
                "/control status",
                _make_access_source(user_id="999"),
            )
        )

    assert result == "cold:status"


@pytest.mark.asyncio
async def test_active_adapter_dispatches_bare_busy_safe_plugin_command():
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        user_id="111",
        chat_type="dm",
    )
    session_key = _active_adapter(runner, adapter, source)
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)
    seen = []
    context.register_command(
        "control",
        lambda raw: seen.append(raw) or "healthy",
        busy_safe_subcommands=("",),
    )

    with patch("hermes_cli.plugins._plugin_manager", manager):
        await adapter.handle_message(
            gateway_run.MessageEvent(
                text="/control",
                message_type=gateway_run.MessageType.TEXT,
                source=source,
            )
        )

    assert seen == [""]
    assert [item["content"] for item in adapter.sent] == ["healthy"]
    assert adapter.sent[0]["metadata"]["notify"] is True
    assert adapter._pending_messages == {}
    assert session_key in adapter._active_sessions


@pytest.mark.asyncio
async def test_active_adapter_awaits_busy_safe_plugin_command():
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        user_id="111",
        chat_type="dm",
    )
    _active_adapter(runner, adapter, source)
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)

    async def handler(raw_args):
        await asyncio.sleep(0)
        return f"paused:{raw_args}"

    context.register_command(
        "control",
        handler,
        busy_safe_subcommands=("pause",),
    )

    with patch("hermes_cli.plugins._plugin_manager", manager):
        await adapter.handle_message(
            gateway_run.MessageEvent(
                text="/control pause",
                message_type=gateway_run.MessageType.TEXT,
                source=source,
            )
        )

    assert [item["content"] for item in adapter.sent] == ["paused:pause"]
    assert adapter.sent[0]["metadata"]["notify"] is True
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_active_adapter_rejects_unsafe_plugin_verb_without_queueing():
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        user_id="111",
        chat_type="dm",
    )
    session_key = _active_adapter(runner, adapter, source)
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)
    called = []
    context.register_command(
        "control",
        lambda raw: called.append(raw) or "started",
        busy_safe_subcommands=("status", "pause"),
    )

    with patch("hermes_cli.plugins._plugin_manager", manager):
        await adapter.handle_message(
            gateway_run.MessageEvent(
                text="/control start new mission",
                message_type=gateway_run.MessageType.TEXT,
                source=source,
            )
        )

    assert called == []
    assert len(adapter.sent) == 1
    assert "can't run mid-turn" in adapter.sent[0]["content"]
    assert adapter.sent[0]["metadata"]["notify"] is True
    assert adapter._pending_messages == {}
    assert session_key in adapter._active_sessions


@pytest.mark.asyncio
async def test_legacy_plugin_command_keeps_existing_busy_path():
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        user_id="111",
        chat_type="dm",
    )
    _active_adapter(runner, adapter, source)
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)
    called = []
    context.register_command(
        "control",
        lambda raw: called.append(raw) or "legacy",
    )

    with patch("hermes_cli.plugins._plugin_manager", manager):
        await adapter.handle_message(
            gateway_run.MessageEvent(
                text="/control status",
                message_type=gateway_run.MessageType.TEXT,
                source=source,
            )
        )

    assert called == []
    adapter._message_handler.assert_not_awaited()
    assert len(adapter.sent) == 1
    assert "Interrupting current task" in adapter.sent[0]["content"]
    assert list(adapter._pending_messages) == [runner._session_key_for_source(source)]


@pytest.mark.asyncio
async def test_active_adapter_rejects_unsafe_verb_after_runner_cleanup():
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        user_id="111",
        chat_type="dm",
    )
    _active_adapter(runner, adapter, source)
    runner._running_agents.clear()
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)
    called = []
    context.register_command(
        "control",
        lambda raw: called.append(raw) or "deleted",
        busy_safe_subcommands=("status",),
    )

    with patch("hermes_cli.plugins._plugin_manager", manager):
        await adapter.handle_message(
            gateway_run.MessageEvent(
                text="/control delete",
                message_type=gateway_run.MessageType.TEXT,
                source=source,
            )
        )

    assert called == []
    assert len(adapter.sent) == 1
    assert "can't run mid-turn" in adapter.sent[0]["content"]
    assert adapter.sent[0]["metadata"]["notify"] is True
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_active_adapter_applies_slash_access_before_busy_dispatch():
    gateway_run = importlib.import_module("gateway.run")
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        user_id="999",
        chat_type="dm",
    )
    _active_adapter(runner, adapter, source)
    runner._check_slash_access = lambda _source, command: (
        f"⛔ /{command} is admin-only here."
    )
    manager = _plugin_manager()
    context = PluginContext(PluginManifest(name="plug"), manager)
    called = []
    context.register_command(
        "control",
        lambda raw: called.append(raw) or "status",
        busy_safe_subcommands=("",),
    )

    with patch("hermes_cli.plugins._plugin_manager", manager):
        await adapter.handle_message(
            gateway_run.MessageEvent(
                text="/control",
                message_type=gateway_run.MessageType.TEXT,
                source=source,
            )
        )

    assert called == []
    assert [item["content"] for item in adapter.sent] == [
        "⛔ /control is admin-only here."
    ]
    assert adapter.sent[0]["metadata"]["notify"] is True
    assert adapter._pending_messages == {}
