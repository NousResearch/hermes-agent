import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _event(text="/context-cmd now"):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
        user_name="User",
        chat_type="dm",
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _register_context_command(monkeypatch, *, active_session_bypass=False):
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="example", key="example"), manager)
    seen = {}

    def handler(raw_args, *, event=None, gateway=None, source=None, session_key=None):
        seen.update(
            raw_args=raw_args,
            event=event,
            gateway=gateway,
            source=source,
            session_key=session_key,
        )
        return "plugin-ok"

    ctx.register_command(
        "context-cmd",
        handler,
        wants_context=True,
        active_session_bypass=active_session_bypass,
    )
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda force=False: manager)
    return seen


@pytest.mark.asyncio
async def test_gateway_dispatches_plugin_command_with_context(monkeypatch):
    seen = _register_context_command(monkeypatch)
    runner = object.__new__(GatewayRunner)
    event = _event()

    result = await runner._dispatch_plugin_command("context-cmd", event, "session-1")

    assert result == "plugin-ok"
    assert seen["raw_args"] == "now"
    assert seen["event"] is event
    assert seen["gateway"] is runner
    assert seen["source"] is event.source
    assert seen["session_key"] == "session-1"


@pytest.mark.asyncio
async def test_busy_path_plugin_command_bypasses_active_session_when_registered(monkeypatch):
    _register_context_command(monkeypatch, active_session_bypass=True)
    runner = object.__new__(GatewayRunner)
    event = _event()

    result = await runner._dispatch_active_session_plugin_command("context-cmd", event, "session-1")

    assert result == "plugin-ok"


@pytest.mark.asyncio
async def test_busy_path_plugin_command_without_bypass_falls_through(monkeypatch):
    _register_context_command(monkeypatch, active_session_bypass=False)
    runner = object.__new__(GatewayRunner)
    event = _event()

    result = await runner._dispatch_active_session_plugin_command("context-cmd", event, "session-1")

    assert result is None
