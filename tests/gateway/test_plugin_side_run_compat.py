"""Opt-in side-run commands must not change legacy command busy semantics."""
from unittest.mock import Mock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.commands import should_bypass_active_session
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.mark.asyncio
async def test_legacy_busy_commands_keep_existing_queue_or_steer_path(monkeypatch):
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="legacy", source="user"), manager)
    handler = Mock(return_value="legacy")
    ctx.register_command("legacy-probe", handler)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda: manager)
    runner = object.__new__(GatewayRunner)
    source = SessionSource(Platform.TELEGRAM, "room", user_id="owner")
    event = MessageEvent(text="/legacy-probe argument", source=source)
    assert not should_bypass_active_session("legacy-probe")
    assert await runner._hm_busy_slash_or_photo(event, source, "parent") == (False, None)
    handler.assert_not_called()
    manager.unload()


@pytest.mark.asyncio
async def test_explicit_reject_is_inline_and_never_executes_handler(monkeypatch):
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="rejecting", source="user"), manager)
    handler = Mock(return_value="must not execute")
    ctx.register_command("reject-probe", handler, busy_policy="reject")
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda: manager)
    runner = object.__new__(GatewayRunner)
    source = SessionSource(Platform.TELEGRAM, "room", user_id="owner")
    event = MessageEvent(text="/reject-probe argument", source=source)
    assert should_bypass_active_session("reject-probe")
    handled, response = await runner._hm_busy_slash_or_photo(event, source, "parent")
    assert handled and "busy" in response
    handler.assert_not_called()
    manager.unload()
