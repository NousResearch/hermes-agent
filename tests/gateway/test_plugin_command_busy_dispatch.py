"""Plugin slash commands on a busy gateway session: ``register_command(busy_policy="dispatch")``
runs the handler mid-turn (and hands a ``gateway_context`` to a handler that declares one);
without it the command stays on the text path, exactly as before."""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import plugins as plugins_mod
from hermes_cli.commands import should_bypass_active_session
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _runner_with_plugin_commands(monkeypatch):
    mgr = PluginManager()
    ctx = PluginContext(PluginManifest(name="probe", source="user"), mgr)
    seen = {}

    def _status(raw_args, *, gateway_context):
        seen["context"] = gateway_context
        return f"status {raw_args}"

    ctx.register_command("probe-status", _status, busy_policy="dispatch")
    ctx.register_command("probe-plain", lambda raw_args: "plain")
    monkeypatch.setattr(plugins_mod, "_ensure_plugins_discovered", lambda force=False: mgr)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner._draining = False
    runner.session_store = None
    adapter = object()
    runner._delivery_adapter_for = lambda source: adapter
    return runner, adapter, seen


def _event(text):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", user_name="t", chat_type="dm")
    return MessageEvent(text=text, source=source, message_id="m1"), source


@pytest.mark.asyncio
async def test_dispatch_policy_runs_mid_turn_with_gateway_context(monkeypatch):
    runner, adapter, seen = _runner_with_plugin_commands(monkeypatch)
    event, source = _event("/probe_status now")

    assert should_bypass_active_session("probe_status") is True
    handled, result = await runner._hm_busy_slash_or_photo(event, source, "qk")

    assert (handled, result) == (True, "status now")
    context = seen["context"]
    assert context.adapter is adapter
    assert context.session_store is runner.async_session_store
    assert context.is_authorized == runner._is_user_authorized_for_source


@pytest.mark.asyncio
async def test_default_policy_stays_on_the_text_path(monkeypatch):
    runner, _adapter, _seen = _runner_with_plugin_commands(monkeypatch)
    event, source = _event("/probe-plain")

    assert should_bypass_active_session("probe-plain") is False
    assert await runner._hm_busy_slash_or_photo(event, source, "qk") == (False, None)
