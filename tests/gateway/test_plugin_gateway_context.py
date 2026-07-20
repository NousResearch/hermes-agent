"""Plugin gateway command provenance and compatibility contracts."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _event(text: str = "/hermes_control status") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="123",
        chat_id="456",
        chat_type="dm",
        thread_id="topic-7",
        is_bot=False,
        profile="operator",
    )
    return MessageEvent(
        text=text,
        source=source,
        message_id="99",
        platform_update_id=101,
        timestamp=datetime(2026, 7, 20, 12, 30, tzinfo=timezone.utc),
        metadata={"mutable": {"secret": True}},
        raw_message=MagicMock(token="must-not-leak"),
    )


def _runner():
    from gateway.run import GatewayRunner

    return object.__new__(GatewayRunner)


def _integrated_runner(*, authorized: bool = True):
    runner = _runner()
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner.session_store = MagicMock()
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._is_user_authorized = lambda _source: authorized
    return runner


def _manager(handler, *, gateway_context: bool) -> PluginManager:
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="control", source="user"), manager)
    context.register_command(
        "hermes-control",
        handler,
        gateway_context=gateway_context,
    )
    return manager


@pytest.mark.asyncio
async def test_legacy_handler_receives_only_raw_args():
    received = []
    manager = _manager(lambda args: received.append(args) or "ok", gateway_context=False)

    with patch("hermes_cli.plugins._plugin_manager", manager):
        handled, response = await _runner()._dispatch_plugin_command(
            _event(), "hermes_control"
        )

    assert (handled, response) == (True, "ok")
    assert received == ["status"]


@pytest.mark.asyncio
async def test_opted_in_handler_gets_frozen_scalar_snapshot():
    captured = []

    async def handler(args, *, context):
        captured.append((args, context))
        return "accepted"

    event = _event("/hermes_control --user-id forged --chat-id forged")
    manager = _manager(handler, gateway_context=True)

    with patch("hermes_cli.plugins._plugin_manager", manager):
        handled, response = await _runner()._dispatch_plugin_command(
            event, "hermes_control"
        )

    assert (handled, response) == (True, "accepted")
    raw_args, context = captured[0]
    assert raw_args == "--user-id forged --chat-id forged"
    assert context.platform == "telegram"
    assert context.user_id == "123"
    assert context.chat_id == "456"
    assert context.thread_id == "topic-7"
    assert context.message_id == "99"
    assert context.platform_update_id == 101
    assert context.command_name == "hermes-control"
    assert context.received_at == "2026-07-20T12:30:00+00:00"
    assert not hasattr(context, "raw_message")
    assert not hasattr(context, "metadata")
    assert not hasattr(context, "text")
    assert not hasattr(context, "profile")
    with pytest.raises(FrozenInstanceError):
        context.user_id = "forged"

    event.source.user_id = "changed"
    event.metadata["mutable"]["secret"] = False
    assert context.user_id == "123"


@pytest.mark.asyncio
async def test_handler_failure_is_handled_without_leaking_exception():
    def handler(_args):
        raise RuntimeError("token=super-secret")

    manager = _manager(handler, gateway_context=False)
    with patch("hermes_cli.plugins._plugin_manager", manager):
        handled, response = await _runner()._dispatch_plugin_command(
            _event(), "hermes-control"
        )

    assert handled is True
    assert response == "Plugin command failed. Check the gateway logs for details."
    assert "super-secret" not in response


@pytest.mark.asyncio
async def test_registered_none_result_is_still_handled():
    manager = _manager(lambda _args: None, gateway_context=False)
    with patch("hermes_cli.plugins._plugin_manager", manager):
        assert await _runner()._dispatch_plugin_command(
            _event(), "hermes-control"
        ) == (True, None)


@pytest.mark.asyncio
async def test_active_agent_dispatches_plugin_without_interrupting():
    received = []
    manager = _manager(lambda args: received.append(args) or "active-ok", gateway_context=False)
    runner = _integrated_runner()
    event = _event()
    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {"seconds_since_activity": 0}
    runner._running_agents[session_key] = running_agent

    with patch("hermes_cli.plugins._plugin_manager", manager):
        response = await runner._handle_message(event)

    assert response == "active-ok"
    assert received == ["status"]
    running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_active_agent_plugin_obeys_slash_access_control():
    handler = MagicMock(return_value="must-not-run")
    manager = _manager(handler, gateway_context=True)
    runner = _integrated_runner()
    runner._check_slash_access = MagicMock(return_value="Command denied")
    event = _event()
    runner._running_agents[build_session_key(event.source)] = MagicMock()

    with patch("hermes_cli.plugins._plugin_manager", manager):
        response = await runner._handle_message(event)

    assert response == "Command denied"
    runner._check_slash_access.assert_called_once_with(event.source, "hermes-control")
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_group_sender_never_invokes_plugin():
    handler = MagicMock(return_value="must-not-run")
    manager = _manager(handler, gateway_context=True)
    runner = _integrated_runner(authorized=False)
    event = _event()
    event.source.chat_type = "group"

    with patch("hermes_cli.plugins._plugin_manager", manager):
        response = await runner._handle_message(event)

    assert response is None
    handler.assert_not_called()
