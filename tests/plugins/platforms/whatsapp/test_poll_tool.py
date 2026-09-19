import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import SendResult
from tools.registry import registry


def _discover_poll_tool():
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager.discover_and_load()
    entry = registry.get_entry("whatsapp_send_poll")
    assert entry is not None
    return manager, entry, sys.modules[entry.handler.__module__]


def test_deferred_plugin_discovery_registers_poll_tool():
    from toolsets import resolve_toolset

    manager, entry, _ = _discover_poll_tool()

    loaded = manager._plugins.get("whatsapp-platform")
    assert loaded is not None
    assert loaded.deferred is True
    assert entry.name == "whatsapp_send_poll"
    assert "whatsapp_send_poll" in resolve_toolset("whatsapp")


@pytest.fixture
def registered_poll_tool(monkeypatch):
    _, entry, tool_module = _discover_poll_tool()
    authorization = MagicMock(
        return_value="native adapter must bypass relay classification"
    )
    monkeypatch.setattr(tool_module, "_authorize_relay_target", authorization)
    send = AsyncMock(
        return_value=SendResult(success=True, message_id="poll-message-id")
    )

    class Adapter:
        send_poll = send

    runner = object()
    monkeypatch.setattr(tool_module, "_live_poll_adapter", lambda: (runner, Adapter()))

    async def dispatch(_runner, make_coro, _log_message):
        assert _runner is runner
        return await make_coro()

    monkeypatch.setattr(tool_module, "_dispatch_on_gateway_loop", dispatch)
    yield tool_module, send, authorization


def test_registered_poll_tool_dispatches_multiple_choice(registered_poll_tool):
    _, send, authorization = registered_poll_tool

    raw = registry.dispatch(
        "whatsapp_send_poll",
        {
            "chat_id": "123456789-987654321@g.us",
            "question": "  Draft time?  ",
            "options": [" 7 PM ", "8 PM"],
            "allow_multiple": True,
        },
    )
    assert isinstance(raw, str)
    result = json.loads(raw)

    assert result == {
        "success": True,
        "platform": "whatsapp",
        "chat_id": "123456789-987654321@g.us",
        "message_id": "poll-message-id",
    }
    send.assert_awaited_once_with(
        "123456789-987654321@g.us",
        "Draft time?",
        ["7 PM", "8 PM"],
        selectable_count=2,
    )
    # The adapter came from the active profile and is the transport used below;
    # the default-profile relay classifier must not be consulted.
    authorization.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "mode", "needle"),
    [
        ({"chat_id": "not-a-jid"}, "validation", "valid WhatsApp chat ID"),
        ({"chat_id": "１２３@g.us"}, "validation", "valid WhatsApp chat ID"),
        ({"chat_id": "١٢٣@g.us"}, "validation", "valid WhatsApp chat ID"),
        ({"options": ["Only one"]}, "validation", "at least two"),
        ({"options": ["A", "a"]}, "validation", "unique"),
        ({"options": ["A", None]}, "validation", "must be a string"),
        ({"question": 42}, "validation", "question must be a string"),
        ({"allow_multiple": "yes"}, "validation", "must be a boolean"),
        ({}, "relay_denial", "unattested relay target"),
        ({}, "missing_id", "without a message ID"),
    ],
)
def test_poll_tool_fails_closed(
    registered_poll_tool, monkeypatch, overrides, mode, needle
):
    tool_module, send, authorization = registered_poll_tool
    args = {
        "chat_id": "120363000000000000@g.us",
        "question": "Question?",
        "options": ["A", "B"],
    }
    args.update(overrides)

    if mode == "relay_denial":
        monkeypatch.setattr(tool_module, "_live_poll_adapter", lambda: (None, None))
        authorization.return_value = "unattested relay target"
    elif mode == "missing_id":
        send.return_value = SendResult(success=True, message_id=None)

    raw = registry.dispatch("whatsapp_send_poll", args)
    assert isinstance(raw, str)
    result = json.loads(raw)

    assert result.get("success") is not True
    assert needle in result["error"]
