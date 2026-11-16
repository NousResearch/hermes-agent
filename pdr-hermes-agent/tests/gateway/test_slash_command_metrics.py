"""Messaging-gateway slash commands reach shared metrics once each, tagged surface ``gateway``."""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.test_gateway_command_dispatch_minimal import _make_runner, _make_source


@pytest.mark.asyncio
async def test_gateway_counts_each_user_slash_command_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_events.record_slash_command", lambda **kw: calls.append(kw))
    runner, _adapter = _make_runner()
    runner._is_user_authorized_for_source = lambda _source: True
    runner._admit_bot_message_for_source = lambda _source: True
    runner._hm_dispatch_idle_commands = AsyncMock(return_value=(True, "ok"))

    def event(text, internal=False):
        return MessageEvent(text=text, message_type=MessageType.TEXT, source=_make_source(),
                            message_id="m1", internal=internal)

    typed = event("/status now")
    await runner._handle_message(typed)
    await runner._handle_message(typed)  # a queued busy-path event re-enters when drained
    await runner._handle_message(event("hello"))
    await runner._handle_message(event("/status", internal=True))

    assert calls == [{"command": "status", "surface": "gateway"}]
