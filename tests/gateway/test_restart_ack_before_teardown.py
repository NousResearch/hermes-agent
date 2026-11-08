"""An idle /restart delivers its acknowledgement BEFORE teardown starts.

The ack used to be the command's return value, sent after ``request_restart`` had begun stopping the
gateway; on a real Bot API round trip the adapter disconnected first and the user saw nothing.
"""

import pytest

from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _idle_restart(monkeypatch, tmp_path, send):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)
    monkeypatch.setattr("gateway.restart.is_gateway_supervisor_process", lambda: False)
    runner, adapter = make_restart_runner()
    monkeypatch.setattr(adapter, "send", send)
    order = []
    runner.request_restart = lambda **kw: order.append("restart") or True
    return runner, order


@pytest.mark.asyncio
async def test_idle_restart_ack_is_delivered_before_teardown_starts(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)
    monkeypatch.setattr("gateway.restart.is_gateway_supervisor_process", lambda: False)
    runner, adapter = make_restart_runner()
    order = []

    async def send(chat_id, content, **kwargs):
        order.append(("send", content, runner._restart_requested))
        return SendResult(success=True, message_id="ack")

    monkeypatch.setattr(adapter, "send", send)
    runner.request_restart = lambda **kw: order.append(("restart", kw)) or True
    event = MessageEvent(text="/restart", message_type=MessageType.TEXT, source=make_restart_source(), message_id="m1")

    result = await runner._handle_message(event)

    assert [step[0] for step in order] == ["send", "restart"]
    assert "Restarting" in order[0][1] and order[0][2] is False  # sent while the gateway was still live
    assert not result  # already delivered: no second copy through the reply path


@pytest.mark.asyncio
async def test_idle_restart_ack_carries_the_ordinary_reply_routing(monkeypatch, tmp_path):
    """The early ack is routed like any command reply: a forum-topic reply goes by topic metadata
    with no reply anchor, and it is notify-worthy (not sent silently)."""
    sent = []

    async def send(chat_id, content, **kwargs):
        sent.append(kwargs)
        return SendResult(success=True, message_id="ack")

    runner, _order = _idle_restart(monkeypatch, tmp_path, send)
    source = make_restart_source(chat_id="-100777", chat_type="group", thread_id="42")
    event = MessageEvent(text="/restart", message_type=MessageType.TEXT, source=source, message_id="m1")

    await runner._handle_message(event)

    assert sent[0]["reply_to"] is None
    assert sent[0]["metadata"]["thread_id"] == "42"
    assert sent[0]["metadata"]["notify"] is True
