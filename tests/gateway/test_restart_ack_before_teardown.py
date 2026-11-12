"""An idle /restart delivers its acknowledgement BEFORE teardown starts.

The ack used to be the command's return value, sent after ``request_restart`` had begun stopping the
gateway; on a real Bot API round trip the adapter disconnected first and the user saw nothing.
"""

import asyncio

import pytest

from hermes_agent.gateway.platforms.base import SendResult
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _idle_restart(monkeypatch, tmp_path, send):
    monkeypatch.setattr("hermes_agent.gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("hermes_agent.gateway.restart.is_container_restart_context", lambda: False)
    monkeypatch.setattr("hermes_agent.gateway.restart.is_gateway_supervisor_process", lambda: False)
    runner, adapter = make_restart_runner()
    monkeypatch.setattr(adapter, "send", send)
    order = []
    runner.request_restart = lambda **kw: order.append("restart") or True
    return runner, order


@pytest.mark.asyncio
async def test_idle_restart_ack_is_delivered_before_teardown_starts(monkeypatch, tmp_path):
    """Restart is requested only after the platform has ANSWERED the ack, not when the send starts."""
    started, answered = asyncio.Event(), asyncio.Event()
    sent = []

    async def send(chat_id, content, **kwargs):
        sent.append(content)
        started.set()
        await answered.wait()  # the Bot API round trip is still in flight
        return SendResult(success=True, message_id="ack")

    runner, order = _idle_restart(monkeypatch, tmp_path, send)
    event = MessageEvent(text="/restart", message_type=MessageType.TEXT, source=make_restart_source(), message_id="m1")

    handled = asyncio.create_task(runner._handle_message(event))
    await asyncio.wait_for(started.wait(), timeout=5)
    for _ in range(20):  # let a restart that does not wait for the answer run now
        await asyncio.sleep(0)
    assert len(sent) == 1 and "Restarting" in sent[0]
    assert order == []  # send in flight: teardown has not been requested
    answered.set()
    result = await asyncio.wait_for(handled, timeout=5)

    assert order == ["restart"]
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


@pytest.mark.asyncio
async def test_local_restart_returns_its_ack(monkeypatch, tmp_path):
    """A LOCAL session (``hermes chat -q "/restart"``) shows the handler's RETURNED reply; its
    adapter's send is a no-op, so an early send there would leave the ack nowhere."""
    from types import SimpleNamespace
    from hermes_agent.gateway.config import Platform
    from hermes_agent.gateway.session import SessionSource
    from hermes_agent.gateway.session_local import LocalSessionAdapter

    async def send(chat_id, content, **kwargs):
        return SendResult(success=True, message_id="ack")

    runner, order = _idle_restart(monkeypatch, tmp_path, send)
    runner.adapters[Platform.LOCAL] = LocalSessionAdapter(SimpleNamespace(runner=runner, sessions={}))
    source = SessionSource(platform=Platform.LOCAL, chat_id="local-1", user_id="u1", chat_type="dm")
    event = MessageEvent(text="/restart", message_type=MessageType.TEXT, source=source, message_id="m1")

    result = await runner._handle_message(event)

    assert order == ["restart"]
    assert result and "Restarting" in result


@pytest.mark.asyncio
async def test_timed_out_restart_ack_is_not_sent_twice(monkeypatch, tmp_path):
    """A send that timed out may already have landed; the reply path must not post a second copy."""
    sent = []

    async def send(chat_id, content, **kwargs):
        sent.append(content)
        return SendResult(success=False, error="Timed out")

    runner, order = _idle_restart(monkeypatch, tmp_path, send)
    event = MessageEvent(text="/restart", message_type=MessageType.TEXT, source=make_restart_source(), message_id="m1")

    result = await runner._handle_message(event)

    assert order == ["restart"] and len(sent) == 1
    assert not result


@pytest.mark.asyncio
@pytest.mark.parametrize("error,retryable", [("Timed out", True), ("httpx.ConnectTimeout: timed out", False)],
                         ids=["retryable-flag", "connect-timeout-text"])
async def test_undelivered_timeout_falls_back_to_the_reply(monkeypatch, tmp_path, error, retryable):
    """A timeout that never left (Telegram flags a wrapped ConnectTimeout / pool timeout ``retryable``;
    other adapters name the connect timeout) was not delivered, so the ack still goes out through the
    reply path, as ``_send_with_retry`` would retry it."""
    async def send(chat_id, content, **kwargs):
        return SendResult(success=False, error=error, retryable=retryable)

    runner, order = _idle_restart(monkeypatch, tmp_path, send)
    event = MessageEvent(text="/restart", message_type=MessageType.TEXT, source=make_restart_source(), message_id="m1")

    result = await runner._handle_message(event)

    assert order == ["restart"]
    assert result and "Restarting" in result
