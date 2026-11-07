"""/api/events fan-out: a subscriber that already left is skipped quietly; a real send fault still warns.

A dashboard tab in a reconnect loop left ~2k ``broadcast send failed`` WARNINGs with full tracebacks in one
evening; each was a subscriber whose socket Starlette had already marked DISCONNECTED.
"""

import asyncio
import logging
from types import SimpleNamespace

from starlette.websockets import WebSocket, WebSocketState

from hermes_cli.web_routers import chat_ws
from tui_gateway.ws import ws_peer_gone


def _subscriber(send_error):
    async def receive():  # pragma: no cover - the broadcast never reads
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        if message["type"] == "websocket.send":
            raise send_error

    sub = WebSocket({"type": "websocket", "path": "/api/events", "headers": []}, receive, send)
    sub.client_state = WebSocketState.CONNECTED
    sub.application_state = WebSocketState.CONNECTED
    return sub


def _broadcast(subs, caplog):
    caplog.set_level(logging.DEBUG, logger=chat_ws._log.name)
    app = SimpleNamespace(state=SimpleNamespace())

    async def run():
        channels, _lock = chat_ws._get_event_state(app)
        channels["chan"] = set(subs)
        await chat_ws._broadcast_event(app, "chan", '{"type":"tool.start"}')

    asyncio.run(run())
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_departed_subscriber_is_skipped_without_a_warning(caplog):
    gone = _subscriber(OSError("client disconnected"))  # Starlette: WebSocketDisconnect(1006) + DISCONNECTED latch

    assert _broadcast([gone], caplog) == []
    assert gone.application_state == WebSocketState.DISCONNECTED
    assert not ws_peer_gone(gone)

    # The next frame meets the latched socket ('Cannot call "send" once a close message has been sent').
    assert _broadcast([gone], caplog) == []


def test_server_closed_subscriber_is_skipped(caplog):
    sub = _subscriber(RuntimeError("must not send"))
    asyncio.run(sub.close())
    assert not ws_peer_gone(sub)
    assert _broadcast([sub], caplog) == []


def test_subscriber_closed_between_check_and_send_is_skipped(caplog):
    class ClosingSubscriber(WebSocket):
        async def send_text(self, data):
            await self.close()
            await super().send_text(data)

    async def receive():
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        pass

    sub = ClosingSubscriber({"type": "websocket", "path": "/api/events", "headers": []}, receive, send)
    sub.client_state = WebSocketState.CONNECTED
    sub.application_state = WebSocketState.CONNECTED
    assert _broadcast([sub], caplog) == []


def test_dynamic_exception_type_is_not_logged(caplog):
    secret = "synthetic-token-ABC123"
    live = _subscriber(type(secret, (RuntimeError,), {})())
    assert len(_broadcast([live], caplog)) == 1
    assert secret not in caplog.text


def test_send_fault_on_a_live_subscriber_still_warns(caplog):
    live = _subscriber(RuntimeError("Bearer synthetic-token-ABC123"))

    warnings = _broadcast([live], caplog)

    assert [r.getMessage() for r in warnings] == ["broadcast send failed for subscriber"]
    assert warnings[0].exc_info is None
    assert "synthetic-token-ABC123" not in caplog.text
