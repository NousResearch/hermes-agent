"""``/api/display/ws`` — raw RFB over WebSocket for the Bot Desktop viewer.

The Desktop renderer calls ``display.observe`` on its authenticated ``/api/ws`` connection, gets a
single-use 30 s ticket pinned to that profile's RFB socket, then opens this route with
``?display_ticket=``. No websockify, no new port: the bridge splices the profile's 0600 Unix socket
into the WebSocket as binary frames with backpressure both ways, and runs the client stream through
:class:`tools.bot_desktop.rfb_filter.RfbClientFilter` so keyboard, pointer and clipboard reach Xvnc
only from the viewer that currently holds the lease. noVNC's ``viewOnly`` is UX; this is the gate.

A lease change closes the evicted viewer's socket with 4000 ``control-taken`` so its UI drops back to
Watch mode and reconnects.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from hermes_cli.web_server_chat import _ws_request_is_allowed

_log = logging.getLogger(__name__)
router = APIRouter()

_READ_CHUNK = 64 * 1024
_CLOSE_CONTROL_TAKEN = 4000
_CLOSE_DESKTOP_GONE = 4001
_CLOSE_BAD_TICKET = 4401
_CLOSE_NOT_ALLOWED = 4403
_CLOSE_PROTOCOL = 1003


def _consume_display_ticket(ws: WebSocket) -> Optional[dict]:
    from hermes_cli.dashboard_auth.ws_tickets import TicketInvalid, consume_ticket
    ticket = ws.query_params.get("display_ticket", "")
    if not ticket:
        return None
    try:
        info = consume_ticket(ticket)
    except TicketInvalid:
        return None
    if info.get("provider") != "bot-desktop" or not info.get("hermes_home"):
        return None
    return info


@router.websocket("/api/display/ws")
async def display_ws(ws: WebSocket) -> None:
    if not _ws_request_is_allowed(ws):
        await ws.close(code=_CLOSE_NOT_ALLOWED)
        return
    info = _consume_display_ticket(ws)
    if info is None:
        await ws.close(code=_CLOSE_BAD_TICKET, reason="display ticket missing, expired or used")
        return

    from hermes_constants import hermes_home_key
    from tools.bot_desktop import lease as _lease
    from tools.bot_desktop.rfb_filter import RfbClientFilter
    from pathlib import Path

    sock = Path(info["hermes_home"]) / "bot-desktop" / "rfb.sock"
    profile_key = hermes_home_key(info["hermes_home"])
    viewer_id = str(info.get("viewer_id") or info.get("user_id") or "viewer")
    if not sock.exists():
        await ws.close(code=_CLOSE_DESKTOP_GONE, reason="Bot Desktop is not running")
        return
    try:
        reader, writer = await asyncio.open_unix_connection(str(sock))
    except OSError as exc:
        _log.warning("display ws: cannot reach RFB socket %s: %s", sock, exc)
        await ws.close(code=_CLOSE_DESKTOP_GONE, reason="Bot Desktop socket unreachable")
        return

    await ws.accept()
    loop = asyncio.get_running_loop()
    evicted = asyncio.Event()
    held = {"ever": _lease.viewer_may_send_input(viewer_id, profile_key=profile_key)}

    def _on_lease(key: str, lease) -> None:
        # A viewer that held control during this connection and lost it to ANOTHER human is kicked so
        # its UI repaints; a plain hand-back to the agent, pure watchers and the new holder stay connected.
        if key != profile_key:
            return
        mine = lease.holder == _lease.HUMAN and lease.viewer_id == viewer_id
        if mine:
            held["ever"] = True
        elif held["ever"] and lease.holder == _lease.HUMAN:
            loop.call_soon_threadsafe(evicted.set)
    unsubscribe = _lease.on_change(_on_lease)

    rfb_filter = RfbClientFilter(lambda: _lease.viewer_may_send_input(viewer_id, profile_key=profile_key))

    async def rfb_to_ws() -> None:
        while True:
            chunk = await reader.read(_READ_CHUNK)
            if not chunk:
                return
            await ws.send_bytes(chunk)  # awaiting the send is the backpressure toward Xvnc

    async def ws_to_rfb() -> None:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data is None:
                await ws.close(code=_CLOSE_PROTOCOL, reason="RFB is binary")
                return
            try:
                allowed = rfb_filter.feed(data)
            except ValueError as exc:
                await ws.close(code=_CLOSE_PROTOCOL, reason=str(exc)[:100])
                return
            if allowed:
                writer.write(allowed)
                await writer.drain()  # backpressure toward the browser

    async def watch_eviction() -> None:
        await evicted.wait()
        await ws.close(code=_CLOSE_CONTROL_TAKEN, reason="control-taken")

    tasks = [asyncio.create_task(rfb_to_ws()), asyncio.create_task(ws_to_rfb()),
             asyncio.create_task(watch_eviction())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, ConnectionError)):
                _log.debug("display ws ended: %r", exc)
    finally:
        unsubscribe()
        writer.close()
        # Closing the viewer window hands control back; a stale holder never pins the agent out.
        if _lease.viewer_may_send_input(viewer_id, profile_key=profile_key):
            _lease.release(viewer_id, profile_key=profile_key)
        try:
            await ws.close()
        except Exception:  # already closed by the peer or by an eviction
            pass
