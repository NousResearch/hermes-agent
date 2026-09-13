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
import threading
from typing import Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from hermes_cli.web_server_chat import _ws_request_is_allowed

_log = logging.getLogger(__name__)
router = APIRouter()

_READ_CHUNK = 64 * 1024
_CLOSE_CONTROL_TAKEN = 4000
_CLEAN_CLOSE = frozenset({1000, 1001})
_CLOSE_DESKTOP_GONE = 4001
_CLOSE_BAD_TICKET = 4401
_CLOSE_NOT_ALLOWED = 4403
_CLOSE_PROTOCOL = 1003
_LEASE_REFRESH_S = 0.25
_active_viewers_lock = threading.Lock()
_active_viewers: dict[str, tuple[object, Callable[[], None]]] = {}


def _claim_active_viewer(profile_key: str, evict: Callable[[], None]) -> object:
    """Claim the one stream slot for a profile and evict its stale viewer, if any."""
    token = object()
    with _active_viewers_lock:
        previous = _active_viewers.get(profile_key)
        _active_viewers[profile_key] = (token, evict)
    if previous is not None:
        previous[1]()
    return token


def _release_active_viewer(profile_key: str, token: object) -> None:
    with _active_viewers_lock:
        current = _active_viewers.get(profile_key)
        if current is not None and current[0] is token:
            _active_viewers.pop(profile_key, None)


def _should_evict(lease, viewer_id: str) -> bool:
    """While a human holds control, only that viewer may observe the framebuffer."""
    from tools.bot_desktop import lease as _lease

    return lease.holder == _lease.HUMAN and lease.viewer_id != viewer_id


async def _admit_active_viewer(
    profile_key: str,
    profile_home: str,
    viewer_id: str,
    evict: Callable[[], None],
) -> Optional[tuple[object, object]]:
    """Atomically validate the lease and bind its epoch to the profile's one viewer slot."""
    from tools.bot_desktop import lease as _lease

    # acquire/release use this same file lock. A takeover that wins first is therefore observed
    # before the slot changes; a viewer that wins first is authorized to begin its pumps at that
    # lease epoch and subsequent transitions reach the subscribed eviction callback below.
    with _lease.locked_snapshot(profile_key=profile_home) as current_lease:
        if _should_evict(current_lease, viewer_id):
            return None
        slot = _claim_active_viewer(profile_key, evict)
        return slot, current_lease


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
    await _bridge(ws, info)


async def _bridge(ws: WebSocket, info: dict) -> None:
    """Pump RFB bytes between the viewer socket and THIS profile's Xvnc, gated by the lease."""
    from hermes_constants import hermes_home_key
    from tools.bot_desktop import lease as _lease
    from tools.bot_desktop.rfb_filter import RfbClientFilter
    from pathlib import Path

    sock = Path(info["hermes_home"]) / "bot-desktop" / "rfb.sock"
    profile_home = str(info["hermes_home"])
    profile_key = hermes_home_key(profile_home)
    viewer_id = str(info.get("viewer_id") or info.get("user_id") or "viewer")
    # A login session can legitimately mint more than one single-use display ticket (reconnects),
    # so authentication alone cannot make takeover private. Refuse every non-holder here before it
    # receives an initial framebuffer; the listener/poller below closes connections already open.
    if _should_evict(_lease.get(profile_key=profile_home), viewer_id):
        await ws.close(code=_CLOSE_CONTROL_TAKEN, reason="control-taken")
        return
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
    # Input gate cache: reading lease.json per client message (a stat + read on the event loop for
    # every pointer move) is replaced by a decision refreshed on this process's on_change callback
    # and by a file re-read at most every _LEASE_REFRESH_S, so another process's takeover still lands.
    allowed = {"input": False, "at": loop.time()}

    def _refresh_allowed(lease=None) -> None:
        if lease is None:
            lease = _lease.get(profile_key=profile_home)
        allowed["input"] = lease.holder == _lease.HUMAN and lease.viewer_id == viewer_id
        allowed["at"] = loop.time()

    def _may_send_input() -> bool:
        if loop.time() - allowed["at"] > _LEASE_REFRESH_S:
            _refresh_allowed()
        return allowed["input"]

    def _on_lease(key: str, lease) -> None:
        if key != profile_key:
            return
        loop.call_soon_threadsafe(_refresh_allowed, lease)
        if _should_evict(lease, viewer_id):
            loop.call_soon_threadsafe(evicted.set)

    unsubscribe = _lease.on_change(_on_lease)

    admission = await _admit_active_viewer(
        profile_key,
        profile_home,
        viewer_id,
        lambda: loop.call_soon_threadsafe(evicted.set),
    )
    if admission is None:
        unsubscribe()
        writer.close()
        await ws.close(code=_CLOSE_CONTROL_TAKEN, reason="control-taken")
        return
    viewer_slot, admitted_lease = admission
    _refresh_allowed(admitted_lease)

    rfb_filter = RfbClientFilter(_may_send_input)

    viewer_closed = asyncio.Event()

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
                # 1000/1001 = the viewer closed the window; anything else is a dropped link.
                if message.get("code") in _CLEAN_CLOSE:
                    viewer_closed.set()
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
        # on_change is process-local. Polling also catches a takeover written by a separate Hermes
        # process, which must stop framebuffer delivery—not merely block that viewer's input.
        while not evicted.is_set():
            if _should_evict(_lease.get(profile_key=profile_home), viewer_id):
                evicted.set()
                break
            try:
                await asyncio.wait_for(evicted.wait(), timeout=_LEASE_REFRESH_S)
            except asyncio.TimeoutError:
                pass
        await ws.close(code=_CLOSE_CONTROL_TAKEN, reason="control-taken")

    tasks = [
        asyncio.create_task(rfb_to_ws()),
        asyncio.create_task(ws_to_rfb()),
        asyncio.create_task(watch_eviction()),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, ConnectionError)):
                _log.debug("display ws ended: %r", exc)
    finally:
        _release_active_viewer(profile_key, viewer_slot)
        unsubscribe()
        writer.close()
        # Closing the viewer window hands control back. A DROPPED link (laptop lid, Wi-Fi, 1006)
        # keeps the human's exclusion: they may be mid-login on that screen and the agent must not
        # resume into it. The Desktop reconnects into the same lease, or the human hands back.
        if viewer_closed.is_set() and _lease.viewer_may_send_input(viewer_id, profile_key=profile_home):
            _lease.release(viewer_id, profile_key=profile_home)
        try:
            await ws.close()
        except Exception:  # already closed by the peer or by an eviction
            pass
