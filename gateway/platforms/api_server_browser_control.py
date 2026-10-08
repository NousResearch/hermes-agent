"""Browser-control transport helpers for the API server's controller WebSocket.

Kept out of ``api_server`` so the controller socket's send, keepalive and turn-target
plumbing live together.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Optional

from gateway.browser_control_broker import browser_turn_target

logger = logging.getLogger(__name__)

#: App-level keepalive cadence for controller sockets. MV3 extension service workers idle out
#: after 30 s and WebSocket ping/pong frames do not reset that timer (only messages do). A worker
#: killed while its socket lingers drops the next command, which then times out after 30 s.
BROWSER_CONTROL_KEEPALIVE_SECONDS = 20.0


def _browser_controller_ws_sender(ws, loop, *, wait_timeout: float = 10.0):
    """Return a loop-aware broker sender for one aiohttp controller socket.

    A wait timeout means the coroutine is still in flight, not that the frame was rejected:
    keep the broker command pending (its own deadline decides); a real send error propagates.
    """

    def send(frame: dict) -> None:
        if ws.closed:
            raise ConnectionError("browser-control websocket is closed")
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            loop.create_task(ws.send_json(frame))
            return
        future = asyncio.run_coroutine_threadsafe(ws.send_json(frame), loop)
        try:
            future.result(timeout=wait_timeout)
        except concurrent.futures.TimeoutError:
            if future.done():
                raise

            def observe_late_send(completed):
                try:
                    completed.result()
                except Exception:
                    logger.exception("browser-controller websocket send failed after wait timeout")
            future.add_done_callback(observe_late_send)
    return send


async def run_controller_keepalive(ws, *, interval: float = BROWSER_CONTROL_KEEPALIVE_SECONDS) -> None:
    """Send an unsolicited heartbeat echo every ``interval`` seconds until the socket closes.

    Controllers treat a heartbeat with an unknown nonce as a no-op, but the message still counts
    as WebSocket activity, which is what keeps an MV3 extension service worker alive.
    """
    sequence = 0
    while not ws.closed:
        await asyncio.sleep(interval)
        if ws.closed:
            return
        sequence += 1
        try:
            await ws.send_json({"method": "browser.controller.heartbeat",
                                "params": {"nonce": f"server-keepalive-{sequence}", "ok": True}})
        except (ConnectionError, RuntimeError):
            return


def bind_browser_turn_target(broker: Any, session_id: Optional[str], user_message: Any) -> None:
    """Pin (or clear) the exact tab this turn's browser envelope names for the session's commands."""
    if not session_id:
        return
    target = browser_turn_target(user_message)
    if target is None:
        broker.clear_turn_target(session_id)
    else:
        broker.set_turn_target(session_id, **target)
