"""Buzz presence: an ephemeral kind-20001 event on the adapter's authenticated WebSocket.

Buzz clients colour the presence dot from these events. The relay keeps the last status for 180 seconds
and expects a heartbeat every 60, so a running agent shows online and a crashed one fades out on its
own. The relay rejects this kind over HTTP, so the ``buzz`` CLI cannot publish it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

PRESENCE_KIND = 20001
HEARTBEAT_INTERVAL = 60.0  # the relay's presence TTL is 3x this, so two lost beats still read as online
OFFLINE_TIMEOUT = 2.0  # shutdown never waits longer than this for the goodbye


async def publish(adapter: Any, sign_event: Callable[..., dict], status: str, websocket: Any = None) -> None:
    """Publish *status* (``online``, ``away`` or ``offline``) on *websocket*, default the adapter's live one.

    Best-effort: no socket (poll transport, reconnecting) is a silent skip, and a failed sign or send is
    logged at debug, never raised: a presence dot must not take the connection down.
    """
    websocket = websocket or getattr(adapter, "_ws", None)
    if websocket is None:
        return
    try:
        # Off-loop: the pure-Python Schnorr signer takes tens of milliseconds.
        event = await asyncio.to_thread(
            sign_event, private_key=adapter._private_key, kind=PRESENCE_KIND, tags=[], content=status)
        await websocket.send(json.dumps(["EVENT", event], separators=(",", ":")))
    except Exception:
        logger.debug("Buzz: presence %s failed", status, exc_info=True)


async def heartbeat_loop(adapter: Any, websocket: Any, sign_event: Callable[..., dict]) -> None:
    """Announce ``online`` on *websocket* now and every ``HEARTBEAT_INTERVAL`` until cancelled.

    Never returns and never raises, so it cannot end the connection it rides on: a dead socket is the read
    loop's to notice.
    """
    while True:
        if adapter._ws is websocket:  # publish_offline() detached it: nothing may follow the goodbye
            await publish(adapter, sign_event, "online", websocket)
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def publish_offline(adapter: Any, sign_event: Callable[..., dict]) -> None:
    """Say ``offline`` on a graceful shutdown, within ``OFFLINE_TIMEOUT``; a crash lets the relay's TTL expire."""
    websocket, adapter._ws = adapter._ws, None
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(publish(adapter, sign_event, "offline", websocket), OFFLINE_TIMEOUT)
