"""Buzz typing indicator: an ephemeral kind-20002 event on the adapter's authenticated WebSocket.

Buzz clients render the "typing" bubble from these events; Buzz's own agents (``buzz-acp``) publish one
every 3 seconds for as long as they work on a turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

TYPING_KIND = 20002
TYPING_INTERVAL = 3.0  # buzz-acp's cadence; the gateway ticks faster, so publish() throttles to it


def thread_tags(root: Any, parent: Any = None) -> list[list[str]]:
    """NIP-10 ``e`` tags that place the bubble in a thread (none for a top-level chat)."""
    if not root:
        return []
    root, parent = str(root), str(parent or root)
    return ([["e", root, "", "root"]] if root != parent else []) + [["e", parent, "", "reply"]]


async def publish(adapter: Any, sign_event: Callable[..., dict], chat_id: str, metadata: dict) -> None:
    """Publish one typing event for *chat_id*, at most once per ``TYPING_INTERVAL`` per chat.

    Best-effort: no socket (poll transport, reconnecting) is a silent skip, and a failed sign or send is
    logged at debug, never raised: a typing bubble must not fail the turn.
    """
    websocket = getattr(adapter, "_ws", None)
    now = time.monotonic()
    if websocket is None or now - adapter._typing_sent.get(chat_id, float("-inf")) < TYPING_INTERVAL:
        return
    adapter._typing_sent[chat_id] = now
    try:
        # The bubble follows the reply: flat when replies post flat, else in the thread they join.
        root = None if adapter._reply_to_mode == "off" else adapter._resolve_reply_anchor(metadata.get("thread_id"))
        # Off-loop: the pure-Python Schnorr signer takes tens of milliseconds.
        event = await asyncio.to_thread(
            sign_event, private_key=adapter._private_key, kind=TYPING_KIND,
            tags=[["h", chat_id], *thread_tags(root, metadata.get("reply_to_message_id"))])
        await websocket.send(json.dumps(["EVENT", event], separators=(",", ":")))
    except Exception:
        logger.debug("Buzz: typing indicator failed in %s", chat_id, exc_info=True)
