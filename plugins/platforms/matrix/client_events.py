"""Raw Matrix events read through the client API."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)

try:
    from mautrix.api import Method
except ImportError:
    class Method(str, Enum):
        GET = "GET"


class UndecryptableEvent(Exception):
    """An encrypted history event could not be decrypted. ``str()`` gives the reason."""


def raw_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    serialize = getattr(event, "serialize", None)
    return serialize() if callable(serialize) else {}


async def decrypt_history_event(client: Any, raw: dict) -> Any:
    if raw.get("type") != "m.room.encrypted":
        return raw
    crypto = getattr(client, "crypto", None)
    if crypto is None:
        raise UndecryptableEvent("missing decryption keys")
    try:
        from mautrix.types import Event

        return await asyncio.wait_for(crypto.decrypt_megolm_event(Event.deserialize(raw)), timeout=10.0)
    except Exception as exc:
        logger.debug("Matrix: could not decrypt history event %s: %s", raw.get("event_id"), exc)
        reason = "missing decryption keys" if type(exc).__name__ == "SessionNotFound" else "decryption failed"
        raise UndecryptableEvent(reason) from exc
