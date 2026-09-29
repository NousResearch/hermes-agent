"""Raw Matrix events read through the client API."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

try:
    from mautrix.api import Method
except ImportError:
    class Method(str, Enum):
        GET = "GET"


def raw_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    serialize = getattr(event, "serialize", None)
    return serialize() if callable(serialize) else {}


async def decrypt_raw_event(client: Any, raw: dict[str, Any]) -> tuple[Any | None, str | None]:
    """Return the decrypted event, or ``None`` with a label for the decryption failure."""
    crypto = getattr(client, "crypto", None)
    if crypto is None:
        return None, "missing decryption keys"
    try:
        from mautrix.types import Event

        event = await asyncio.wait_for(crypto.decrypt_megolm_event(Event.deserialize(raw)), timeout=10.0)
    except Exception as exc:
        return None, "missing decryption keys" if type(exc).__name__ == "SessionNotFound" else "decryption failed"
    if event is None:
        return None, "missing decryption keys"
    return event, None
