"""Raw Matrix events read through the client API."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any
from urllib.parse import quote


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


async def raw_state_event(
    client: Any, room_id: str, event_type: str, state_key: str = "",
    query: dict[str, str] | None = None,
) -> Any:
    # Read raw JSON instead of using mautrix's get_state_event. Its typed
    # contents fill in mautrix defaults, which differ from the spec for
    # `invite`, and it raises when a server ignores `format=event` and
    # returns the content only.
    path = (
        f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}"
        f"/state/{quote(event_type, safe='')}/{quote(state_key, safe='')}"
    )
    return await client.api.request(Method.GET, path, query_params=query)


async def decrypt_history_event(client: Any, raw: dict[str, Any]) -> Any:
    if raw.get("type") != "m.room.encrypted":
        return raw
    crypto = getattr(client, "crypto", None)
    if crypto is None:
        raise UndecryptableEvent("missing decryption keys")
    try:
        from mautrix.types import EncryptedEvent, JSON

        return await asyncio.wait_for(crypto.decrypt_megolm_event(EncryptedEvent.deserialize(JSON(raw))), timeout=10.0)
    except Exception as exc:
        logger.debug("Matrix: could not decrypt history event %s: %s", raw.get("event_id"), exc)
        reason = "missing decryption keys" if type(exc).__name__ == "SessionNotFound" else "decryption failed"
        raise UndecryptableEvent(reason) from exc
