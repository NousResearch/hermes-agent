"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record, not provider-visible message
# content. They must not influence context-pressure decisions.
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset({"timestamp"})

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time; all other callers use
    the local wall clock. Returns the same mapping for use at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp and append one live transcript message.

    A durable row may be re-staged as the current turn after a cold resume. If
    that exact row is already the history tail, replace the tail by durable
    identity instead of appending it twice. Content equality is deliberately
    irrelevant: two distinct rows with identical text are distinct turns.
    """
    stamped = stamp_message_timestamp(message, timestamp=timestamp)
    row_id = stamped.get("_row_id")
    if (
        isinstance(row_id, int)
        and messages
        and isinstance(messages[-1], MutableMapping)
        and messages[-1].get("_row_id") == row_id
    ):
        messages[-1] = stamped
    else:
        messages.append(stamped)
    return message
