"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar
from uuid import uuid4


# These fields describe Hermes' durable record, not provider-visible message
# content. They must not influence context-pressure decisions.
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset({"timestamp"})

RUNTIME_NOTIFICATION_TOOL_NAME = "__runtime_notification__"

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
    """Stamp and append one live transcript message."""
    messages.append(stamp_message_timestamp(message, timestamp=timestamp))
    return message


def append_runtime_notification(
    messages: list[Any], content: Any, *, call_id: Optional[str] = None,
) -> MutableMapping[str, Any]:
    """Append a provider-valid virtual-tool result authored by the runtime.

    The assistant tool-call envelope is protocol scaffolding; the durable tool
    row carries the source structurally through ``tool_name`` and
    ``tool_call_id`` instead of forging a user turn or sniffing its content.
    """
    call_id = call_id or f"runtime-notification-{uuid4().hex}"
    tool_call = {
        "id": call_id,
        "type": "function",
        "function": {"name": RUNTIME_NOTIFICATION_TOOL_NAME, "arguments": "{}"},
    }
    tail = messages[-1] if messages and isinstance(messages[-1], MutableMapping) else None
    if tail is not None and tail.get("role") == "assistant" and not tail.get("tool_calls"):
        tail["tool_calls"] = [tool_call]
    else:
        append_message(messages, {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
            "_runtime_notification_scaffold": True,
        })
    return append_message(messages, {
        "role": "tool",
        "name": RUNTIME_NOTIFICATION_TOOL_NAME,
        "tool_name": RUNTIME_NOTIFICATION_TOOL_NAME,
        "content": content,
        "tool_call_id": call_id,
        "_runtime_notification": True,
    })
