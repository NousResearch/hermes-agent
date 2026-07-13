"""Deprecated shim — use :mod:`agent.concierge`.

Kept so mid-flight imports and older tests keep working while surfaces
migrate to Concierge naming.
"""

from __future__ import annotations

from agent.concierge import (  # noqa: F401
    ConciergeResult,
    OneRoomResult,
    concierge_enabled,
    handle_concierge,
    handle_one_room_control,
    one_room_control_enabled,
)

__all__ = [
    "ConciergeResult",
    "OneRoomResult",
    "concierge_enabled",
    "handle_concierge",
    "handle_one_room_control",
    "one_room_control_enabled",
]
