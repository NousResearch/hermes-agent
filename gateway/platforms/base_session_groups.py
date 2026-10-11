"""Session-group turn serialization for ``BasePlatformAdapter`` (#79198).

Chats on different adapters that share one session key (``gateway.session.session_group_source``)
take turns under one per-key lock in arrival order, each holding it through its own delivery, so no
turn runs or replies while another chat's turn owns the key.

Imports nothing from ``gateway.platforms.base``, which imports this module.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import Any, Optional

from gateway.session import key_source_for

# One turn lock per shared key across all adapters; held by live turns only.
_GROUP_TURN_LOCKS: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()


def group_turn_lock_for(event: Any, session_key: str) -> Optional[asyncio.Lock]:
    """The per-key turn lock for a session-group event, else ``None`` (ungrouped events)."""
    source = getattr(event, "source", None)
    if source is None or key_source_for(source) is source:
        return None
    lock = _GROUP_TURN_LOCKS.get(session_key)
    if lock is None:
        lock = _GROUP_TURN_LOCKS[session_key] = asyncio.Lock()
    return lock


async def acquire_group_turn(event: Any, session_key: str) -> Optional[asyncio.Lock]:
    """Wait for the group turn lock of a session-group event; the held lock, or ``None``."""
    lock = group_turn_lock_for(event, session_key)
    if lock is not None:
        await lock.acquire()
    return lock


async def defer_to_group_turn(adapter: Any, event: Any, session_key: str) -> bool:
    """True when another chat in the session group owns the key. *event* is then handled as a busy
    arrival on *adapter*: commands and clarify answers act now, anything else queues as this chat's
    own next turn (behind the lock)."""
    lock = group_turn_lock_for(event, session_key)
    if lock is None or not lock.locked():
        return False
    await adapter._handle_message_while_active(event, session_key)
    queued = None if session_key in adapter._active_sessions else adapter.get_pending_message(session_key)
    if queued is not None:
        adapter._start_session_processing(queued, session_key)
    return True
