"""Session persistence policy and ephemeral (temporary) chat registry.

A temporary chat must leave NOTHING on disk. Guarding only transcript writers is
insufficient: token/cost accounting, titles, compression handoffs, and side stores
all write session-keyed data.

The persistence policy tracks active temporary/ephemeral session identities in-process.
Reference counting ensures multiple owners (e.g. subagents or simultaneous handles)
do not prematurely unmark the policy before all owners finalize.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict

logger = logging.getLogger(__name__)

# Process-local registry of active temporary (ephemeral) session IDs mapped to their owner reference counts.
_EPHEMERAL_SESSION_COUNTS: Dict[str, int] = {}
_EPHEMERAL_LOCK = threading.Lock()


def mark_session_ephemeral(session_id: str) -> None:
    """Register *session_id* as temporary so no row or artifact is ever persisted for it.

    Increments the reference count if already registered.
    """
    if not session_id:
        return
    with _EPHEMERAL_LOCK:
        _EPHEMERAL_SESSION_COUNTS[session_id] = _EPHEMERAL_SESSION_COUNTS.get(session_id, 0) + 1


def unmark_session_ephemeral(session_id: str, *, force: bool = False) -> None:
    """Decrement the owner reference count for *session_id* and remove it when count reaches zero.

    If *force* is True, drops the registration immediately regardless of count.
    """
    if not session_id:
        return
    with _EPHEMERAL_LOCK:
        if force:
            _EPHEMERAL_SESSION_COUNTS.pop(session_id, None)
        elif session_id in _EPHEMERAL_SESSION_COUNTS:
            _EPHEMERAL_SESSION_COUNTS[session_id] -= 1
            if _EPHEMERAL_SESSION_COUNTS[session_id] <= 0:
                del _EPHEMERAL_SESSION_COUNTS[session_id]


def is_session_ephemeral(session_id: str) -> bool:
    """Return True iff *session_id* is currently registered as a temporary (ephemeral) chat."""
    if not session_id:
        return False
    with _EPHEMERAL_LOCK:
        return _EPHEMERAL_SESSION_COUNTS.get(session_id, 0) > 0


# Aliases for compatibility with different subsystems
register_ephemeral_session = mark_session_ephemeral
unregister_ephemeral_session = unmark_session_ephemeral
is_ephemeral_session = is_session_ephemeral

__all__ = [
    "mark_session_ephemeral",
    "unmark_session_ephemeral",
    "is_session_ephemeral",
    "register_ephemeral_session",
    "unregister_ephemeral_session",
    "is_ephemeral_session",
]
