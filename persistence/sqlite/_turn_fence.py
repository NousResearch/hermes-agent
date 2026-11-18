"""The claimed-turn fence on a SQLite store's writes.

``StoreHolder._execute_write`` asks :func:`turn_fence_for` once per write which claimed turn's fence the write must
honour (``persistence.core.fences.current_turn_fence`` for one of the store keys the file is served under) and runs
:func:`check_turn_fence` first inside the write's own ``BEGIN IMMEDIATE`` transaction, before the callback: the
authority epoch (``runtime_epoch``) and, unless the fence is ``owner_only``, the session's execution generation
(``sessions.runtime_generation``) must still be the ones the claim stamped, else the write raises
``persistence.TurnFenceSuperseded`` and rolls back with nothing of it landed. These are the counters the admission claim
already stamps (``claim_session_input``); the fence adds no counter of its own.
"""

from __future__ import annotations

import sqlite3
import threading
import weakref
from pathlib import Path
from typing import Any, Optional

from persistence.core.errors import TurnFenceSuperseded
from persistence.core.fences import TurnFence, current_turn_fence, fence_holds, writing_turn
from persistence.core.identity import SessionId

__all__ = ["check_turn_fence", "fence_store_keys", "serve_fence_key", "turn_fence_for"]

_EPOCH_SQL = "SELECT epoch FROM runtime_epoch WHERE singleton=1"
_GENERATION_SQL = "SELECT runtime_generation FROM sessions WHERE id=?"


# holder -> the store keys (``ProfileScope.store_key``) its file is served under besides its own folder's. Kept here,
# beside the holder rather than on it, so serving a key asks nothing of the holder: any object the registry opens as a
# holder serves a store as it did before the fence.
_served: weakref.WeakKeyDictionary[Any, frozenset] = weakref.WeakKeyDictionary()
_served_lock = threading.Lock()
# holder -> its own profile folder's store key, resolved once
_own: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()


def serve_fence_key(holder: Any, key: str) -> None:
    """Record that ``holder``'s file is served under store key ``key``: a claimed turn that store issued fences the
    writes made there."""
    keys = _served.get(holder)
    if keys is not None and key in keys:
        return
    with _served_lock:
        _served[holder] = _served.get(holder, frozenset()) | {key}


def fence_store_keys(holder: Any) -> frozenset:
    """Every store key a claimed turn fencing ``holder``'s writes may carry: the keys it is served under and its own
    profile folder's (``<home>/state.db``)."""
    own = _own.get(holder)
    if own is None:
        from persistence.core.context import canonical_home_key

        own = _own[holder] = canonical_home_key(Path(holder.db_path).parent)
    return _served.get(holder, frozenset()) | {own}


def turn_fence_for(holder: Any) -> Optional[TurnFence]:
    """The fence of the claimed turn the current write belongs to when ``holder``'s file is the store that issued it,
    else ``None``. Outside any turn this costs one context-variable read."""
    mark = writing_turn()
    if mark is None or mark.store is None or mark.store not in fence_store_keys(holder):
        return None
    return current_turn_fence(mark.store)


def check_turn_fence(conn: sqlite3.Connection, fence: Optional[TurnFence]) -> None:
    """Refuse the current write once ``fence`` no longer holds its session's lease; runs on the write's connection
    inside its transaction. ``None`` checks nothing."""
    if fence is None:
        return
    row = conn.execute(_EPOCH_SQL).fetchone()
    generation = None
    if not fence.owner_only:
        session = conn.execute(_GENERATION_SQL, (fence.session_id,)).fetchone()
        generation = None if session is None else session[0]
    if not fence_holds(fence, None if row is None else row[0], generation):
        raise TurnFenceSuperseded(
            SessionId(fence.session_id),
            f"turn {fence.turn_id!r} of session {fence.session_id!r} no longer holds its lease (fence epoch="
            f"{fence.epoch}, generation={fence.generation}{', owner only' if fence.owner_only else ''}); "
            "refusing its write")
