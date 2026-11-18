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
from typing import Any, Optional

from persistence.core.errors import TurnFenceSuperseded
from persistence.core.fences import TurnFence, current_turn_fence, fence_holds, writing_turn
from persistence.core.identity import SessionId

__all__ = ["check_turn_fence", "turn_fence_for"]

_EPOCH_SQL = "SELECT epoch FROM runtime_epoch WHERE singleton=1"
_GENERATION_SQL = "SELECT runtime_generation FROM sessions WHERE id=?"


def turn_fence_for(holder: Any) -> Optional[TurnFence]:
    """The fence of the claimed turn the current write belongs to when ``holder``'s file is the store that issued it,
    else ``None``. Outside any turn this costs one context-variable read."""
    mark = writing_turn()
    if mark is None or mark.store is None or mark.store not in holder.fence_store_keys():
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
