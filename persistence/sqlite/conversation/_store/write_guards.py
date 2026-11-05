"""Active turn-lease and compression-lock guards on session rows."""


from __future__ import annotations
from typing import Iterable
from persistence.sqlite._holder import StoreHolder


def _guarded_ids(holder: StoreHolder, conn, ids: Iterable[str]) -> set:
    """Ids in *ids* protected by a live turn lease / compression lock. Idle compression-ended
    parents are closed, not live, so they are not guarded (prune and delete share this)."""
    return {sid for sid in ids if _write_guards_reject(holder, conn, sid, allow_closed_compression_parent=True)}


def _write_guards_reject(holder: StoreHolder, conn, sid: str, **kwargs) -> bool:
    """True when a live turn lease / compression lock protects ``sid``; expired or
    dead-holder guards are reclaimed and fenced as a side effect."""
    from persistence.sqlite._state.errors import SessionCompressionInProgressError
    from persistence.sqlite._state.errors import SessionTurnLeaseLostError
    try:
        _check_transcript_write_guards(holder,
            conn, sid, compression_lock_holder=None, turn_lease_holder=None,
            reject_active_turn_lease=True, reject_active_compression_lock=True, **kwargs)
    except (SessionCompressionInProgressError, SessionTurnLeaseLostError):
        return True
    return False


# Sibling store modules import this module's copies in turn, so their copies are imported after these.
from persistence.sqlite.conversation._store.messages import _check_transcript_write_guards
