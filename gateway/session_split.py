"""Checks shared by the two places a turn moves its route onto a new session (a compression split).

``_sync_session_after_run`` (right after ``run_conversation``) and the agent-result sync in
``_hmwa_shape_agent_response`` both move ``SessionEntry.session_id`` when the agent comes back on
another id. Only a compression continuation of the turn's session may move the route: another
session (an async subagent's, #154994) is outside the conversation's lineage, and a route saved
onto it is refused ``admission_conflict`` on every later lookup, resume and delivery.
"""
from __future__ import annotations

import contextlib
import logging

logger = logging.getLogger("gateway.run")


def is_compression_continuation(store, session_key, session_id, new_session_id) -> bool:
    """Whether *new_session_id* continues *session_id* by compression in *session_key*'s store.

    Both compression rotations publish the parent→child chain before the agent's id moves
    (``publish_compression_child`` in one transaction, ``_adopt_live_compression_child`` after
    re-reading the tip), so a real split is on ``get_compression_chain``. A row the store holds
    that is not on the chain is another conversation. Anything unverifiable (no store for the key,
    no row yet, a read error) keeps the previous behaviour and allows the move."""
    db = None
    with contextlib.suppress(Exception):
        db_for_key = getattr(store, "_db_for_key", None)
        db = db_for_key(session_key) if callable(db_for_key) else None
    if db is None or not callable(getattr(db, "get_compression_chain", None)):
        return True
    try:
        if db.get_session(new_session_id) is None:
            return True
        return new_session_id in db.get_compression_chain(session_id)
    except Exception:
        logger.warning("Could not verify session split %s → %s; allowing it as before",
                       session_id, new_session_id, exc_info=True)
        return True


def restore_route(store, entry, *, published_id, restored_id) -> None:
    """Undo a split whose publication failed: move *entry* back to *restored_id* and save, but
    only while it still names *published_id* (a concurrent ``/new`` or route recovery wins).
    Holds the store's routing lock, as ``_save`` expects; a failing save is logged, not raised,
    so the publication's own error is the one the caller sees."""
    lock = getattr(store, "_lock", None)
    with lock if lock is not None else contextlib.nullcontext():
        if getattr(entry, "session_id", None) != published_id:
            return
        entry.session_id = restored_id
        try:
            store._save()
        except Exception:
            logger.warning("Could not save the restored route %s after a failed split to %s",
                           restored_id, published_id, exc_info=True)
