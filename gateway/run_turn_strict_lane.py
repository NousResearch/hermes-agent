"""Strict internal-delivery admission for ``_hmwa_resolve_session``.

A strict event (``gateway_session_strict``) is pinned to one session id and must never create a
session. The route recheck asks the in-memory session store, which cannot see that the persisted
row was ended or that its transcript is empty, and a newer run can start while that recheck is
awaited. Both gaps let a stale event reach the side effects that follow resolution (source cache,
Telegram topic binding, heartbeat owner) before later run-generation checks drop it.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

logger = logging.getLogger(__name__)


async def strict_lane_session_entry(runner, expected_session_key: str, pinned_session_id: str,
                                    run_generation=None):
    """Return the pinned, live, current session entry, or ``None`` to drop the event.

    Order matters: prove the persisted context BEFORE the recheck, so the database has the last
    word rather than the cache; check the run generation AFTER the recheck and before any side
    effect. ``run_generation`` is None only for direct callers outside a turn (tests, evals); the
    turn handler always passes it.
    """
    if not await prove_live_persisted_context(runner, pinned_session_id):
        return None
    session_entry = await runner.async_session_store.lookup_by_session_key(expected_session_key)
    if session_entry is None or not pinned_session_id or session_entry.session_id != pinned_session_id:
        logger.warning(
            "Dropping internally routed event: expected session id=%s is no longer current for key=%s",
            pinned_session_id or "missing", expected_session_key or "missing",
        )
        return None
    if run_generation is not None and not runner._is_session_run_current(
            session_entry.session_key, run_generation):
        logger.warning(
            "strict_session_run_superseded: pinned=%s key=%s",
            pinned_session_id or "missing", expected_session_key or "missing",
        )
        return None
    return session_entry


async def prove_live_persisted_context(runner, pinned_session_id: str) -> bool:
    """Prove a live persisted context exists before internal delivery.

    Absence, an ended session, an empty transcript, a missing reader or a failed read all fail
    closed, so an internal lane never opens a fresh conversation on a dead one. Bounded metadata
    read only (``SessionDB.get_session_context_state``).
    """
    if not pinned_session_id:
        return False
    session_db = getattr(runner, "_session_db", None)
    if session_db is None:
        logger.warning("strict_session_no_session_db: pinned=%s", pinned_session_id)
        return False
    reader = getattr(session_db, "get_session_context_state", None)
    if not callable(reader):
        logger.warning("strict_session_context_reader_missing on %s", type(session_db).__name__)
        return False
    try:
        if inspect.iscoroutinefunction(reader):
            context_state = await reader(pinned_session_id)
        else:
            # A synchronous reader runs in a thread: never block the event loop on SQLite.
            context_state = await asyncio.to_thread(reader, pinned_session_id)
        if inspect.isawaitable(context_state):
            # Neither known shape (sync SessionDB, AsyncSessionDB offloading). A sync callable
            # returning an awaitable is not caught by iscoroutinefunction; fail closed rather
            # than guess.
            logger.warning(
                "strict_session_context_reader_unexpected_awaitable: pinned=%s reader=%r",
                pinned_session_id, getattr(reader, "__qualname__", reader),
            )
            return False
    except Exception as exc:
        logger.warning(
            "strict_session_context_lookup_failed: pinned=%s error_type=%s",
            pinned_session_id, type(exc).__name__,
        )
        logger.debug("strict_session_context_lookup_traceback", exc_info=True)
        return False
    if (
        not context_state
        or context_state.get("id") != pinned_session_id
        or context_state.get("ended_at")
        or int(context_state.get("message_count") or 0) <= 0
    ):
        logger.warning("strict_session_no_live_context: pinned=%s", pinned_session_id)
        return False
    return True
