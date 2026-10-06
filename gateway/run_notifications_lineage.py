"""Session-lineage lookups for async-delegation completions (resolver and pre-flight classifier).

Both walks are read-only and fail closed: ``None`` means ownership could not be verified, so the
caller leaves the chat route alone and the result stays in the delegation records.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from gateway.session import SessionEntry

logger = logging.getLogger("gateway.run")

# A delegate_task child (``sessions.source == "subagent"``) is an internal execution transcript,
# never a chat route owner. A RUNNING child's background-process notice is stamped with the CHILD
# session id (tools/process_registry.py), so a completion pinned to it must be mapped up the
# ``parent_session_id`` chain to the chat that spawned it before any route check or switch.
SUBAGENT_SESSION_SOURCE = "subagent"
# Real nesting is capped far lower by delegation.max_spawn_depth; past this the lineage is corrupt.
MAX_SUBAGENT_OWNER_HOPS = 16
# Stamped on the chat session an async-delegation repin moves the route away from. Deliberately NOT
# in gateway.run._USER_BOUNDARY_END_REASONS: a repin is bookkeeping, not the user closing the
# thread, so results still pinned to that session retarget to the chat's current session instead
# of being dropped as permanently gone.
ASYNC_DELEGATION_REPIN_END_REASON = "async_delegation_repin"


def is_subagent_row(row: Optional[Dict[str, Any]]) -> bool:
    return row is not None and str(row.get("source") or "") == SUBAGENT_SESSION_SOURCE


async def resolve_subagent_owner(
    session_db: Any, session_id: str, row: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Walk a subagent session up ``parent_session_id`` to the first non-subagent session.

    Returns ``(owner_id, owner_row)``; a non-subagent row comes back unchanged. ``None`` for a
    missing parent, a cycle, a lookup error or too many hops.
    """
    seen: set = set()
    for _ in range(MAX_SUBAGENT_OWNER_HOPS + 1):
        if row is None:
            logger.warning(
                "Async-delegation completion has subagent session %s with no parent row; "
                "dropping injection.", session_id,
            )
            return None
        if not is_subagent_row(row):
            return session_id, row
        if session_id in seen:
            logger.warning(
                "Async-delegation completion has cyclic subagent lineage at session %s; "
                "dropping injection.", session_id,
            )
            return None
        seen.add(session_id)
        parent_id = str(row.get("parent_session_id") or "").strip()
        if not parent_id:
            logger.warning(
                "Async-delegation completion pinned to subagent session %s with no parent; "
                "dropping injection.", session_id,
            )
            return None
        try:
            row = await session_db.get_session(parent_id)
        except Exception:
            logger.debug("Subagent owner lookup failed for %s", parent_id, exc_info=True)
            return None
        session_id = parent_id
    logger.warning(
        "Async-delegation completion exceeded %d subagent lineage hops (last session %s); "
        "dropping injection.", MAX_SUBAGENT_OWNER_HOPS, session_id,
    )
    return None


async def _resolve_compression_lineage_target(
    session_db: Any, session_entry: SessionEntry, pinned_session_id: str,
) -> Optional[str]:
    """Return the live compression tip of ``pinned_session_id`` if the route owns that lineage, else None."""
    try:
        target_session_id = await session_db.get_compression_tip(pinned_session_id)
    except Exception:
        logger.debug("Async-delegation compression-tip lookup failed for %s", pinned_session_id, exc_info=True)
        target_session_id = None
    if not target_session_id or target_session_id == pinned_session_id:
        logger.warning(
            "Async-delegation completion pinned to compressed session %s "
            "without a continuation; dropping injection.", pinned_session_id,
        )
        return None
    try:
        tip_row = await session_db.get_session(target_session_id)
    except Exception:
        logger.debug("Async-delegation compression-tip row lookup failed for %s", target_session_id, exc_info=True)
        tip_row = None
    if tip_row is None or tip_row.get("ended_at"):
        logger.warning(
            "Async-delegation compression continuation %s is %s; dropping injection.",
            target_session_id, "unknown" if tip_row is None else "ended",
        )
        return None
    route_owns_lineage = session_entry.session_id in {pinned_session_id, target_session_id}
    if not route_owns_lineage:
        # Across several rotations, accept a stale route only when its own tip is the same live target.
        try:
            route_row = await session_db.get_session(session_entry.session_id)
            route_tip = (
                await session_db.get_compression_tip(session_entry.session_id)
                if route_row is not None
                and route_row.get("ended_at")
                and route_row.get("end_reason") == "compression"
                else None
            )
        except Exception:
            logger.debug("Async-delegation route-tip lookup failed for %s", session_entry.session_id, exc_info=True)
            route_tip = None
        route_owns_lineage = route_tip == target_session_id
    if not route_owns_lineage:
        logger.warning(
            "Async-delegation completion for compression lineage %s -> %s "
            "does not own current route %s; dropping injection.",
            pinned_session_id, target_session_id, session_entry.session_id,
        )
        return None
    return target_session_id
