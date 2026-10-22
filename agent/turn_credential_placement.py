"""Which pooled credential a live session runs its next turn on.

Runs between turns (``restore_primary_runtime``), the one point where switching credentials
cannot tear a request.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.credential_pool_balanced import observe_agent_placement, warm_entry

logger = logging.getLogger(__name__)


def place_credential_between_turns(agent: Any) -> None:
    """Under ``balanced``, put the session where its prompt cache is, or where the most headroom is
    once the cache went cold; under the other strategies, move back onto the credential a quota
    bench rotated the session off."""
    pool = getattr(agent, "_credential_pool", None)
    if pool is not None and getattr(pool, "balanced", False) is True:
        agent._credential_pool_revert_id = None  # balanced decides on cache and headroom, not on history
        _place_session(agent, pool)
    else:
        _revert_credential_rotation(agent)


def _place_session(agent: Any, pool: Any) -> None:
    """A warm session stays on, or returns to, the credential its prompt cache is on (moving it would
    rewrite its whole prefix on another account); a cold one moves to the most headroom."""
    current = getattr(agent, "_credential_pool_entry_id", None)
    warm_on = warm_entry(pool.provider, getattr(agent, "session_id", None))
    if warm_on is not None and warm_on == current:
        return
    model = getattr(agent, "model", None)
    try:
        entry = (pool.reclaim(warm_on, model=model) if warm_on is not None
                 else pool.best_entry_for_cold_session(model=model, base_url=getattr(agent, "base_url", None)))
        if entry is not None and entry.id != current and agent._swap_credential(entry) is not False:
            reason = "back to its warm prompt cache" if warm_on is not None else "most quota headroom"
            logger.info("Session moved to credential %s (%s): %s", entry.id, entry.label, reason)
            current = entry.id
    except Exception:  # the turn still runs on the credential it has
        logger.warning("Between-turn credential placement failed", exc_info=True)
    # Count the session on its credential now rather than at its first response, so sessions
    # placed in the same moment spread instead of all taking the same pick.
    observe_agent_placement(agent, current)


def _revert_credential_rotation(agent) -> None:
    """Move a live session back onto the credential a quota bench rotated it off, once the bench
    lifts. New sessions already do this through ``select()``; without it a long-lived (gateway)
    session keeps billing the fallback for its whole life (#114501). Credential-only: the
    model/base_url/compressor restore stays gated on ``_fallback_activated``."""
    revert_id = getattr(agent, "_credential_pool_revert_id", None)
    if not revert_id:
        return
    pool = getattr(agent, "_credential_pool", None)
    if pool is None or getattr(agent, "_credential_pool_entry_id", None) == revert_id:
        agent._credential_pool_revert_id = None
        return
    try:
        entry = pool.reclaim(revert_id, model=getattr(agent, "model", None))
    except Exception as exc:
        logger.warning("Credential revert check failed: %s", exc)
        return
    if entry is None:
        return  # still cooling down; check again next turn
    if agent._swap_credential(entry) is not False:
        logger.info(
            "Credential %s (%s) available again — reverted pool rotation",
            getattr(entry, "id", "?"), getattr(entry, "label", "?"),
        )
    agent._credential_pool_revert_id = None
