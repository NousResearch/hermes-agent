"""Credential-pool rehydration and rotation revert for cached agents, split out of agent_runtime_helpers.

Extracted so ``agent.agent_runtime_helpers`` stays under its code-health
FILE_LINES cap. The helpers late-import the pool accessors from the facade so a
test patching ``agent.agent_runtime_helpers.load_pool`` still intercepts.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agent.agent_runtime_helpers")


def _rehydrate_credential_pool(agent, current_provider: str):
    """Attach the live pool to a cached agent whose active key is in it.

    Cached gateway agents can predate pool attachment after a routing/config
    change. Rehydrate only when the active key is itself in the live pool;
    never replace an explicit unpooled credential with a different account.
    Returns the attached pool, or None when nothing matched.
    """
    from agent.agent_runtime_helpers import _ra, credential_pool_matches_provider, load_pool

    pool = agent._credential_pool
    if pool is not None:
        return pool
    try:
        live_pool = load_pool(current_provider) if current_provider else None
    except Exception:
        logger.debug("credential pool rehydration read failed", exc_info=True)
        live_pool = None
    active_key = getattr(agent, "api_key", None)
    if (
        live_pool is not None
        and active_key
        and credential_pool_matches_provider(
            live_pool, current_provider, base_url=getattr(agent, "base_url", None)
        )
    ):
        matching_entry = next(
            (entry for entry in live_pool.entries() if getattr(entry, "runtime_api_key", None) == active_key),
            None,
        )
        if matching_entry is not None:
            agent._credential_pool = pool = live_pool
            if not getattr(agent, "_credential_pool_entry_id", None):
                agent._credential_pool_entry_id = getattr(matching_entry, "id", None)
            _ra().logger.info(
                "Rehydrated credential pool for cached %s agent using active entry %s",
                current_provider, getattr(matching_entry, "id", "?"),
            )
    return pool


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
