"""Credential rotation recovery for live agent sessions."""

import logging

logger = logging.getLogger("agent.agent_runtime_helpers")


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
