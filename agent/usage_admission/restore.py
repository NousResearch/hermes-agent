"""Usage-admission checks for restoring a session onto its primary provider."""

from __future__ import annotations

import logging
from typing import Any

from agent.usage_admission import policy

logger = logging.getLogger(__name__)


def snapshot_key_came_from_pool(pool: Any, rt: dict[str, Any]) -> bool:
    """A fallback without a pool clears the live binding, but the snapshot's pool-entry provenance says
    the saved key came from a pool: reload it and re-run admission instead of bypassing a usage ceiling."""
    return pool is None and bool(str(rt.get("credential_pool_entry_id") or "").strip())


def primary_excluded_by_usage_policy(agent: Any, pool: Any, primary_provider: str, primary_model: str) -> bool:
    """True when ``primary_provider`` has configured usage ceilings, every pool entry is unavailable, and
    at least one is unavailable because of policy (not cooldown alone). Logs the reason."""
    if pool is None or not policy.get_pool_usage_limits(primary_provider):
        return False
    refresh = getattr(pool, "_refresh_usage_admission_cache", None)
    if callable(refresh):
        refresh()
    if pool.has_available(model=primary_model or None) or not policy.usage_admission_policy_denied(primary_provider):
        return False
    logger.info(
        "Primary %s is excluded by credential usage policy; staying on fallback %s/%s",
        primary_provider or "?", agent.provider, agent.model,
    )
    return True


def revert_credential_rotation(agent) -> None:
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
        return  # still cooling down, or over its usage ceiling (``reclaim`` applies it); check again next turn
    if agent._swap_credential(entry) is not False:
        logger.info(
            "Credential %s (%s) available again — reverted pool rotation",
            getattr(entry, "id", "?"), getattr(entry, "label", "?"),
        )
    agent._credential_pool_revert_id = None
