"""Eligibility gates shared by native reactive and quota-preflight fallback selection."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _should_skip_fallback_candidate(agent, fb: dict, fb_key: tuple, fb_provider: str, fb_model: str, unavailable: set) -> bool:
    """True when the entry is already unavailable, malformed, locally unusable, or resolves
    to the backend that just failed (falling back to it would loop the failure)."""
    from agent.chat_completion_helpers import _candidate_pool_exhausted, _fallback_entry_unavailable_without_network

    if fb_key in unavailable:
        logger.debug("Fallback skip: %s previously marked unavailable", fb_key)
        return True
    if not fb_provider or not fb_model:
        return True
    from agent.fallback_cooldown import _is_entitlement_rejected
    if _is_entitlement_rejected(agent, fb_provider, fb_model):
        logger.info("Fallback skip: %s/%s was rejected as unentitled for this account", fb_provider, fb_model)
        return True
    if _candidate_pool_exhausted(agent, fb_provider, fb_model):
        logger.warning("Fallback skip: %s/%s credential pool is exhausted (every entry in cooldown)", fb_provider, fb_model)
        return True
    local_skip_reason = _fallback_entry_unavailable_without_network(agent, fb)
    if local_skip_reason:
        unavailable.add(fb_key)
        logger.warning("Fallback skip: %s/%s is not locally usable (%s); suppressing for this session", fb_provider, fb_model, local_skip_reason)
        return True
    # Identity semantics (axes, shim aliases, credential surfaces, multi-endpoint pools)
    # are owned by agent.backend_identity — do not re-implement comparisons here.
    # Skip entries that resolve to the same backend that just failed — falling back to it loops the failure.
    # See #22548, #62984, #70893.
    from agent.backend_identity import BackendIdentity, should_skip_candidate
    current_ident = BackendIdentity.build(provider=getattr(agent, "provider", ""),
        model=getattr(agent, "model", ""), base_url=str(getattr(agent, "base_url", "") or ""))
    fb_ident = BackendIdentity.build(provider=fb_provider, model=fb_model, base_url=(fb.get("base_url") or ""))
    if should_skip_candidate(fb_ident, current_ident):
        logger.warning(
            "Fallback skip: chain entry %s/%s resolves to the same backend as the current one (%s)",
            fb_provider, fb_model, current_ident.base_url or current_ident.provider)
        return True
    from agent.quota_preflight import candidate_quota
    return candidate_quota(agent, fb).state == "exhausted"
