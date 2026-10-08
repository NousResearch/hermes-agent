"""Which fallback candidate may be used: eligibility checks plus the ``pre_fallback_activate`` veto.

Every automatic switch onto a ``fallback_providers`` entry passes through here — the mid-turn
chokepoint :func:`agent.chat_completion_helpers.try_activate_fallback` (``should_skip_fallback_candidate``
then ``fallback_candidate_vetoed``) and the resolution-time walkers that pick a fallback when the
primary's credentials or quota are unusable before the first request
(:func:`hermes_cli.fallback_config.gated_fallback_entries`).

``pre_fallback_activate`` lets a plugin keep a session on its primary — for example "only sessions I
approved may fall back; ask me otherwise". A callback returning ``{"action": "block", "message": ...}``
vetoes the switch and the primary's own error then surfaces as usual; anything else allows it. The hook
runs on the caller thread and is NOT timeout-bounded, so a callback may wait for a human decision.
Dispatch is fail-open like the other plugin hooks: no plugin, a raising callback or a malformed result
never disables fallback. A policy plugin that must fail closed catches its own errors and blocks.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

PRE_FALLBACK_ACTIVATE_HOOK = "pre_fallback_activate"
_MAX_MESSAGE_CHARS = 500


def candidate_pool_exhausted(agent, fb_provider: str, fb_model: str) -> bool:
    """True when every credential the candidate would use sits in an exhaustion cooldown longer
    than the retry loop's longest wait (the 600s Retry-After cap): switching to it only fails the
    turn the same way the primary just did (#89401). A short throttle still gets its chance."""
    pool = getattr(agent, "_credential_pool", None)
    if pool is None or (getattr(pool, "provider", "") or "").strip().lower() != fb_provider:
        try:
            from agent.credential_pool import load_pool
            pool = load_pool(fb_provider)
        except Exception:
            return False
    if pool is None or not pool.has_credentials() or pool.has_available(model=fb_model):
        return False
    until = pool.next_available_at(model=fb_model)
    return until is None or until - time.time() > 600


def should_skip_fallback_candidate(agent, fb: dict, fb_key: tuple, fb_provider: str, fb_model: str, unavailable: set) -> bool:
    """True when the entry is already unavailable, malformed, locally unusable, or resolves
    to the backend that just failed (falling back to it would loop the failure)."""
    from agent.chat_completion_helpers import _fallback_entry_unavailable_without_network
    if fb_key in unavailable:
        logger.debug("Fallback skip: %s previously marked unavailable", fb_key)
        return True
    if not fb_provider or not fb_model:
        return True
    from agent.fallback_cooldown import _is_entitlement_rejected
    if _is_entitlement_rejected(agent, fb_provider, fb_model):
        logger.info("Fallback skip: %s/%s was rejected as unentitled for this account", fb_provider, fb_model)
        return True
    if candidate_pool_exhausted(agent, fb_provider, fb_model):
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
    return False


def _session_env(name: str) -> str:
    from gateway.session_context import get_session_env
    return get_session_env(name, "") or ""


def fallback_veto(
    stage: str,
    *,
    session_id: str = "",
    parent_session_id: str = "",
    platform: str = "",
    job_id: str = "",
    from_provider: str = "",
    from_model: str = "",
    to_provider: str = "",
    to_model: str = "",
    reason: str = "",
) -> Optional[str]:
    """Return the block message when a ``pre_fallback_activate`` callback vetoes the switch, else ``None``.

    ``stage`` is ``"turn"`` for the mid-turn switch and ``"startup"`` for resolution-time fallback.
    ``session_id`` / ``platform`` default to the bound session context when the caller does not know
    them; ``job_id`` is set for cron jobs resolved before their agent exists. The first block wins.
    """
    try:
        from hermes_cli.plugins import has_hook, invoke_hook
        if not has_hook(PRE_FALLBACK_ACTIVATE_HOOK):
            return None
        results = invoke_hook(
            PRE_FALLBACK_ACTIVATE_HOOK, stage=str(stage or ""),
            session_id=str(session_id or "") or _session_env("HERMES_SESSION_ID"),
            parent_session_id=str(parent_session_id or ""),
            platform=str(platform or "") or _session_env("HERMES_SESSION_PLATFORM"),
            job_id=str(job_id or ""), from_provider=str(from_provider or ""), from_model=str(from_model or ""),
            to_provider=str(to_provider or ""), to_model=str(to_model or ""), reason=str(reason or ""),
        )
    except Exception:  # fail-open boundary: a broken plugin layer must never strand a session on a dead primary
        logger.warning("pre_fallback_activate dispatch failed; allowing fallback", exc_info=True)
        return None
    for result in results or ():
        if isinstance(result, dict) and str(result.get("action") or "").strip().lower() == "block":
            message = str(result.get("message") or "").strip()[:_MAX_MESSAGE_CHARS]
            return message or "fallback blocked by a plugin policy"
    return None


def log_fallback_veto(stage: str, to_provider: str, to_model: str, message: str) -> None:
    logger.warning("Fallback to %s/%s blocked by plugin policy (%s): %s", to_provider, to_model, stage, message)


def fallback_candidate_vetoed(agent, reason, fb_provider: str, fb_model: str) -> bool:
    """Mid-turn veto for the chosen candidate. A veto parks the chain index at its end, so later
    recovery paths in the same turn neither ask again nor walk past it; ``restore_primary_runtime``
    resets the index next turn, which asks again if the primary fails again."""
    from agent.chat_completion_helpers import _fallback_reason_text
    message = fallback_veto(
        "turn", session_id=getattr(agent, "session_id", "") or "",
        parent_session_id=getattr(agent, "_parent_session_id", "") or "",
        platform=getattr(agent, "platform", "") or "",
        from_provider=getattr(agent, "provider", "") or "", from_model=getattr(agent, "model", "") or "",
        to_provider=fb_provider, to_model=fb_model, reason=_fallback_reason_text(reason))
    if message is None:
        return False
    log_fallback_veto("turn", fb_provider, fb_model, message)
    agent._buffer_diagnostic_status(f"⚠️ Model fallback to {fb_model} via {fb_provider} was not used: {message}")
    agent._fallback_index = len(agent._fallback_chain)
    return True

