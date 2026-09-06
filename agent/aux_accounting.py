"""Ambient session-accounting context for auxiliary LLM calls.

Aux calls (vision, compression, title generation, web_extract, session_search, ...) go
through ``agent.auxiliary_client`` which has no session handle, so their usage was
historically discarded. The agent loop publishes ``(session_db, session_id)`` here
(mirroring ``agent.portal_tags``) and the aux client records usage at its single
response-validation chokepoint. ContextVar semantics isolate concurrent agents, propagate
to worker threads via ``tools.thread_context`` and to asyncio tasks automatically.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Optional

logger = logging.getLogger(__name__)

# (session_db, session_id) for the active agent turn, or None outside one.
_accounting: ContextVar[Optional[tuple]] = ContextVar("aux_accounting_context", default=None)

# MoA advisor/aggregator usage is already folded into conversation_loop's
# update_token_counts delta (tokens AND cost); recording it here would double-count.
_EXCLUDED_TASKS = frozenset({"moa_reference", "moa_aggregator"})


def set_accounting_context(session_db: Any, session_id: Optional[str]):
    """Publish the active session's accounting handles; returns the token for ``reset_accounting_context``.

    ``None`` handles (no DB / no session id) clear the context.
    """
    if session_db is None or not session_id:
        return _accounting.set(None)
    return _accounting.set((session_db, session_id))


def reset_accounting_context(token) -> None:
    """Restore the previous accounting context (pair with ``set_...``)."""
    try:
        _accounting.reset(token)
    except Exception:
        _accounting.set(None)


def record_aux_usage(
    response: Any, task: Optional[str], *, provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Record an auxiliary response's token usage against the ambient session.

    Strictly best-effort (accounting must never break an aux call). No-ops outside an
    agent turn, for main-loop-accounted tasks (``_EXCLUDED_TASKS``), or without usage.
    The model is read from ``response.model`` (accurate after aux provider fallback);
    *provider*/*base_url* reflect the originally-resolved route.
    """
    try:
        if not task or task in _EXCLUDED_TASKS:
            return
        ctx = _accounting.get()
        if ctx is None:
            return
        session_db, session_id = ctx
        raw_usage = getattr(response, "usage", None)
        if raw_usage is None:
            return

        from agent.usage_pricing import estimate_usage_cost, normalize_usage

        usage = normalize_usage(raw_usage, provider=provider)
        if not (
            usage.input_tokens or usage.output_tokens
            or usage.cache_read_tokens or usage.cache_write_tokens
            or usage.reasoning_tokens
        ):
            return
        model = str(getattr(response, "model", "") or "") or "unknown"
        estimated_cost = None
        try:
            cost = estimate_usage_cost(model, usage, provider=provider, base_url=base_url)
            if cost.amount_usd is not None:
                estimated_cost = float(cost.amount_usd)
        except Exception:
            logger.debug("Aux usage cost estimation failed", exc_info=True)
        session_db.record_auxiliary_usage(
            session_id, task, model=model, billing_provider=provider, billing_base_url=base_url,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens, cache_write_tokens=usage.cache_write_tokens,
            reasoning_tokens=usage.reasoning_tokens, estimated_cost_usd=estimated_cost,
        )
    except Exception:
        logger.debug("Aux usage recording failed (non-fatal)", exc_info=True)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def get_accounting_context() -> Optional[tuple]:
    """Return ``(session_db, session_id)`` for the active turn, or ``None``."""
    return _accounting.get()
# ---- END PLUGIN-COMPAT ----
