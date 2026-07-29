"""Ambient session-accounting context for auxiliary LLM calls.

Auxiliary calls (vision, compression, title generation, web_extract,
session_search, ...) funnel through ``agent.auxiliary_client`` which has no
session handle — so their token usage was historically discarded, leaving
dashboard analytics blind to aux model spend (issue #23270).

Instead of threading ``session_db``/``session_id`` parameters through every
aux call site, the agent loop publishes them here (mirroring the Nous Portal
conversation context in ``agent.portal_tags``) and the auxiliary client
records usage at its single response-validation chokepoint.

ContextVar semantics give us the right isolation for free:

* concurrent agents in one process (gateway sessions, delegate subagents)
  never see each other's accounting context;
* worker threads spawned via ``tools.thread_context.propagate_context_to_thread``
  (MoA fan-out, background review) inherit the parent turn's context;
* asyncio tasks inherit the context of the code that created them.

MoA reference/aggregator slots are explicitly EXCLUDED from recording:
``agent/conversation_loop.py`` already folds MoA advisor usage and cost into
the main loop's ``update_token_counts`` delta, so recording them here would
double-count (see ``_EXCLUDED_TASKS``).
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Optional

logger = logging.getLogger(__name__)

# (session_db, session_id) for the active agent turn, or None outside one.
_accounting: ContextVar[Optional[tuple]] = ContextVar(
    "aux_accounting_context", default=None
)

# Aux tasks whose usage is already accounted by the main loop — recording
# them here would double-count. MoA advisor/aggregator usage is folded into
# conversation_loop's update_token_counts delta (tokens AND cost).
_EXCLUDED_TASKS = frozenset({"moa_reference", "moa_aggregator"})


def set_accounting_context(session_db: Any, session_id: Optional[str]):
    """Publish the active session's accounting handles for aux usage recording.

    Called by the agent loop at turn entry. Returns the ContextVar token so
    callers can ``reset_accounting_context(token)`` on turn exit. Publishing
    ``None`` handles (no DB / no session id) clears the context.
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


def get_accounting_context() -> Optional[tuple]:
    """Return ``(session_db, session_id)`` for the active turn, or ``None``."""
    return _accounting.get()


def record_aux_usage(
    response: Any,
    task: Optional[str],
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Record an auxiliary response's token usage against the ambient session.

    Called from the auxiliary client's response-validation chokepoint. Strictly
    best-effort: any failure is swallowed (accounting must never break an aux
    call). No-ops when:

    * no accounting context is published (call is outside any agent turn),
    * the task is main-loop-accounted (MoA slots — see ``_EXCLUDED_TASKS``),
    * the response carries no usage object.

    The model is read from ``response.model`` (accurate even after the aux
    client's provider-fallback chains); *provider*/*base_url* reflect the
    originally-resolved route and are best-effort.
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
            cost = estimate_usage_cost(
                model, usage, provider=provider, base_url=base_url
            )
            if cost.amount_usd is not None:
                estimated_cost = float(cost.amount_usd)
        except Exception:
            logger.debug("Aux usage cost estimation failed", exc_info=True)

        session_db.record_auxiliary_usage(
            session_id,
            task,
            model=model,
            billing_provider=provider,
            billing_base_url=base_url,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            estimated_cost_usd=estimated_cost,
        )
    except Exception:
        logger.debug("Aux usage recording failed (non-fatal)", exc_info=True)
