"""Configured compaction budgets and per-delivery opt-in for external context engines."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def configured_compression_budget(
    model: str, context_length: int, *, provider: str = "", threshold_percent: float = 0.50,
    model_thresholds: dict | None = None, max_tokens: int | None = None,
    threshold_tokens_cap: int | None = None, aux_context_ceiling: int | None = None,
) -> tuple[int, float, float, int]:
    """Pure configured policy: capacity, base/effective percentages, and trigger.

    Shared with the built-in compressor; never constructs clients or queries a provider.
    """
    from agent.context_compressor import ContextCompressor as CC, resolve_model_threshold
    base = resolve_model_threshold(model, model_thresholds or {}, threshold_percent, provider)
    effective = CC._effective_threshold_percent(context_length, base)
    reserve = CC._coerce_max_tokens(max_tokens)
    capacity = CC._effective_input_window(context_length, reserve)
    trigger = CC._compute_threshold_tokens(context_length, effective, reserve)
    cap = CC._coerce_threshold_tokens_cap(threshold_tokens_cap)
    if cap is not None:
        trigger = min(trigger, cap, context_length)
    if isinstance(aux_context_ceiling, int) and aux_context_ceiling > 0:
        trigger = min(trigger, aux_context_ceiling)
    return capacity, base, effective, trigger


def update_runtime_context_compressor(agent: Any, context_length: int, *, reason: str, **runtime: Any) -> None:
    """Refresh the active route, clearing a previous route's implicit output reservation."""
    from agent.context_compressor import ContextCompressor
    from agent.agent_init import _compressor_max_tokens
    compressor = agent.context_compressor
    if isinstance(compressor, ContextCompressor):
        compressor.update_model(context_length=context_length, max_tokens=_compressor_max_tokens(agent) or 0, **runtime)
    else:
        compressor.update_model(context_length=context_length, **runtime)
        apply_context_engine_compression_budget(agent, context_length, reason=reason)


def apply_auxiliary_context_ceiling(agent: Any, aux_context: int) -> bool:
    """Install a feasibility ceiling only if the external engine accepts that delivery."""
    from agent.context_compressor import ContextCompressor
    compressor = agent.context_compressor
    if isinstance(compressor, ContextCompressor):
        compressor._aux_context_ceiling = compressor.threshold_tokens = aux_context
        return True
    previous_ceiling = getattr(agent, "_context_engine_aux_context_ceiling", None)
    agent._context_engine_aux_context_ceiling = aux_context
    accepted = False
    try:
        accepted = apply_context_engine_compression_budget(
            agent, compressor.context_length, reason="auxiliary_feasibility")
        return accepted
    finally:
        if not accepted:
            agent._context_engine_aux_context_ceiling = previous_ceiling


def apply_context_engine_compression_budget(agent: Any, context_length: int, *, reason: str) -> bool:
    """Only singleton True transfers host threshold ownership for THIS delivery.

    Same-runtime window corrections retain the auxiliary ceiling. Runtime identity changes
    invalidate it just as ContextCompressor.update_model does; feasibility re-probes separately.
    Declining a later delivery relinquishes ownership, not the engine's previously accepted state.
    """
    from agent.context_compressor import ContextCompressor
    compressor = getattr(agent, "context_compressor", None)
    agent._context_engine_compression_budget_accepted = False
    setter = getattr(compressor, "set_compression_budget", None)
    if not callable(setter) or isinstance(compressor, ContextCompressor) or context_length <= 0:
        return False
    runtime = tuple(getattr(agent, field, "") for field in ("model", "provider", "base_url", "api_mode"))
    if getattr(agent, "_context_engine_budget_runtime", None) != runtime:
        agent._context_engine_aux_context_ceiling = None
    agent._context_engine_budget_runtime = runtime
    from agent.agent_init import _compressor_max_tokens
    try:
        capacity, _, _, trigger = configured_compression_budget(
            agent.model, context_length, provider=getattr(agent, "provider", ""),
            threshold_percent=getattr(agent, "_compression_threshold_percent", 0.50),
            model_thresholds=getattr(agent, "_compression_model_thresholds", {}),
            max_tokens=_compressor_max_tokens(agent),
            threshold_tokens_cap=getattr(agent, "_compression_threshold_tokens_cap", None),
            aux_context_ceiling=getattr(agent, "_context_engine_aux_context_ceiling", None),
        )
    except (ValueError, TypeError, OverflowError):
        # Optional host policy must not undo a successful engine-owned runtime update.
        logger.warning("Cannot derive optional context engine compression budget", exc_info=True)
        return False
    try:
        accepted = setter(capacity, trigger, reason=reason) is True
    except Exception:
        logger.warning("Context engine rejected host compression budget", exc_info=True)
        accepted = False
    agent._context_engine_compression_budget_accepted = accepted
    if accepted:
        compressor.threshold_tokens = trigger
    return accepted
