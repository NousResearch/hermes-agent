"""Unified selection-time guard registry for model switching surfaces.

Guard modules (``model_cost_guard``, ``model_data_policy_guard``) keep their public APIs — existing
tests and mock patch points remain valid; this module only aggregates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from agent.models_dev import ModelInfo


@dataclass(frozen=True)
class SelectionWarning:
    """A selection-time warning a surface must confirm before applying."""

    kind: str  # "cost" | "data_policy" | "context_cache" | future guard kinds
    title: str
    model: str
    provider: str
    message: str


@dataclass(frozen=True)
class SelectionContext:
    """Live-session facts a surface threads into the registry. Model-only guards (cost, data-policy)
    ignore it; guards about the *switch itself* (context-cache) need the size of the conversation at
    stake and the model it is currently on. Surfaces without a live agent omit it and those guards
    stay silent."""

    context_tokens: Optional[int] = None
    current_model: Optional[str] = None


def selection_context_for_agent(agent: object) -> Optional[SelectionContext]:
    """:class:`SelectionContext` from a live ``AIAgent``: the compressor's measured
    ``last_prompt_tokens`` (what the provider billed on the latest turn), else the session prompt
    counter. ``None`` when no live size is known — the guard then stays silent rather than guess."""
    if agent is None:
        return None
    try:
        cc = getattr(agent, "context_compressor", None)
        tokens = int(getattr(cc, "last_prompt_tokens", 0) or 0) if cc else 0
        if tokens <= 0:
            tokens = int(getattr(agent, "session_prompt_tokens", 0) or 0)
    except Exception:
        tokens = 0
    if tokens <= 0:
        return None
    return SelectionContext(context_tokens=tokens, current_model=getattr(agent, "model", "") or None)


def _wrap(kind: str, title: str, warning, model_name: str, provider: Optional[str]):
    """Lift a raw guard payload into a :class:`SelectionWarning` (None passes through). Duck-typed:
    payloads may carry only ``.message``."""
    if warning is None:
        return None
    return SelectionWarning(
        kind=kind, title=title, model=getattr(warning, "model", model_name),
        provider=getattr(warning, "provider", provider or ""), message=warning.message)


def _cost_guard(
    model_name: str, provider: Optional[str], base_url: Optional[str], api_key: Optional[str],
    model_info: Optional[ModelInfo], ctx: Optional[SelectionContext] = None) -> Optional[SelectionWarning]:
    from hermes_cli.model_cost_guard import expensive_model_warning

    warning = expensive_model_warning(
        model_name, provider=provider, base_url=base_url, api_key=api_key, model_info=model_info)
    return _wrap("cost", "Expensive Model Warning", warning, model_name, provider)


def _data_policy_guard(
    model_name: str, provider: Optional[str], base_url: Optional[str], api_key: Optional[str],
    model_info: Optional[ModelInfo], ctx: Optional[SelectionContext] = None) -> Optional[SelectionWarning]:
    from hermes_cli.model_data_policy_guard import data_training_warning

    warning = data_training_warning(model_name, provider=provider, base_url=base_url)
    return _wrap("data_policy", "Data-Training Tier Warning", warning, model_name, provider)


# Context-token threshold above which a mid-session switch asks for confirmation: providers key
# prompt caches per model, so the first call after a switch re-reads the whole context uncached.
DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD = 100_000


def _context_cache_threshold() -> int:
    """``model.switch_context_confirm_tokens`` from config.yaml (0 disables), else the default."""
    try:
        from hermes_cli.config import load_config

        model_cfg = (load_config() or {}).get("model", {})
        raw = model_cfg.get("switch_context_confirm_tokens") if isinstance(model_cfg, dict) else None
        if raw is not None:
            return max(0, int(raw))
    except Exception:
        pass
    return DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD


def _context_cache_guard(
    model_name: str, provider: Optional[str], base_url: Optional[str], api_key: Optional[str],
    model_info: Optional[ModelInfo], ctx: Optional[SelectionContext] = None) -> Optional[SelectionWarning]:
    """Confirm a mid-session switch that abandons a large cached context. Fires only when the surface
    supplied live facts showing the active context at/above the threshold; smaller sessions, sessions
    with no measured size and same-model re-selects (cache stays warm) are silent."""
    if ctx is None or not ctx.context_tokens:
        return None
    target = (model_name or "").strip()
    current = (ctx.current_model or "").strip()
    if not target or (current and target == current):
        return None
    threshold = _context_cache_threshold()
    tokens = int(ctx.context_tokens)
    if threshold <= 0 or tokens < threshold:
        return None
    message = "\n".join([
        "!!! LARGE CONTEXT MODEL SWITCH !!!",
        "",
        f"This session holds ~{tokens:,} tokens of context.",
        f"Switching to {target} makes the next reply re-read all of it uncached (providers key "
        "prompt caches per model) — a one-time full-price input cost.",
        "",
        f"Threshold: model.switch_context_confirm_tokens (currently {threshold:,}; 0 disables this check).",
        "Confirm only if you intend to switch now."])
    return SelectionWarning(
        kind="context_cache", title="Large Context Switch Warning", model=target,
        provider=(provider or "").strip(), message=message)


# Registry, evaluated in order. Add new guard classes here — never at the
# individual surfaces.
_GUARDS = (_cost_guard, _data_policy_guard, _context_cache_guard)


def selection_warnings(
    model_name: str, *, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_key: Optional[str] = None, model_info: Optional[ModelInfo] = None,
    include_kinds: Optional[Iterable[str]] = None,
    selection_context: Optional[SelectionContext] = None) -> List[SelectionWarning]:
    """Warnings from every registered guard (empty in the common case). ``include_kinds`` restricts
    which kinds are returned; ``selection_context`` carries live-session facts for switch-aware guards.
    Guard exceptions are swallowed — never break model selection."""
    wanted = set(include_kinds) if include_kinds is not None else None
    results: List[SelectionWarning] = []
    for guard in _GUARDS:
        try:
            warning = guard(model_name, provider, base_url, api_key, model_info, selection_context)
        except Exception:
            continue
        if warning is not None and (wanted is None or warning.kind in wanted):
            results.append(warning)
    return results


def combined_message(warnings: List[SelectionWarning]) -> str:
    """One confirm-prompt body for several warnings (one prompt beats two sequential ones)."""
    return "\n\n".join(w.message for w in warnings)


def combined_selection_warning(
    model_name: str, *, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_key: Optional[str] = None, model_info: Optional[ModelInfo] = None,
    selection_context: Optional[SelectionContext] = None,
) -> Optional[SelectionWarning]:
    """Drop-in for ``expensive_model_warning`` call sites: ``None``, the single warning, or a merged
    ``kind="multiple"`` warning stacking every message."""
    warnings = selection_warnings(
        model_name, provider=provider, base_url=base_url, api_key=api_key, model_info=model_info,
        selection_context=selection_context)
    if not warnings:
        return None
    if len(warnings) == 1:
        return warnings[0]
    return SelectionWarning(
        kind="multiple", title="Model Selection Warning", model=warnings[0].model,
        provider=warnings[0].provider, message=combined_message(warnings))
