"""Unified selection-time guard registry for model switching surfaces.

Guard modules (``model_cost_guard``, ``model_data_policy_guard``) keep their public APIs — existing
tests and mock patch points remain valid; this module only aggregates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from agent.backend_identity import same_route
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
    # Which class of evidence ``context_tokens`` is. The names are the switch summary's
    # (``context_switch_guard.SIZE_FROM_*``) because both surfaces describe one assessment: only a
    # provider reading taken on this session's own route is a context size, a display seed is a
    # local estimate, and the session prompt counter is a cumulative total.
    context_tokens_source: str = "measured"
    current_model: Optional[str] = None
    # The rest of the route the session is on. A model string alone cannot tell a reselect of the
    # warm deployment from an endpoint or alias move onto a different one, and the two answers
    # differ, so the surface supplies the whole route and the guard resolves it the way the switch
    # summary does.
    current_provider: Optional[str] = None
    current_base_url: Optional[str] = None


def selection_context_for_agent(agent: object) -> Optional[SelectionContext]:
    """:class:`SelectionContext` from a live ``AIAgent``: how much conversation the switch abandons,
    the route it was taken on, and the evidence the figure came from — the compressor's provider
    reading (``last_real_prompt_tokens``, what the latest turn was billed) is a measurement, the
    display seed written from a local estimate is an estimate, and the session prompt counter is a
    cumulative total. ``None`` when no size is known — the guard then stays silent rather than
    guess."""
    if agent is None:
        return None
    source = "measured"
    try:
        cc = getattr(agent, "context_compressor", None)
        tokens = int(getattr(cc, "last_real_prompt_tokens", 0) or 0) if cc else 0
        if tokens <= 0:
            # ``update_model`` clears the real reading, so a seed written afterwards
            # (``maybe_seed_preflight_display_tokens``) is the compressor's only figure and it states
            # a local estimate. Only then does the session counter stand in, and as a total, not a
            # size.
            tokens = int(getattr(cc, "last_prompt_tokens", 0) or 0) if cc else 0
            source = "estimate" if tokens > 0 else "counter"
        if tokens <= 0:
            tokens = int(getattr(agent, "session_prompt_tokens", 0) or 0)
    except Exception:
        tokens, source = 0, "measured"
    if tokens <= 0:
        return None
    return SelectionContext(
        context_tokens=tokens, context_tokens_source=source,
        current_model=getattr(agent, "model", "") or None,
        current_provider=getattr(agent, "provider", "") or None,
        current_base_url=getattr(agent, "base_url", "") or None)


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
    supplied live facts showing the session at/above the threshold; smaller sessions, sessions with
    no size at all and a reselect of the deployment the session already runs on (cache stays warm)
    are silent. The figure is quoted with the evidence class it was read from — a display seed or a
    session total is not the size of what the next reply re-reads — and the cost of the move is
    stated as the condition it is, because nothing here observes whether the provider still holds a
    cache for this session or how it bills a re-read."""
    if ctx is None or not ctx.context_tokens:
        return None
    target = (model_name or "").strip()
    if not target:
        return None
    # Whether this is a switch at all is the *resolved* transition — model, provider label and
    # endpoint — not the model string on its own, and it is asked through the same owner the switch
    # summary reads (``agent.backend_identity.same_route``): an endpoint-only move under one label is
    # another route, and two aliases at one URL and model are the deployment that already served the
    # session. Resolving it independently here is what let the two screens describe one transition
    # two ways.
    if same_route(
            old_model=(ctx.current_model or "").strip(),
            old_provider=(ctx.current_provider or "").strip(),
            old_base_url=(ctx.current_base_url or "").strip(),
            new_model=target, new_provider=(provider or "").strip(),
            new_base_url=(base_url or "").strip()):
        return None
    threshold = _context_cache_threshold()
    tokens = int(ctx.context_tokens)
    if threshold <= 0 or tokens < threshold:
        return None
    # The size sentence is the summary's, quoted through the same evidence contract: a provider
    # reading for this session's own route is the context size, and anything weaker says what it is
    # instead of claiming to be what the next reply re-reads.
    from hermes_cli.context_switch_guard import SIZE_FROM_MEASURED, size_label

    size = (f"This session holds ~{tokens:,} tokens of context."
            if ctx.context_tokens_source == SIZE_FROM_MEASURED
            else f"{size_label(tokens, ctx.context_tokens_source)}.")
    message = "\n".join([
        "!!! LARGE CONTEXT MODEL SWITCH !!!",
        "",
        size,
        f"Switching to {target} moves off the route this session was served on (providers key prompt "
        f"caches per model), so if {target} has not served it there is no warm prefix cache and the "
        f"next reply reads the conversation as uncached input.",
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
