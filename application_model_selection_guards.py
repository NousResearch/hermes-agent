"""Shared application confirmation policy for model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True, slots=True)
class SelectionWarning:
    kind: str
    title: str
    model: str
    provider: str
    message: str


@dataclass(frozen=True, slots=True)
class SelectionContext:
    context_tokens: Optional[int] = None
    current_model: Optional[str] = None


def selection_context_for_agent(agent: object) -> Optional[SelectionContext]:
    if agent is None:
        return None
    try:
        compressor = getattr(agent, "context_compressor", None)
        tokens = int(getattr(compressor, "last_prompt_tokens", 0) or 0) if compressor else 0
        if tokens <= 0:
            tokens = int(getattr(agent, "session_prompt_tokens", 0) or 0)
    except Exception:
        tokens = 0
    if tokens <= 0:
        return None
    return SelectionContext(tokens, getattr(agent, "model", "") or None)


def _wrap(kind, title, warning, model, provider):
    if warning is None:
        return None
    return SelectionWarning(
        kind,
        title,
        getattr(warning, "model", model),
        getattr(warning, "provider", provider or ""),
        warning.message,
    )


def _cost_warning(model, provider, base_url, api_key, model_info, _ctx):
    from hermes_cli.model_cost_guard import expensive_model_warning

    return _wrap(
        "cost",
        "Expensive Model Warning",
        expensive_model_warning(
            model,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model_info=model_info,
        ),
        model,
        provider,
    )


def _data_warning(model, provider, base_url, _api_key, _model_info, _ctx):
    from hermes_cli.model_data_policy_guard import data_training_warning

    return _wrap(
        "data_policy",
        "Data-Training Tier Warning",
        data_training_warning(model, provider=provider, base_url=base_url),
        model,
        provider,
    )


def _context_warning(model, provider, _base_url, _api_key, _model_info, ctx, threshold):
    if ctx is None or not ctx.context_tokens:
        return None
    target = str(model or "").strip()
    current = str(ctx.current_model or "").strip()
    tokens = int(ctx.context_tokens)
    if not target or (current and target == current) or threshold <= 0 or tokens < threshold:
        return None
    message = "\n".join([
        "!!! LARGE CONTEXT MODEL SWITCH !!!",
        "",
        f"This session holds ~{tokens:,} tokens of context.",
        (
            f"Switching to {target} makes the next reply re-read all of it uncached "
            "(providers key prompt caches per model) — a one-time full-price input cost."
        ),
        "",
        (
            "Threshold: model.switch_context_confirm_tokens "
            f"(currently {threshold:,}; 0 disables this check)."
        ),
        "Confirm only if you intend to switch now.",
    ])
    return SelectionWarning("context_cache", "Large Context Switch Warning", target, provider or "", message)


DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD = 100_000


def _context_threshold_from_config() -> int:
    """Read the active profile's cache-switch confirmation threshold (0 disables)."""
    try:
        from hermes_cli.config import load_config
        config = load_config() or {}
        section = config.get("model") if isinstance(config, dict) else None
        value = section.get("switch_context_confirm_tokens") if isinstance(section, dict) else None
        return max(0, int(value)) if value is not None else DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD
    except Exception:
        return DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD


def _context_cache_guard(model, provider, base_url, api_key, model_info, ctx):
    """The single context guard policy, including active profile threshold."""
    return _context_warning(
        model, provider, base_url, api_key, model_info, ctx, _context_threshold_from_config(),
    )


def selection_warnings(
    model_name: str, *, provider: str = "", base_url: str = "", api_key: str = "",
    model_info=None, selection_context: Optional[SelectionContext] = None,
    include_kinds: Optional[Iterable[str]] = None,
    context_threshold: int | None = None,
) -> list[SelectionWarning]:
    """One application-owned, ordered warning registry for every selection surface."""
    wanted = set(include_kinds) if include_kinds is not None else None
    warnings = []
    for kind, fn in (("cost", _cost_warning), ("data_policy", _data_warning)):
        if wanted is not None and kind not in wanted:
            continue
        try:
            warning = fn(model_name, provider, base_url, api_key, model_info, selection_context)
        except Exception:
            warning = None
        if warning is not None:
            warnings.append(warning)
    if wanted is None or "context_cache" in wanted:
        try:
            threshold = _context_threshold_from_config() if context_threshold is None else context_threshold
            warning = _context_warning(
                model_name, provider, base_url, api_key, model_info, selection_context, threshold,
            )
        except Exception:
            warning = None
        if warning is not None:
            warnings.append(warning)
    return warnings


def combined_message(warnings: Iterable[SelectionWarning]) -> str:
    return "\n\n".join(warning.message for warning in warnings)


def combined_selection_warning(
    model_name: str, *, provider: str = "", base_url: str = "", api_key: str = "",
    model_info=None, selection_context: Optional[SelectionContext] = None,
    context_threshold: int | None = None,
    include_kinds: Optional[Iterable[str]] = None,
) -> Optional[SelectionWarning]:
    warnings = selection_warnings(
        model_name, provider=provider, base_url=base_url, api_key=api_key,
        model_info=model_info, selection_context=selection_context,
        include_kinds=include_kinds, context_threshold=context_threshold,
    )
    if not warnings:
        return None
    if len(warnings) == 1:
        return warnings[0]
    return SelectionWarning(
        "multiple", "Model Selection Warning", warnings[0].model,
        warnings[0].provider, combined_message(warnings),
    )


__all__ = [
    "DEFAULT_CONTEXT_CACHE_SWITCH_THRESHOLD",
    "SelectionContext", "SelectionWarning", "_context_cache_guard",
    "selection_warnings", "combined_message", "combined_selection_warning",
    "selection_context_for_agent",
]
