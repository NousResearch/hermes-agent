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

    kind: str  # "cost" | "data_policy" | future guard kinds
    title: str
    model: str
    provider: str
    message: str


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
    model_info: Optional[ModelInfo]) -> Optional[SelectionWarning]:
    from hermes_cli.model_cost_guard import expensive_model_warning

    warning = expensive_model_warning(
        model_name, provider=provider, base_url=base_url, api_key=api_key, model_info=model_info)
    return _wrap("cost", "Expensive Model Warning", warning, model_name, provider)


def _data_policy_guard(
    model_name: str, provider: Optional[str], base_url: Optional[str], api_key: Optional[str],
    model_info: Optional[ModelInfo]) -> Optional[SelectionWarning]:
    from hermes_cli.model_data_policy_guard import data_training_warning

    warning = data_training_warning(model_name, provider=provider, base_url=base_url)
    return _wrap("data_policy", "Data-Training Tier Warning", warning, model_name, provider)


# Registry, evaluated in order. Add new guard classes here — never at the
# individual surfaces.
_GUARDS = (_cost_guard, _data_policy_guard)


def selection_warnings(
    model_name: str, *, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_key: Optional[str] = None, model_info: Optional[ModelInfo] = None,
    include_kinds: Optional[Iterable[str]] = None) -> List[SelectionWarning]:
    """Warnings from every registered guard (empty in the common case). ``include_kinds`` restricts
    which kinds are returned. Guard exceptions are swallowed — never break model selection."""
    wanted = set(include_kinds) if include_kinds is not None else None
    results: List[SelectionWarning] = []
    for guard in _GUARDS:
        try:
            warning = guard(model_name, provider, base_url, api_key, model_info)
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
) -> Optional[SelectionWarning]:
    """Drop-in for ``expensive_model_warning`` call sites: ``None``, the single warning, or a merged
    ``kind="multiple"`` warning stacking every message."""
    warnings = selection_warnings(
        model_name, provider=provider, base_url=base_url, api_key=api_key, model_info=model_info)
    if not warnings:
        return None
    if len(warnings) == 1:
        return warnings[0]
    return SelectionWarning(
        kind="multiple", title="Model Selection Warning", model=warnings[0].model,
        provider=warnings[0].provider, message=combined_message(warnings))
