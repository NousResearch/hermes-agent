"""Configured model ids the provider's catalog no longer lists.

A provider retires a model id — OpenRouter promotes a free SKU to paid, renames a
variant, or drops a SKU altogether — and a config that still names it fails at the
first request with HTTP 404. Nothing in the install says so: the picker does not show
the model, the scheduled jobs carry the failure on their own, and the health check is
silent about the cause.

This walks the config's model slots and compares every id with the provider's own
catalog cache (``$HERMES_HOME/provider_models_cache.json``), so the check costs no
network call and works offline. A provider whose catalog is not cached is skipped:
an unknown catalog is not evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# Providers whose ids are namespaced by their own slug (an OpenRouter id is
# "vendor/model"), so a slot's provider is known from the slot itself.
_NAMESPACED = {"openrouter", "nous"}


@dataclass(frozen=True)
class ModelRef:
    """One model id the config asks for, and the provider that must serve it."""

    slot: str
    model_id: str
    provider: str


@dataclass(frozen=True)
class UnavailableModel:
    """A configured id the provider's catalog does not carry."""

    ref: ModelRef
    catalog_size: int

    def render(self) -> str:
        return (f"{self.ref.slot}: {self.ref.model_id!r} is not listed by "
                f"{self.ref.provider} ({self.catalog_size} models in its catalog)")


def _slot_provider(section: Any, key: str = "provider") -> Optional[str]:
    if not isinstance(section, dict):
        return None
    value = section.get(key)
    if not isinstance(value, str):
        return None
    provider = value.strip().lower()
    return provider or None


def _model_of(section: Any) -> Optional[str]:
    if not isinstance(section, dict):
        return None
    value = section.get("model")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _add(out: list[ModelRef], slot: str, model_id: Optional[str], provider: Optional[str]) -> None:
    if not model_id or not provider:
        return
    if provider in ("auto", "none", ""):
        return
    out.append(ModelRef(slot=slot, model_id=model_id, provider=provider))


def _reference_models(out: list[ModelRef], section: Any, prefix: str) -> None:
    """``reference_models`` is a list of ``{provider, model}`` entries."""
    if not isinstance(section, list):
        return
    for index, entry in enumerate(section):
        if not isinstance(entry, dict):
            continue
        _add(out, f"{prefix}[{index}].model", _model_of(entry), _slot_provider(entry))


def configured_models(config: dict[str, Any]) -> list[ModelRef]:
    """Every model id the config asks for, with the provider that must serve it.

    Slots mirror the ones the runtime reads: ``model.default`` is the chat model (the
    runtime resolver and ~14 other readers select it), ``model.aliases`` maps a short
    name onto a model, ``auxiliary`` holds the helper models, and ``moa`` holds the
    aggregator and its reference models.
    """
    out: list[ModelRef] = []
    if not isinstance(config, dict):
        return out

    model = config.get("model")
    if isinstance(model, dict):
        provider = _slot_provider(model)
        default = model.get("default")
        if isinstance(default, str) and default.strip():
            _add(out, "model.default", default.strip(), provider)
        aliases = model.get("aliases")
        if isinstance(aliases, dict):
            for name, target in aliases.items():
                if isinstance(target, str) and target.strip():
                    _add(out, f"model.aliases.{name}", target.strip(), provider)

    auxiliary = config.get("auxiliary")
    if isinstance(auxiliary, dict):
        # The OpenRouter auxiliary is pinned by its own key, not by a nested provider.
        _add(out, "auxiliary.openrouter_model", auxiliary.get("openrouter_model"), "openrouter")
        for name, section in auxiliary.items():
            if name == "openrouter_model" or not isinstance(section, dict):
                continue
            _add(out, f"auxiliary.{name}.model", _model_of(section), _slot_provider(section))

    moa = config.get("moa")
    if isinstance(moa, dict):
        _add(out, "moa.aggregator.model", _model_of(moa.get("aggregator")),
             _slot_provider(moa.get("aggregator")))
        _reference_models(out, moa.get("reference_models"), "moa.reference_models")
        presets = moa.get("presets")
        if isinstance(presets, dict):
            for name, preset in presets.items():
                if not isinstance(preset, dict):
                    continue
                _add(out, f"moa.presets.{name}.aggregator.model",
                     _model_of(preset.get("aggregator")), _slot_provider(preset.get("aggregator")))
                _reference_models(out, preset.get("reference_models"),
                                  f"moa.presets.{name}.reference_models")

    delegation = config.get("delegation")
    if isinstance(delegation, dict):
        _add(out, "delegation.model", _model_of(delegation), _slot_provider(delegation))

    return out


def _cache_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "provider_models_cache.json"


def cached_catalogs() -> dict[str, set[str]]:
    """``provider -> model ids`` from the install's provider catalog cache.

    The cache is the install's own view of each provider (refreshed in the background,
    served stale for up to a week), so the check needs no network. A provider with no
    cached row is absent from the result: an unknown catalog is not evidence.
    """
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    catalogs: dict[str, set[str]] = {}
    for provider, row in data.items():
        if not isinstance(row, dict):
            continue
        models = row.get("models")
        if isinstance(models, list) and models:
            catalogs[str(provider).strip().lower()] = {str(m) for m in models}
    return catalogs


def unavailable_models(
    config: dict[str, Any], catalogs: Optional[dict[str, set[str]]] = None
) -> list[UnavailableModel]:
    """The configured ids the provider's catalog does not carry."""
    if catalogs is None:
        catalogs = cached_catalogs()
    out: list[UnavailableModel] = []
    seen: set[tuple[str, str]] = set()
    for ref in configured_models(config):
        catalog = catalogs.get(ref.provider)
        if not catalog:
            continue  # the provider's catalog is unknown here
        if ref.model_id in catalog:
            continue
        key = (ref.slot, ref.model_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(UnavailableModel(ref=ref, catalog_size=len(catalog)))
    return out


def iter_slots(config: dict[str, Any]) -> Iterable[ModelRef]:
    """The configured model slots, provider included (for callers that judge them)."""
    return configured_models(config)
