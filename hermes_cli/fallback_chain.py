"""Helpers for normalizing configured fallback provider chains."""

from __future__ import annotations

from typing import Any


def is_opencode_free_fallback_entry(entry: Any) -> bool:
    """Return True when an entry requests the current OpenCode free rotation."""
    if not isinstance(entry, dict):
        return False

    from hermes_cli import models as model_catalog

    raw_provider = str(entry.get("provider") or "").strip().lower()
    provider = model_catalog.normalize_provider(raw_provider)
    model = str(entry.get("model") or "").strip().lower()

    if raw_provider in model_catalog.OPENCODE_FREE_FALLBACK_PROVIDER_ALIASES:
        return True
    return (
        provider == "opencode-zen"
        and model in model_catalog.OPENCODE_FREE_FALLBACK_MODEL_ALIASES
    )


def is_nous_free_fallback_entry(entry: Any) -> bool:
    """Return True when an entry requests the current Nous Portal free rotation."""
    if not isinstance(entry, dict):
        return False

    from hermes_cli import models as model_catalog

    raw_provider = str(entry.get("provider") or "").strip().lower()
    provider = model_catalog.normalize_provider(raw_provider)
    model = str(entry.get("model") or "").strip().lower()

    if raw_provider in model_catalog.NOUS_FREE_FALLBACK_PROVIDER_ALIASES:
        return True
    return provider == "nous" and model in model_catalog.NOUS_FREE_FALLBACK_MODEL_ALIASES


def is_nvidia_auto_fallback_entry(entry: Any) -> bool:
    """Return True when an entry requests rotating across NVIDIA NIM models."""
    if not isinstance(entry, dict):
        return False

    from hermes_cli import models as model_catalog

    provider = model_catalog.normalize_provider(entry.get("provider"))
    model = str(entry.get("model") or "").strip().lower()
    return provider == "nvidia" and model in model_catalog.NVIDIA_AUTO_FALLBACK_MODEL_ALIASES


def is_dynamic_fallback_entry(entry: Any) -> bool:
    """Return True when *entry* expands into multiple runtime routes."""
    return (
        is_opencode_free_fallback_entry(entry)
        or is_nous_free_fallback_entry(entry)
        or is_nvidia_auto_fallback_entry(entry)
    )


def _dedupe_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    provider = str(entry.get("provider") or "").strip().lower()
    model = str(entry.get("model") or "").strip().lower()
    base_url = str(entry.get("base_url") or "").strip().rstrip("/").lower()
    return provider, model, base_url


def _valid_explicit_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("provider") and entry.get("model"))


def _expand_opencode_free_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    from hermes_cli import models as model_catalog

    expanded: list[dict[str, Any]] = []
    for model in model_catalog.opencode_free_model_ids():
        item = dict(entry)
        item["provider"] = "opencode-zen"
        item["model"] = model
        expanded.append(item)
    return expanded


def _expand_nous_free_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    from hermes_cli import models as model_catalog

    expanded: list[dict[str, Any]] = []
    for model in model_catalog.nous_free_model_ids():
        item = dict(entry)
        item["provider"] = "nous"
        item["model"] = model
        expanded.append(item)
    return expanded


def _expand_nvidia_auto_entry(
    entry: dict[str, Any],
    *,
    exclude_model: str | None = None,
) -> list[dict[str, Any]]:
    from hermes_cli import models as model_catalog

    expanded: list[dict[str, Any]] = []
    for model in model_catalog.nvidia_fallback_model_ids(exclude=exclude_model):
        item = dict(entry)
        item["provider"] = "nvidia"
        item["model"] = model
        expanded.append(item)
    return expanded


def normalize_fallback_entries(
    fallback_model: Any,
    *,
    exclude_model: str | None = None,
) -> list[dict[str, Any]]:
    """Return executable fallback entries, expanding dynamic provider entries.

    The legacy single-dict format and the newer list format are both accepted.
    Dynamic entries:

    - OpenCode Zen free: ``{"provider": "opencode-zen", "model": "auto-free"}``
    - Nous Portal free: ``{"provider": "nous", "model": "auto-free"}``
    - NVIDIA NIM rotation: ``{"provider": "nvidia", "model": "auto"}``

  ``exclude_model`` is applied only to NVIDIA auto rotation so the primary
    model is not re-tried as the first fallback hop.
    """
    if isinstance(fallback_model, list):
        raw_entries = list(fallback_model)
    elif isinstance(fallback_model, dict):
        raw_entries = [fallback_model]
    else:
        raw_entries = []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        if is_opencode_free_fallback_entry(raw):
            entries = _expand_opencode_free_entry(raw)
        elif is_nous_free_fallback_entry(raw):
            entries = _expand_nous_free_entry(raw)
        elif is_nvidia_auto_fallback_entry(raw):
            entries = _expand_nvidia_auto_entry(raw, exclude_model=exclude_model)
        elif _valid_explicit_entry(raw):
            entries = [dict(raw)]
        else:
            continue

        for entry in entries:
            key = _dedupe_key(entry)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(entry)

    return normalized
