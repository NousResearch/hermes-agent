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


def normalize_fallback_entries(fallback_model: Any) -> list[dict[str, Any]]:
    """Return executable fallback entries, expanding dynamic OpenCode free entries.

    The legacy single-dict format and the newer list format are both accepted.
    A user can request the rotating OpenCode free set either as
    ``{"provider": "opencode-zen", "model": "auto-free"}`` or as the virtual
    provider ``{"provider": "opencode-free"}``.
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
