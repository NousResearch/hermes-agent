"""Configured provider/model pairs used to narrow every model picker."""

from __future__ import annotations

from typing import Any


def configured_allowed_models(config: Any) -> list[dict[str, str]]:
    """Return valid ``model_catalog.allowed_models`` entries."""
    raw = config.get("allowed_models") if isinstance(config, dict) else config
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        key = (provider.lower(), model.lower())
        if provider and model and key not in seen:
            entries.append({"provider": provider, "model": model})
            seen.add(key)
    return entries


def model_is_allowed(model: str, provider: str, allowed_models: Any) -> bool:
    """Whether a resolved provider/model pair is present in the configured set."""
    allowed = configured_allowed_models(allowed_models)
    if not allowed:
        return True
    target = (str(provider or "").strip().lower(), str(model or "").strip().lower())
    return any((entry["provider"].lower(), entry["model"].lower()) == target for entry in allowed)


def filter_allowed_model_rows(rows: list[dict], allowed_models: Any) -> list[dict]:
    """Return picker rows narrowed to configured provider/model pairs.

    Invalid or empty configuration fails open so a hand-edited config cannot
    make every picker unusable.
    """
    allowed = configured_allowed_models(allowed_models)
    if not allowed:
        return rows
    pairs = {(entry["provider"].lower(), entry["model"].lower()) for entry in allowed}
    filtered: list[dict] = []
    for row in rows:
        slug = str(row.get("slug") or "").strip().lower()
        models = [model for model in row.get("models") or [] if (slug, str(model).lower()) in pairs]
        if not models:
            continue
        narrowed = dict(row)
        narrowed["models"] = models
        narrowed["total_models"] = len(models)
        filtered.append(narrowed)
    return filtered
