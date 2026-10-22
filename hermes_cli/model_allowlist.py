"""Configured provider/model pairs used to narrow every model picker."""

from __future__ import annotations

from typing import Any


def _raw_allowed_models(config: Any) -> Any:
    """The configured value: a list of pairs, or a present non-list."""
    if isinstance(config, dict) and "allowed_models" in config:
        return config.get("allowed_models")
    return config


def allowed_models_restrict(config: Any) -> bool:
    """True when a non-empty ``allowed_models`` value is present.

    Absent / explicit ``[]`` keep the legacy unrestricted catalog. A present
    non-empty value (including all-invalid entries or a non-list) is restrictive
    so a malformed allowlist cannot silently widen to the full catalog.
    """
    raw = _raw_allowed_models(config)
    if raw is None:
        return False
    if isinstance(raw, list):
        return len(raw) > 0
    return True


def configured_allowed_models(config: Any) -> list[dict[str, str]]:
    """Return valid ``model_catalog.allowed_models`` entries."""
    raw = _raw_allowed_models(config)
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
    if not allowed_models_restrict(allowed_models):
        return True
    allowed = configured_allowed_models(allowed_models)
    target = (str(provider or "").strip().lower(), str(model or "").strip().lower())
    return any((entry["provider"].lower(), entry["model"].lower()) == target for entry in allowed)


def filter_allowed_model_rows(rows: list[dict], allowed_models: Any) -> list[dict]:
    """Return picker rows narrowed to configured provider/model pairs.

    Absent or explicit empty configuration fails open. A present non-empty
    allowlist is restrictive even when every entry is invalid.
    """
    if not allowed_models_restrict(allowed_models):
        return rows
    pairs = {
        (entry["provider"].lower(), entry["model"].lower())
        for entry in configured_allowed_models(allowed_models)
    }
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
