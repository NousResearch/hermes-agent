"""Provider-scoped model filtering for picker inventories."""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Iterable, Mapping


def _patterns_for_provider(
    provider: str,
    excluded_models: object,
) -> tuple[str, ...]:
    if not isinstance(excluded_models, Mapping):
        return ()

    provider_key = str(provider or "").strip().lower()
    if not provider_key:
        return ()

    raw_patterns: object = None
    for key, value in excluded_models.items():
        if str(key or "").strip().lower() == provider_key:
            raw_patterns = value
            break

    if not isinstance(raw_patterns, list):
        return ()

    return tuple(
        pattern.strip().lower()
        for pattern in raw_patterns
        if isinstance(pattern, str) and pattern.strip()
    )


def filter_model_ids(
    provider: str,
    model_ids: Iterable[str],
    excluded_models: object,
) -> list[str]:
    """Remove IDs matching this provider's case-insensitive glob rules."""
    models = list(model_ids)
    patterns = _patterns_for_provider(provider, excluded_models)
    if not patterns:
        return models

    return [
        model
        for model in models
        if not any(
            fnmatchcase(str(model).lower(), pattern)
            for pattern in patterns
        )
    ]


def filter_provider_rows(
    rows: Iterable[dict],
    excluded_models: object,
) -> list[dict]:
    """Apply provider-scoped exclusions without mutating inventory rows."""
    filtered_rows: list[dict] = []
    for row in rows:
        models = row.get("models") or []
        filtered_models = filter_model_ids(
            str(row.get("slug") or ""),
            models,
            excluded_models,
        )
        if filtered_models == list(models):
            filtered_rows.append(row)
            continue

        filtered_row = dict(row)
        filtered_row["models"] = filtered_models
        filtered_row["total_models"] = len(filtered_models)
        filtered_rows.append(filtered_row)

    return filtered_rows
