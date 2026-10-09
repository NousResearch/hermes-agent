"""Application-side fact acquisition for picker/setup model candidates."""

from __future__ import annotations

from collections.abc import Iterable

from models.selection import picker_model_ids as _picker_model_ids


def picker_model_ids(
    provider: str,
    model_ids: Iterable[str],
    *,
    base_url: str = "",
    current_model: str = "",
) -> list[str]:
    """Project caller-owned catalog ids through the canonical selection seam."""

    from hermes_cli.models_validate import provider_allows_model_whitespace

    allow_whitespace = (
        True if not str(provider or "").strip()
        else provider_allows_model_whitespace(provider, base_url)
    )
    return list(
        _picker_model_ids(
            provider,
            model_ids,
            current_model=current_model,
            allow_whitespace=allow_whitespace,
        )
    )


def picker_model_ids_for_provider_data(
    provider_data: dict,
    *,
    current_model: str = "",
    cached_fallback: bool = True,
) -> list[str]:
    """Acquire a provider row's catalog facts, then build canonical candidates."""

    provider = str(provider_data.get("slug") or "")
    model_ids = list(provider_data.get("models") or [])
    if not model_ids and cached_fallback and provider:
        try:
            from hermes_cli.models import cached_provider_model_ids

            model_ids = list(cached_provider_model_ids(provider) or ())
        except Exception:
            model_ids = []

    return picker_model_ids(
        provider,
        model_ids,
        base_url=str(provider_data.get("api_url") or provider_data.get("base_url") or ""),
        current_model=current_model,
    )


def project_picker_rows(rows: Iterable[dict]) -> list[dict]:
    """Return provider rows with canonical selectable/featured model projections."""

    projected: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        original = list(item.get("models") or [])
        models = picker_model_ids(
            str(item.get("slug") or ""),
            original,
            base_url=str(item.get("api_url") or item.get("base_url") or ""),
        )
        item["models"] = models
        total = item.get("total_models")
        if isinstance(total, int):
            item["total_models"] = max(0, total - (len(original) - len(models)))
        featured = item.get("featured_models")
        if isinstance(featured, list):
            item["featured_models"] = picker_model_ids(
                str(item.get("slug") or ""),
                featured,
                base_url=str(item.get("api_url") or item.get("base_url") or ""),
            )
        projected.append(item)
    return projected


__all__ = [
    "picker_model_ids",
    "picker_model_ids_for_provider_data",
    "project_picker_rows",
]
