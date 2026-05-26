"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

from typing import Any


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


_GEMINI_35_FLASH_PRIMARY_IDS = {
    "gemini-3.5-flash",
    "google/gemini-3.5-flash",
}


def _gemini_preview_fallback_for_primary(provider: str, model: str) -> dict[str, Any] | None:
    """Return the protective Gemini 3 Flash preview fallback, if applicable."""

    provider = str(provider or "").strip().lower()
    model = str(model or "").strip()
    model_l = model.lower()
    if model_l not in _GEMINI_35_FLASH_PRIMARY_IDS:
        return None
    if not provider:
        return None

    fallback_model = "google/gemini-3-flash-preview" if "/" in model else "gemini-3-flash-preview"
    return {"provider": provider, "model": fallback_model}


def augment_fallback_chain_for_primary(
    chain: Any,
    *,
    provider: str,
    model: str,
) -> list[dict[str, Any]]:
    """Append a narrow protective fallback for Gemini 3.5 Flash primaries.

    ``gemini-3.5-flash`` remains the primary/stable model.  The older
    ``gemini-3-flash-preview`` is only added as a fallback route, never as a
    resolver alias or replacement mapping.  Explicit duplicate fallback entries
    win and are not repeated.
    """

    augmented = [dict(entry) for entry in (chain or []) if isinstance(entry, dict)]
    protective = _gemini_preview_fallback_for_primary(provider, model)
    if not protective:
        return augmented

    existing = {_entry_identity(entry) for entry in augmented}
    if _entry_identity(protective) not in existing:
        augmented.append(protective)
    return augmented


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain
