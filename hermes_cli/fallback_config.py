"""Helpers for reading the effective fallback chain from config."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel model name (nous provider only): resolves to the Portal's current
# ``freeRecommendedModels[0]`` at chain-build time. Lets free-tier users say
# "whatever is free right now" instead of hand-tracking Portal rotations.
RECOMMENDED_FREE_SENTINEL = "recommended:free"


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _resolve_recommended_free() -> str | None:
    """Resolve the sentinel via the Portal recommended-models cache.

    Import is local to avoid a module cycle and to keep this file cheap to
    import. Any failure returns None — the caller drops the entry with a
    warning and the rest of the chain still works.
    """
    try:
        from hermes_cli.models import get_nous_recommended_free_model
        return get_nous_recommended_free_model()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("recommended:free resolution failed: %s", exc)
        return None


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

        if model == RECOMMENDED_FREE_SENTINEL and provider == "nous":
            resolved = _resolve_recommended_free()
            if not resolved:
                logger.warning(
                    "fallback entry 'nous/%s' skipped: no free recommendation "
                    "available from the Portal (network down and no cached "
                    "answer). The rest of the chain is unaffected.",
                    RECOMMENDED_FREE_SENTINEL,
                )
                continue
            model = resolved

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
