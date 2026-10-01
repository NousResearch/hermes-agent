"""Pure Nous Portal recommendation semantics."""

from __future__ import annotations

from typing import Any


def recommended_aux_model(
    payload: dict[str, Any] | None,
    *,
    vision: bool = False,
    free_tier: bool = False,
) -> str | None:
    """Select the provider recommendation from an already-fetched Portal payload."""
    if not isinstance(payload, dict) or not payload:
        return None
    kind = "Vision" if vision else "Compaction"
    tiers = ("free",) if free_tier else ("paid", "free")
    for tier in tiers:
        entry = payload.get(f"{tier}Recommended{kind}Model")
        model_name = entry.get("modelName") if isinstance(entry, dict) else None
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    return None


__all__ = ["recommended_aux_model"]
