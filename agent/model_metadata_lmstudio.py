"""LM Studio metadata cache helpers."""

from __future__ import annotations

from typing import Any, Optional


def _lmstudio_loaded_context(model: dict[str, Any], lookup_model: Optional[str] = None) -> Optional[int]:
    """Context of the requested loaded instance, or the first loaded instance if unspecified."""
    if not isinstance(model, dict):
        return None
    instances = model.get("loaded_instances", [])
    if not isinstance(instances, list):
        return None
    if lookup_model:
        matching = next(
            (instance for instance in instances
             if isinstance(instance, dict) and instance.get("id") == lookup_model),
            None,
        )
        # An exact instance is authoritative.  Do not silently borrow a
        # sibling's runtime context when its metadata is missing or invalid.
        if matching is not None:
            instances = [matching]
    for inst in instances:
        cfg = inst.get("config", {}) if isinstance(inst, dict) else None
        ctx = cfg.get("context_length") if isinstance(cfg, dict) else None
        if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
            return ctx
    return None


def _add_lmstudio_alias(
    cache: dict[str, dict[str, Any]],
    alias_ranks: dict[str, int],
    model_id: str,
    entry: dict[str, Any],
    *,
    short_alias_rank: int,
) -> None:
    """Cache an exact id and deterministically select its short-slug alias."""
    cache[model_id] = entry
    if "/" in model_id:
        short_id = model_id.split("/", 1)[1]
        if short_alias_rank > alias_ranks.get(short_id, 0):
            cache[short_id] = entry
            alias_ranks[short_id] = short_alias_rank
