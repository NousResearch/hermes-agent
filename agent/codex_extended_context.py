"""Codex catalog-backed context choices, separate from verified legacy -900k aliases."""

from __future__ import annotations

CODEX_EXTENDED_CONTEXT_SUFFIX = "-extended-context"


def extended_context_base(model: str | None) -> str | None:
    """Base identity of an explicit Hermes-only context choice, not an entitlement check."""
    bare = (model or "").strip()
    if bare.endswith(CODEX_EXTENDED_CONTEXT_SUFFIX):
        base = bare.removesuffix(CODEX_EXTENDED_CONTEXT_SUFFIX)
        if base and not base.endswith((CODEX_EXTENDED_CONTEXT_SUFFIX, "-900k")):
            return base
    return None


def catalog_extended_budgets(entries: list) -> dict[str, int]:
    """Only exact, visible entries with consistent and explicitly published bounds qualify."""
    pairs: dict[str, set[tuple[int, int] | None]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("slug"), str):
            continue
        slug = entry["slug"].strip()
        if not slug or slug.endswith((CODEX_EXTENDED_CONTEXT_SUFFIX, "-900k")):
            continue
        ctx, cap = entry.get("context_window"), entry.get("max_context_window")
        visible = str(entry.get("visibility", "")).strip().lower() not in {"hide", "hidden"}
        pair = (ctx, cap) if visible and type(ctx) is int and type(cap) is int else None
        pairs.setdefault(slug, set()).add(pair)
    budgets = {}
    for slug, observed in pairs.items():
        if len(observed) != 1:
            continue
        bound = next(iter(observed))
        if bound is not None and 0 < bound[0] < min(900_000, bound[1]):
            budgets[slug] = min(900_000, bound[1])
    return budgets
