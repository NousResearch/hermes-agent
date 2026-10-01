"""Reusable default-model selection policy over a caller-supplied model universe."""

from __future__ import annotations

from collections.abc import Iterable

from models.identity import ModelRef
from models.metadata.pricing import _is_model_free
from models.catalog_policy import restrict_to_nous_policy
from models.selection_types import (
    ModelSelection,
    SelectionCandidate,
    SelectionPolicy,
    SelectionRequest,
)
from providers.identity import normalize_provider


def _ordered_ids(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        model = str(value or "").strip()
        key = model.lower()
        if not model or key in seen:
            continue
        seen.add(key)
        ordered.append(model)
    return tuple(ordered)


def select_default_model(
    provider: str,
    model_ids: Iterable[str],
    *,
    preferred_model: str = "",
    allow_preferred_without_models: bool = False,
    deprioritized_model_ids: Iterable[str] = (),
    purpose: str = "default",
) -> ModelSelection:
    """Choose a deterministic default from an already-effective model universe.

    The caller owns discovery, entitlement, policy and pricing facts. This seam
    owns only the choice: preferred model when eligible, otherwise stable input
    order, with optional models pushed behind ordinary candidates.
    """

    from models.selection import select_model

    provider_id = normalize_provider(provider)
    ids = list(_ordered_ids(model_ids))
    catalogued = {model.lower() for model in ids}
    preferred = str(preferred_model or "").strip()
    if not ids and preferred and allow_preferred_without_models:
        ids.append(preferred)

    deprioritized = {str(value or "").strip().lower() for value in deprioritized_model_ids}
    ordinary = [mid for mid in ids if mid.lower() not in deprioritized]
    if ordinary:
        ordered = ordinary + [mid for mid in ids if mid.lower() in deprioritized]
    else:
        ordered = ids

    preferred_ref = None
    if preferred:
        preferred_ref = next(
            (ModelRef(provider_id, mid) for mid in ordered if mid.lower() == preferred.lower()),
            None,
        )
        if preferred_ref is not None and ordinary and preferred.lower() in deprioritized:
            preferred_ref = None

    candidates = tuple(
        SelectionCandidate(
            ref=ModelRef(provider_id, model),
            catalogued=model.lower() in catalogued,
            source="default",
            stable_order=index,
        )
        for index, model in enumerate(ordered)
    )
    preferred_refs = (preferred_ref,) if preferred_ref is not None else ()
    return select_model(
        SelectionRequest(
            candidates=candidates,
            purpose=purpose,
            policy=SelectionPolicy(name=purpose, preferred=preferred_refs),
        )
    )


def select_nous_default_model(
    curated_model_ids: Iterable[str],
    *,
    portal_recommended_model_ids: Iterable[str] = (),
    pricing: dict[str, dict[str, object]] | None = None,
    policy_allowed_ids: set[str] | frozenset[str] | None = None,
    free_tier: bool = False,
    preferred_model: str = "",
) -> ModelSelection:
    """Choose the Nous silent default from already-fetched account/catalog facts."""

    curated = _ordered_ids(curated_model_ids)
    portal = _ordered_ids(portal_recommended_model_ids)
    seen = {mid.lower() for mid in curated}
    universe = curated + tuple(mid for mid in portal if mid.lower() not in seen)
    allowed = frozenset(policy_allowed_ids) if policy_allowed_ids else None
    universe = tuple(restrict_to_nous_policy(list(universe), allowed, rescue_empty=True))

    price_map = pricing or {}
    subscription = tuple(
        mid for mid in universe
        if isinstance(price_map.get(mid), dict)
        and price_map[mid].get("billing_mode") == "subscription"
    )
    if free_tier and (price_map or portal):
        free_portal = {mid.lower() for mid in portal}
        universe = tuple(
            mid for mid in universe
            if _is_model_free(mid, price_map) or mid.lower() in free_portal
        )
        subscription = tuple(mid for mid in subscription if mid in universe)

    return select_default_model(
        "nous",
        universe,
        preferred_model=preferred_model,
        deprioritized_model_ids=subscription,
        purpose="nous_default",
    )


__all__ = ["select_default_model", "select_nous_default_model"]
