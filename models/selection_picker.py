"""Pure picker/setup candidate construction from caller-supplied model ids."""

from __future__ import annotations

from collections.abc import Iterable

from models.identity import ModelRef, normalize_model_id
from providers.identity import normalize_provider


def list_picker_candidates(
    provider: str,
    model_ids: Iterable[str],
    *,
    current_model: str = "",
    allow_whitespace: bool = False,
) -> tuple[ModelRef, ...]:
    """Canonicalize, filter, deduplicate and order picker candidates.

    The caller owns catalog acquisition and endpoint-specific facts. This seam
    owns candidate interpretation only; it performs no network, credential,
    pricing, availability, confirmation, or presentation work.
    """

    provider_id = normalize_provider(provider)
    raw_ids = tuple(str(value or "").strip() for value in model_ids)
    known_ids = tuple(value for value in raw_ids if value)

    candidates: list[ModelRef] = []
    seen: set[ModelRef] = set()
    for raw in raw_ids:
        if not raw:
            continue
        if not allow_whitespace and any(ch.isspace() for ch in raw):
            continue
        canonical = normalize_model_id(provider_id, raw, known_ids=known_ids)
        ref = ModelRef(provider_id, canonical)
        if not ref.model or ref in seen:
            continue
        seen.add(ref)
        candidates.append(ref)

    if current_model:
        current = ModelRef(
            provider_id,
            normalize_model_id(provider_id, current_model, known_ids=known_ids),
        )
        if current in seen:
            candidates = [current, *(ref for ref in candidates if ref != current)]

    return tuple(candidates)


def picker_model_ids(
    provider: str,
    model_ids: Iterable[str],
    *,
    current_model: str = "",
    allow_whitespace: bool = False,
) -> tuple[str, ...]:
    return tuple(
        ref.model
        for ref in list_picker_candidates(
            provider,
            model_ids,
            current_model=current_model,
            allow_whitespace=allow_whitespace,
        )
    )


__all__ = ["list_picker_candidates", "picker_model_ids"]
