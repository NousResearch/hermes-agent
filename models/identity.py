"""Canonical model identity without catalogue, route, credential, or config ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from models.aliases import (
    AmbiguousModelAliasError,
    MODEL_ALIASES,
    ModelAliasPattern,
    model_alias_sort_key,
    resolve_declared_model_id,
    resolve_model_alias,
)
from providers import get_provider_profile, normalize_provider


__all__ = [
    "AmbiguousModelAliasError",
    "MODEL_ALIASES",
    "ModelAliasPattern",
    "ModelRef",
    "format_model_ref",
    "model_alias_sort_key",
    "normalize_model_id",
    "normalize_model_ref",
    "parse_configured_provider_ref",
    "parse_model_ref",
    "resolve_declared_model_id",
    "resolve_model_alias",
    "suggest_prefixed_model_id",
]


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Canonical provider/model identity used across Hermes surfaces."""

    provider: str
    model: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", normalize_provider(self.provider))
        object.__setattr__(self, "model", str(self.model or "").strip())


def _canonical_provider_ids(values: Iterable[str]) -> set[str]:
    return {
        canonical
        for value in values
        if (canonical := normalize_provider(str(value or "").strip()))
    }


def _named_custom_ids(values: Iterable[str]) -> tuple[str, ...]:
    ids = {
        normalized
        for value in values
        if (normalized := str(value or "").strip().lower()).startswith("custom:")
    }
    return tuple(sorted(ids, key=len, reverse=True))


def parse_model_ref(
    raw: str,
    default_provider: str = "",
    *,
    known_provider_ids: Iterable[str] = (),
    named_custom_provider_ids: Iterable[str] = (),
) -> ModelRef:
    """Parse Hermes provider:model syntax without guessing from punctuation."""

    value = str(raw or "").strip()
    if not value:
        return ModelRef(default_provider, "")

    lowered = value.lower()
    for provider_id in _named_custom_ids(named_custom_provider_ids):
        marker = f"{provider_id}:"
        if lowered.startswith(marker):
            model = value[len(marker):].strip()
            if model:
                return ModelRef(provider_id, model)

    colon = value.find(":")
    if colon > 0:
        provider_part = value[:colon].strip()
        model_part = value[colon + 1 :].strip()
        if provider_part and model_part:
            canonical = normalize_provider(provider_part)
            if canonical in _canonical_provider_ids(known_provider_ids):
                return ModelRef(canonical, model_part)

    return ModelRef(default_provider, value)


def parse_configured_provider_ref(
    raw: str,
    configured_provider_ids: Iterable[str],
) -> ModelRef | None:
    """Parse an explicit configured provider/model reference."""

    value = str(raw or "").strip()
    if "/" not in value:
        return None

    provider_part, model_part = (part.strip() for part in value.split("/", 1))
    if not provider_part or not model_part:
        return None

    configured = _canonical_provider_ids(configured_provider_ids)
    canonical = normalize_provider(provider_part)
    if canonical not in configured:
        return None
    return ModelRef(canonical, model_part)


def format_model_ref(ref: ModelRef) -> str:
    """Encode a model reference as the stable provider:model wire form."""

    if not ref.model:
        return ""
    return f"{ref.provider}:{ref.model}" if ref.provider else ref.model


def normalize_model_id(
    provider: str,
    model: str,
    *,
    known_ids: Iterable[str] = (),
) -> str:
    """Normalize via provider-owned rules and caller-supplied candidates."""

    canonical = normalize_provider(provider)
    value = str(model or "").strip()
    if not value:
        return value
    profile = get_provider_profile(canonical)
    if profile is None:
        return value
    return profile.normalize_model_id(value, known_ids=tuple(known_ids))


def suggest_prefixed_model_id(
    provider: str,
    model: str,
    *,
    known_ids: Iterable[str] = (),
) -> str | None:
    """Return an unambiguous provider-qualified repair from caller-owned candidates."""

    value = str(model or "").strip()
    if not value or "/" in value:
        return None
    normalized = normalize_model_id(provider, value, known_ids=known_ids)
    return normalized if normalized != value and "/" in normalized else None


def normalize_model_ref(
    ref: ModelRef,
    *,
    known_ids: Iterable[str] = (),
) -> ModelRef:
    """Normalize a model reference through the canonical identity seam."""

    return ModelRef(
        ref.provider,
        normalize_model_id(ref.provider, ref.model, known_ids=known_ids),
    )
