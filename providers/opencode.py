"""Pure identity and routing projections for OpenCode provider families."""

from __future__ import annotations

from typing import Optional

from providers.identity import normalize_provider
from providers.routing import normalize_provider_base_url

_OPENCODE_FAMILIES = ("opencode-go", "opencode-zen")


def opencode_provider_family(provider_id: Optional[str]) -> Optional[str]:
    """Return the OpenCode family for a built-in or family-prefixed provider."""

    raw = str(provider_id or "").strip().lower()
    if not raw:
        return None
    family_candidate = raw.removeprefix("custom:")
    canonical = normalize_provider(provider_id or "")
    if canonical in _OPENCODE_FAMILIES:
        return canonical
    return next(
        (family for family in _OPENCODE_FAMILIES if family_candidate.startswith(family)),
        None,
    )


def normalize_opencode_model_id(
    provider_id: Optional[str], model_id: Optional[str]
) -> str:
    """Strip a provider/family prefix from an OpenCode model identifier."""

    family = opencode_provider_family(provider_id)
    current = str(model_id or "").strip()
    if not current or family is None:
        return current
    for prefix in (f"{provider_id or family}/", f"{family}/"):
        if current.lower().startswith(prefix.lower()):
            return current[len(prefix) :]
    return current


def normalize_opencode_base_url(
    provider_id: Optional[str], api_mode: Optional[str], base_url: Optional[str]
) -> str:
    """Expose shared provider base-URL normalization for OpenCode adapters."""

    return normalize_provider_base_url(
        str(provider_id or ""), str(api_mode or ""), str(base_url or "")
    )


__all__ = [
    "normalize_opencode_base_url",
    "normalize_opencode_model_id",
    "opencode_provider_family",
]
