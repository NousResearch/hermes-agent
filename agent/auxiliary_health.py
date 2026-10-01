"""Endpoint identity for auxiliary custom-provider health checks."""
from typing import Any, Optional

from agent.configured_provider_resolution import get_configured_provider_entry
from providers import normalize_provider, normalize_route_base_url, resolves_to_custom_provider

def _canonical_health_provider(provider: str) -> str:
    raw = str(provider or "").strip().lower()
    if raw in {"custom", "local/custom"}:
        return "custom"
    if raw == "codex":
        raw = "openai-codex"
    return normalize_provider(raw)


def _unhealthy_cache_key(provider: str, base_url: Optional[str] = None) -> Any:
    """Canonical provider key, or endpoint-specific key for a custom route, scoped to profile home."""
    from hermes_constants import hermes_home_key

    label = _canonical_health_provider(provider)
    endpoint = normalize_route_base_url(_custom_health_base_url(provider, base_url))
    home_key = hermes_home_key()
    if endpoint:
        return home_key, "custom-endpoint", endpoint
    return home_key, label


def _custom_health_base_url(provider: str, explicit_base_url: Optional[str] = None) -> str:
    """Return the concrete custom endpoint used to scope health and failed-route checks."""
    from agent.auxiliary_client import _current_custom_base_url

    explicit = str(explicit_base_url or "").strip()
    raw = str(provider or "").strip().lower()
    label = _canonical_health_provider(provider)
    if raw in {"custom", "local/custom"}:
        return explicit or _current_custom_base_url()
    if label.startswith("custom:") and explicit:
        return explicit
    if resolves_to_custom_provider(raw):
        return explicit or _current_custom_base_url()
    entry = get_configured_provider_entry(provider)
    if entry:
        return explicit or str(entry.get("base_url") or "").strip()
    return ""




def fallback_candidate_unavailable_reason(exc: Exception) -> Optional[str]:
    """Why a fallback candidate cannot serve this walk (``_FALLBACK_REASONS`` label), or None.

    The same capacity classes that admitted the primary failure into the chain (payment/quota,
    rate limit, connection, route-incompatible model, malformed response) mean "this lane is out
    for now, try the next configured one"; anything else (a 400 request-shape error, a ValueError)
    is the caller's bug and must still propagate. Auth errors are excluded on purpose: they have
    their own refresh-then-quarantine path in the candidate helpers (#106367)."""
    from agent.auxiliary_client import _FALLBACK_REASONS
    return next(
        (label for predicate, label in _FALLBACK_REASONS if label != "auth error" and predicate(exc)),
        None,
    )


# Quarantine hold per unavailable-reason label. Payment/quota depletion and a dead credential
# last hours, so those keep the long default TTL (None); a per-minute 429, a dropped connection or
# a garbled body clears in seconds — holding the lane for 10 minutes process-wide would hide a
# healthy fallback from every aux task over one transient blip.
_TRANSIENT_CANDIDATE_QUARANTINE_SECONDS = 60.0
_CANDIDATE_QUARANTINE_TTL: dict[str, Optional[float]] = {
    "rate limit": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "connection error": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "invalid provider response": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
}


def fallback_candidate_quarantine_ttl(reason: Optional[str]) -> Optional[float]:
    """Seconds to hide a fallback candidate for ``reason`` (a ``_FALLBACK_REASONS`` label, or None
    for a stale credential); None means the long default TTL."""
    return _CANDIDATE_QUARANTINE_TTL.get(reason or "")
