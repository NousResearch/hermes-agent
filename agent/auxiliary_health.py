"""Endpoint identity for auxiliary custom-provider health checks."""
import contextlib
from typing import Any, Optional

from hermes_cli.route_identity import normalize_route_base_url

def _unhealthy_cache_key(provider: str, base_url: Optional[str] = None, model: Optional[str] = None) -> Any:
    """Provider-wide key, or endpoint-specific key for an explicit custom endpoint — prefixed with the
    active profile home: a 402 on profile A's account must not hide the provider from profile B's
    (differently funded) account in the same multiplexed process."""
    from agent.auxiliary_client import _normalize_chain_label, _normalize_resolved_model
    from hermes_constants import hermes_home_key
    label = _normalize_chain_label(provider)
    endpoint = normalize_route_base_url(_custom_health_base_url(provider, base_url))
    key = (hermes_home_key(), "custom-endpoint", endpoint) if endpoint else (hermes_home_key(), label)
    model = _normalize_resolved_model(model, provider)  # configured ``gpt-x`` and wire ``openai/gpt-x`` share a key
    return (*key, model.strip().lower()) if model and model.strip() else key


def _custom_health_base_url(provider: str, explicit_base_url: Optional[str] = None) -> str:
    """Return the concrete custom endpoint used to scope health and failed-route checks."""
    from agent.auxiliary_client import _current_custom_base_url
    explicit = str(explicit_base_url or "").strip()
    from agent.auxiliary_client import _normalize_chain_label
    label = _normalize_chain_label(provider)
    if label == "local/custom":
        return explicit or _current_custom_base_url()
    if label.startswith("custom:") and explicit:
        return explicit
    with contextlib.suppress(ImportError):
        from hermes_cli.runtime_provider import _get_named_custom_provider, _resolves_to_custom
        if _resolves_to_custom(label):
            return explicit or _current_custom_base_url()
        entry = _get_named_custom_provider(provider)
        if entry:
            return explicit or str(entry.get("base_url") or "").strip()
    return ""




# Quarantine hold per unavailable-reason label. Payment/quota depletion and a dead credential
# last hours, so those keep the long default TTL (None); a per-minute 429, a dropped connection or
# a garbled body clears in seconds — holding the lane for 10 minutes process-wide would hide a
# healthy fallback from every aux task over one transient blip.
_TRANSIENT_CANDIDATE_QUARANTINE_SECONDS = 60.0
_CANDIDATE_QUARANTINE_TTL: dict[str, Optional[float]] = {
    "rate limit": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "connection error": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "invalid provider response": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "provider overloaded": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "server error": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
}


def fallback_candidate_quarantine_ttl(reason: Optional[str]) -> Optional[float]:
    """Seconds to hide a fallback candidate for ``reason`` (a ``fallback_label`` label, or None
    for a stale credential); None means the long default TTL."""
    return _CANDIDATE_QUARANTINE_TTL.get(reason or "")
