"""Endpoint identity for auxiliary custom-provider health checks."""
import contextlib
from typing import Any, Optional

from hermes_cli.route_identity import normalize_route_base_url

def _unhealthy_cache_key(provider: str, base_url: Optional[str] = None) -> Any:
    """Provider-wide key, or endpoint-specific key for an explicit custom endpoint — prefixed with the
    active profile home: a 402 on profile A's account must not hide the provider from profile B's
    (differently funded) account in the same multiplexed process."""
    from agent.auxiliary_client import _normalize_chain_label
    from hermes_constants import hermes_home_key
    label = _normalize_chain_label(provider)
    endpoint = normalize_route_base_url(_custom_health_base_url(provider, base_url))
    home_key = hermes_home_key()
    if endpoint:
        return home_key, "custom-endpoint", endpoint
    return home_key, label


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
