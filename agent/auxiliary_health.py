"""Endpoint identity for auxiliary custom-provider health checks."""
import contextlib
import logging
import time
from typing import Any, Optional

from hermes_cli.route_identity import normalize_route_base_url

logger = logging.getLogger(__name__)


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


# "Recently 402'd" unhealthy-provider cache: a depleted provider stays so for hours, so hiding it
# for a TTL saves an RTT per aux call. In-process only (profiles may use different keys).
_AUX_UNHEALTHY_TTL_SECONDS = 600  # 10 minutes
_aux_unhealthy_until: dict[Any, float] = {}
_aux_unhealthy_logged_at: dict[Any, float] = {}
_aux_unhealthy_reason: dict[Any, str] = {}

_AUX_UNHEALTHY_PAYMENT_REASON = "payment / credit error"


def _mark_provider_unhealthy(
    provider: str, ttl: Optional[float] = None, *, base_url: Optional[str] = None,
    reason: str = _AUX_UNHEALTHY_PAYMENT_REASON, level: int = logging.WARNING,
) -> None:
    """Hide one provider endpoint until the TTL expires. ``reason`` is what the log says and what the
    skip line echoes: absent credentials are an expected state (DEBUG), a confirmed 402 is a fault
    (WARNING) — the old fixed payment wording sent local-only users chasing billing (#64144)."""
    from agent.auxiliary_client import _normalize_chain_label
    label = _normalize_chain_label(provider)
    if not label:
        return
    key = _unhealthy_cache_key(label, base_url)
    ttl = _AUX_UNHEALTHY_TTL_SECONDS if ttl is None else ttl
    expires_at = time.time() + ttl
    _aux_unhealthy_until[key] = expires_at
    _aux_unhealthy_reason[key] = reason
    logger.log(
        level,
        "Auxiliary: marking %s unhealthy for %ds (%s). "
        "Subsequent auxiliary calls will skip it until %s.",
        label, int(ttl), reason, time.strftime("%H:%M:%S", time.localtime(expires_at)),
    )


def _is_provider_unhealthy(label: str, base_url: Optional[str] = None) -> bool:
    """True iff this provider endpoint is unhealthy and unexpired; lazily evicts expired entries."""
    if not label:
        return False
    key = _unhealthy_cache_key(label, base_url)
    expires_at = _aux_unhealthy_until.get(key)
    if expires_at is None:
        return False
    if time.time() >= expires_at:
        _aux_unhealthy_until.pop(key, None)
        _aux_unhealthy_logged_at.pop(key, None)
        _aux_unhealthy_reason.pop(key, None)
        return False
    return True


def _log_skip_unhealthy(
    label: str, task: Optional[str] = None, *, base_url: Optional[str] = None,
) -> None:
    """Log a skipped unhealthy provider at most once per minute per label."""
    now = time.time()
    key = _unhealthy_cache_key(label, base_url)
    if now - _aux_unhealthy_logged_at.get(key, 0.0) >= 60:
        _aux_unhealthy_logged_at[key] = now
        expires_at = _aux_unhealthy_until.get(key, now)
        logger.info(
            "Auxiliary %s: skipping %s (%s, retry in %ds)",
            task or "call", label, _aux_unhealthy_reason.get(key, _AUX_UNHEALTHY_PAYMENT_REASON),
            max(0, int(expires_at - now)),
        )


def _reset_aux_unhealthy_cache() -> None:
    """Clear the unhealthy cache (tests / explicit user reset)."""
    _aux_unhealthy_until.clear()
    _aux_unhealthy_logged_at.clear()
    _aux_unhealthy_reason.clear()
    _aux_timeout_streak.clear()


# Consecutive-timeout quarantine. A full-budget timeout is ambiguous — a slow reasoning model
# and a hung endpoint look identical to a single request — so one timeout never hides a
# provider (the slow-model contract, #89445). But consecutive full-budget timeouts with no
# successful response in between are a hung endpoint, and every further call would otherwise
# pay the whole timeout again: a wedged local llama-server turned each image of a vision run
# into timeout + fallback. Two strikes trip the quarantine; the streak survives the trip, so
# while quarantined the endpoint is not dialed at all, one probe per cooldown window
# re-trips on a single timeout, and any success on the endpoint clears the streak.
_AUX_TIMEOUT_STRIKES = 2
# Shorter than the payment hold: a timeout is the more ambiguous signal, so half-open
# recovery comes sooner.
_AUX_TIMEOUT_QUARANTINE_TTL_SECONDS = 300
_aux_timeout_streak: dict[Any, int] = {}


class _AuxTimeoutQuarantine(TimeoutError):
    """Raised instead of dialing an endpoint quarantined for consecutive timeouts. The name
    and the "timed out" text keep ``_is_timeout_error`` classifying it as a timeout (so the
    ladder falls through to the fallback chain), while the ``isinstance`` check in the
    provider-fallback rung keeps the skip itself from counting as a fresh strike."""


def _note_aux_timeout(provider: str, *, base_url: Optional[str] = None) -> None:
    """Count one full-budget timeout on the endpoint; quarantine it after consecutive strikes.

    The streak is deliberately not reset when the quarantine trips: a re-trip must take a
    single strike (one probe per cooldown window), not another consecutive pair."""
    key = _unhealthy_cache_key(provider, base_url)
    streak = _aux_timeout_streak.get(key, 0) + 1
    _aux_timeout_streak[key] = streak
    if streak < _AUX_TIMEOUT_STRIKES:
        return
    _mark_provider_unhealthy(
        provider, ttl=_AUX_TIMEOUT_QUARANTINE_TTL_SECONDS, base_url=base_url,
        reason=f"{_AUX_TIMEOUT_STRIKES} consecutive request timeouts")


def _clear_aux_timeout_streak(provider: str, *, base_url: Optional[str] = None) -> None:
    """A success on the endpoint proves it alive: forget its timeout streak."""
    _aux_timeout_streak.pop(_unhealthy_cache_key(provider, base_url), None)


def _assert_route_dialable(req: Any) -> None:
    """Raise :class:`_AuxTimeoutQuarantine` instead of dialing an endpoint that is currently
    quarantined for consecutive timeouts — the fallback chain answers without paying the
    doomed full-budget RTT (the timeout mirror of the payment quarantine's skip contract)."""
    if _is_provider_unhealthy(req.resolved_provider, req.base_info):
        raise _AuxTimeoutQuarantine(
            f"endpoint {req.base_info or req.resolved_provider} skipped: quarantined after "
            f"{_AUX_TIMEOUT_STRIKES} consecutive requests timed out; dialing withheld")
