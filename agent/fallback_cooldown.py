"""Profile/backend quota benches, primary throttling and session entitlement markers."""
import logging
import math
import time
import threading

from agent.backend_identity import BackendIdentity, should_skip_candidate
from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

_RATE_LIMIT_FAILOVER_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit})

# Credential pools already bench individual accounts. These slots cover a backend
# after pool recovery is spent, including keyless routes and exhausted fallbacks.
_QUOTA_BENCHES: dict[tuple[str, BackendIdentity], tuple[float, int]] = {}
_QUOTA_LOCK = threading.RLock()


def _runtime_identity(agent) -> BackendIdentity:
    return BackendIdentity.build(agent.provider, agent.model, agent.base_url)


def quota_reset_at(provider: str, model: str, base_url: str = "") -> float | None:
    """Active quota reset for this profile's deployment, including configured aliases."""
    from hermes_constants import hermes_home_key
    home = hermes_home_key()
    candidate = BackendIdentity.build(provider, model, base_url)
    with _QUOTA_LOCK:
        resets = [until for (scope, backend), (until, _) in _QUOTA_BENCHES.items()
                  if scope == home and until > time.time() and should_skip_candidate(candidate, backend)]
    return max(resets) if resets else None


def record_quota_exhaustion(agent, classified, error_context) -> None:
    """Share only proven billing/allowance exhaustion, never a transient or upstream 429."""
    if not (
        (classified.reason == FailoverReason.billing and not classified.billing_unverified)
        or (classified.reason == FailoverReason.rate_limit and classified.error_context.get("quota_exhausted"))
    ):
        return
    from hermes_constants import hermes_home_key
    key = (hermes_home_key(), _runtime_identity(agent))
    reset_at = (error_context or {}).get("reset_at") or classified.error_context.get("reset_at")
    delay = _provider_reset_delay(reset_at)
    with _QUOTA_LOCK:
        until, count = _QUOTA_BENCHES.get(key, (0, 0))
        if until > time.time():
            return  # another attempt during the same window must not extend the bench
        delay = delay if delay is not None else min(60 * 2 ** min(count, 8), 14400)
        _QUOTA_BENCHES[key] = (time.time() + delay, count + 1)


def confirm_backend_success(agent) -> None:
    """Clear exhaustion/backoff and announce a primary recovery only after a usable response."""
    from hermes_constants import hermes_home_key
    home, current = hermes_home_key(), _runtime_identity(agent)
    with _QUOTA_LOCK:
        for key in list(_QUOTA_BENCHES):
            if key[0] == home and should_skip_candidate(current, key[1]):
                del _QUOTA_BENCHES[key]
    pending = getattr(agent, "_pending_primary_recovery_notice", None)
    if not agent._fallback_activated:
        agent._rate_limit_backoff_count = 0
        agent._rate_limited_until = 0
    if pending and not agent._fallback_activated:
        agent._pending_primary_recovery_notice = None
        try:
            agent._emit_diagnostic_status(pending)
        except Exception:
            logger.debug("Primary recovery notification failed", exc_info=True)


def resume_available_fallback(agent) -> bool:
    """Try the highest configured fallback after a quota reset, retaining a healthy active route."""
    from agent.chat_completion_helpers import _fallback_entry_key, _should_skip_fallback_candidate
    current = _runtime_identity(agent)
    unavailable = getattr(agent, "_unavailable_fallback_keys", None) or set()
    for index, entry in enumerate(agent._fallback_chain):
        provider, model = str(entry.get("provider") or "").lower(), str(entry.get("model") or "")
        candidate = BackendIdentity.build(provider, model, entry.get("base_url") or "")
        if should_skip_candidate(candidate, current) and quota_reset_at(provider, model, candidate.base_url) is None:
            return False  # already on the highest usable configured route
        if _should_skip_fallback_candidate(agent, entry, _fallback_entry_key(entry), provider, model, unavailable):
            continue
        agent._fallback_index = index
        return agent._try_activate_fallback()
    return False


def guard_quota_request(agent) -> None:
    """Keep an all-exhausted chain from probing a known-empty backend on every retry."""
    reset = quota_reset_at(agent.provider, agent.model, agent.base_url)
    if reset is None:
        return
    import httpx
    from openai import RateLimitError
    body = {"error": {"code": "terminal_quota_exhausted", "message": "Backend quota cooldown is still active", "reset_at": reset}}
    response = httpx.Response(429, request=httpx.Request("POST", agent.base_url), json=body)
    raise RateLimitError(body["error"]["message"], response=response, body=body)


def _provider_reset_delay(reset_at) -> float | None:
    """Seconds until the provider-declared reset, or None when missing/invalid/expired."""
    from agent.credential_pool import _parse_absolute_timestamp
    parsed = _parse_absolute_timestamp(reset_at)
    delay = parsed - time.time() if parsed is not None else None
    if delay is not None and math.isfinite(delay) and delay > 0:
        return delay
    return None


def switch_deferred_by_reset(agent, reason: "FailoverReason | None", reset_at) -> bool:
    """Opt-in ``fallback.min_switch_reset_seconds`` (default 0 = off, #117484): when the primary's
    rate limit reopens sooner than N seconds, switching model mid-task costs more than waiting, so
    the fallback walk is skipped and the retry loop's own backoff rides out the window. Only for
    rate-limit failovers leaving the primary with a valid future ``reset_at``."""
    if reason not in _RATE_LIMIT_FAILOVER_REASONS or getattr(agent, "_fallback_activated", False):
        return False
    try:
        from hermes_cli.config import load_config
        threshold = float((load_config() or {}).get("fallback", {}).get("min_switch_reset_seconds") or 0)
    except Exception:
        return False
    if threshold <= 0:
        return False
    delay = _provider_reset_delay(reset_at)
    if delay is None or delay >= threshold:
        return False
    logging.info("Rate limit resets in %.0f s (< fallback.min_switch_reset_seconds=%.0f): staying on the primary", delay, threshold)
    return True


def _arm_rate_limit_cooldown(
    agent, reason: "FailoverReason | None", reset_at=None,
) -> int | None:
    """Arm the primary cooldown until the provider reset, or use exponential backoff.

    ``reset_at`` is an absolute wall-clock timestamp while ``_rate_limited_until`` is monotonic;
    convert through a duration so wall-clock epoch values never enter the monotonic comparison.
    Missing, invalid, or expired provider resets retain the 60s → 2m → ... → 4h fallback.
    Only arm when leaving the primary: chain-switching from an active fallback means the primary
    was not the failing source. Return the armed cooldown in seconds, or None when not armed.
    """
    if reason not in _RATE_LIMIT_FAILOVER_REASONS:
        return None
    rt = agent._primary_runtime or {}
    primary = BackendIdentity.build(rt.get("provider"), rt.get("model"), rt.get("base_url"))
    if getattr(agent, "_fallback_activated", False) and not should_skip_candidate(_runtime_identity(agent), primary):
        return None
    backoff_count = getattr(agent, "_rate_limit_backoff_count", 0)
    agent._rate_limit_backoff_count = backoff_count + 1
    provider_delay = _provider_reset_delay(reset_at)
    if provider_delay is not None:
        backoff_seconds = math.ceil(provider_delay)
        source = "provider reset"
    else:
        backoff_seconds = min(60 * (2 ** backoff_count), 14400)
        source = "exponential fallback"
    agent._rate_limited_until = time.monotonic() + backoff_seconds
    logging.info(
        "Rate-limit backoff level %d: cooldown %d s (%.1f min, backoff#%d, %s)",
        backoff_count, backoff_seconds, backoff_seconds / 60, backoff_count + 1, source,
    )
    return backoff_seconds


def _mark_entitlement_rejected_model(agent, api_error) -> bool:
    """Record a Codex ChatGPT-account 400 that rejects the current model for this account.

    Pool rotation runs first (recover_with_credential_pool benches (credential, model) and
    moves to the next entitled entry, #71970); this runs only once no pool entry is left for
    the model, so the (provider, model) pair is treated as dead for the session: the fallback
    walk skips it and restore_primary_runtime stops switching back — otherwise every turn
    re-fails on the primary, announces an unverified "Primary model restored", and oscillates
    forever (#106475).
    """
    if getattr(api_error, "status_code", None) != 400:
        return False
    from agent.error_classifier import CODEX_ACCOUNT_MODEL_ENTITLEMENT_MARKER
    haystack = str(getattr(api_error, "message", "") or api_error).lower()
    if CODEX_ACCOUNT_MODEL_ENTITLEMENT_MARKER not in haystack:
        return False
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    model = str(getattr(agent, "model", "") or "").strip()
    if not provider or not model:
        return False
    pool = getattr(agent, "_credential_pool", None)
    if pool is not None and pool.has_available(model=model):
        return False  # another pool entry is still eligible for this model; rotation owns it
    rejected = getattr(agent, "_entitlement_rejected_models", None)
    if rejected is None:
        rejected = agent._entitlement_rejected_models = set()
    if (provider, model) in rejected:
        return True
    rejected.add((provider, model))
    logger.warning(
        "Model entitlement rejection: this account is not entitled to %s via %s; "
        "treating it as unavailable for this session",
        model, provider,
    )
    agent._buffer_diagnostic_status(
        f"🚫 This account is not entitled to {model} via {provider}; it will be skipped "
        "until restart. Switch to an entitled model via /model or `hermes model`."
    )
    return True


def _is_entitlement_rejected(agent, provider: str, model: str) -> bool:
    """True when (provider, model) — as configured or normalized — was rejected as unentitled
    for this account (see _mark_entitlement_rejected_model)."""
    rejected = getattr(agent, "_entitlement_rejected_models", None) or ()
    if not rejected:
        return False
    if (provider, model) in rejected:
        return True
    from hermes_cli.model_normalize import normalize_model_for_provider
    return (provider, normalize_model_for_provider(model, provider)) in rejected
