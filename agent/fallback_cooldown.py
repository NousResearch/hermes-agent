"""Primary rate-limit cooldown arming and per-session model rejection markers, shared by the
fallback walk (chat_completion_helpers) and restore_primary_runtime (agent_runtime_helpers)."""
import logging
import math
import time

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

_RATE_LIMIT_FAILOVER_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit})


def _provider_reset_delay(reset_at) -> float | None:
    """Seconds until the provider-declared reset, or None when missing/invalid/expired."""
    from agent.credential_pool import _parse_absolute_timestamp
    parsed = _parse_absolute_timestamp(reset_at)
    delay = parsed - time.time() if parsed is not None else None
    if delay is not None and math.isfinite(delay) and delay > 0:
        return delay
    return None


_BILLING_RECOVERY_PROBE_MIN_INTERVAL_S = 60.0


def _probe_primary_billing_recovery(agent) -> bool:
    """Clear billing-attributed primary state once the account is funded again (#126818).

    A credit exhaustion benches the primary credential for the full hour and arms a
    rate-limit cooldown on the session — but unlike a rate-limit window it can be cured
    at ANY moment by a portal top-up, so a fixed bench pins a long-lived session to its
    (often paid) fallback while the primary is usable again. When the primary is in
    billing-backed cooldown, probe the provider's account status (throttled to one probe
    a minute, everything fail-open) and on a funded answer clear the pool's
    billing-attributed benches. Only a session cooldown explicitly armed for billing
    loses its timer and backoff counter; independent 429/unknown cooldowns stay intact.

    Returns True when billing pool benches were cleared this call, not necessarily
    when the session can restore the primary.
    """
    primary = getattr(agent, "_primary_runtime", None) or {}
    primary_provider = str(primary.get("provider") or "").strip().lower()
    primary_model = str(primary.get("model") or "").strip()
    if primary_provider != "nous" or not primary_model:
        return False
    # Only a session actually routed away from the primary probes; a session on the
    # primary owns its own recovery through the normal per-turn restore.
    if (getattr(agent, "provider", "") or "").strip().lower() == primary_provider:
        return False
    now = time.monotonic()
    last_probe = getattr(agent, "_billing_recovery_probe_at", None)
    if last_probe is not None and now - last_probe < _BILLING_RECOVERY_PROBE_MIN_INTERVAL_S:
        return False
    # Cheap pre-check before any network I/O: only probe when the primary actually sits in
    # billing-backed cooldown — a pool bench attributed to billing, not merely an
    # armed session cooldown. Rate-limit windows and auth benches are not ours to clear.
    from agent.credential_pool import FAILURE_REASON_BILLING, load_pool

    def _pool_has_billing_bench(p) -> bool:
        try:
            return any(
                e.last_status == "exhausted" and e.failure_reason == FAILURE_REASON_BILLING
                for e in p.entries()
            )
        except Exception:
            return False

    pool = getattr(agent, "_credential_pool", None)
    pool_is_nous = pool is not None and (getattr(pool, "provider", "") or "").strip().lower() == "nous"
    has_billing_bench = pool_is_nous and _pool_has_billing_bench(pool)
    if not has_billing_bench:
        # No pool attached (or the fallback walk rebound it to the fallback's provider):
        # the primary's own pool, if any, lives on disk.
        try:
            disk_pool = load_pool("nous")
            has_billing_bench = bool(disk_pool) and _pool_has_billing_bench(disk_pool)
            if has_billing_bench:
                pool = disk_pool
        except Exception:
            has_billing_bench = False
    if not has_billing_bench:
        return False
    agent._billing_recovery_probe_at = now
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info
        info = get_nous_portal_account_info(force_fresh=True)
        access = getattr(info, "paid_service_access_info", None)
        total = getattr(access, "total_usable_credits", None)
        if isinstance(total, (int, float)) and math.isfinite(total):
            # The portal reports a balance: funded means strictly positive usable credits.
            usable = total > 0
        else:
            usable = bool(getattr(info, "is_paid", False))
        if not usable:
            return False
    except Exception:
        logger.debug("Billing-recovery probe failed; keeping billing cooldown", exc_info=True)
        return False
    cleared = False
    try:
        if pool is None or (getattr(pool, "provider", "") or "").strip().lower() != "nous":
            pool = load_pool("nous")
        if pool is not None and pool.clear_billing_benches() > 0:
            cleared = True
    except Exception:
        logger.debug("Billing-recovery pool unbench failed", exc_info=True)
    # Pool recovery is independent of session ownership: another entry may have
    # armed a live 429 after this billing bench. Unknown ownership stays intact.
    if cleared and getattr(agent, "_rate_limit_cooldown_reason", None) == FailoverReason.billing:
        if getattr(agent, "_rate_limited_until", 0) > now:
            agent._rate_limited_until = now
        agent._rate_limit_backoff_count = 0
        agent._rate_limit_cooldown_reason = None
    if cleared:
        logger.info(
            "Primary nous account is funded again (top-up detected); cleared billing benches; "
            "primary restoration remains subject to independent cooldowns (#126818)"
        )
    return cleared


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
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
    if getattr(agent, "_fallback_activated", False) and not (primary_provider and current_provider == primary_provider):
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
    agent._rate_limit_cooldown_reason = reason
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
