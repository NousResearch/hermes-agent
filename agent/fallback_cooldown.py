"""Primary cooldown arming (rate limits and outages) and per-session model rejection markers, shared
by the fallback walk (chat_completion_helpers) and restore_primary_runtime (agent_runtime_helpers)."""
import logging
import math
import time

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

_RATE_LIMIT_FAILOVER_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit})
# Failovers that say the primary ROUTE is down (not that this request was bad): restoring it on the
# very next turn re-fails, burns that turn's retries and flaps the "restored" notice (#46856).
# ``unknown`` only reaches the fallback walk after the retry budget is spent on it.
_OUTAGE_FAILOVER_REASONS = frozenset({
    FailoverReason.overloaded, FailoverReason.server_error, FailoverReason.timeout, FailoverReason.auth,
    FailoverReason.auth_permanent, FailoverReason.upstream_blocked, FailoverReason.model_not_found,
    FailoverReason.unknown,
})
DEFAULT_OUTAGE_COOLDOWN_SECONDS = 300
_MAX_COOLDOWN_SECONDS = 14400


def _provider_reset_delay(reset_at) -> float | None:
    """Seconds until the provider-declared reset, or None when missing/invalid/expired."""
    from agent.credential_pool import _parse_absolute_timestamp
    parsed = _parse_absolute_timestamp(reset_at)
    delay = parsed - time.time() if parsed is not None else None
    if delay is not None and math.isfinite(delay) and delay > 0:
        return delay
    return None


def switch_deferred_by_reset(agent, reason: FailoverReason | None, reset_at) -> bool:
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


def _leaving_primary(agent) -> bool:
    """False when chain-switching from an active fallback: the primary was not the failing source.
    A fallback on the primary's provider with another model is still a fallback, so the route must
    match on provider AND model before an outage there may re-bench the primary."""
    if not getattr(agent, "_fallback_activated", False):
        return True
    rt = getattr(agent, "_primary_runtime", None) or {}

    def norm(value) -> str:
        return str(value or "").strip().lower()

    return bool(norm(rt.get("provider"))) and all(
        norm(getattr(agent, field, "")) == norm(rt.get(field)) for field in ("provider", "model"))


def outage_cooldown_seconds() -> float:
    """``fallback.outage_cooldown_seconds`` (0 = restore the primary every turn, the old behaviour),
    bounded to [0, 4 h]; an unreadable or invalid value keeps the default."""
    from hermes_cli.config import load_config
    try:
        fallback_cfg = (load_config() or {}).get("fallback")
    except Exception:  # health: allow BLE001 -- a config read failure must not abort the fallback swap
        fallback_cfg = None
    raw = fallback_cfg.get("outage_cooldown_seconds") if isinstance(fallback_cfg, dict) else None
    try:
        seconds = float(DEFAULT_OUTAGE_COOLDOWN_SECONDS if raw is None else raw)
    except (TypeError, ValueError):
        seconds = DEFAULT_OUTAGE_COOLDOWN_SECONDS
    return min(max(seconds, 0.0), _MAX_COOLDOWN_SECONDS) if math.isfinite(seconds) else DEFAULT_OUTAGE_COOLDOWN_SECONDS


def _arm_outage_cooldown(agent, reason: FailoverReason | None, reset_at=None) -> int | None:
    """Bench a primary that failed over for an outage for ``fallback.outage_cooldown_seconds`` (or a
    longer provider-declared reset), so later turns stay on the working fallback instead of retrying
    the dead route. ``restore_primary_runtime`` sends one probe turn at expiry (``_primary_probe_pending``)."""
    if reason not in _OUTAGE_FAILOVER_REASONS or not _leaving_primary(agent):
        return None
    configured = outage_cooldown_seconds()
    if configured <= 0:
        return None
    seconds = math.ceil(min(max(configured, _provider_reset_delay(reset_at) or 0), _MAX_COOLDOWN_SECONDS))
    agent._rate_limited_until = time.monotonic() + seconds
    agent._primary_outage_cooldown = True
    logging.info("Primary outage (%s): staying on the fallback for %d s before one probe turn", reason.value, seconds)
    return seconds


def arm_primary_cooldown(agent, reason: FailoverReason | None, reset_at=None) -> int | None:
    """Arm the primary's cooldown for a failover that leaves it; seconds armed, or None."""
    agent._primary_probe_pending = False
    if reason in _RATE_LIMIT_FAILOVER_REASONS:
        # A backup's own throttle says nothing about the primary: its outage probe stays owed.
        if _leaving_primary(agent):
            agent._primary_outage_cooldown = False
        return _arm_rate_limit_cooldown(agent, reason, reset_at=reset_at)
    return _arm_outage_cooldown(agent, reason, reset_at=reset_at)


def probe_failed_fast(agent, reason: FailoverReason | None) -> bool:
    """True on the primary's probe turn after an outage cooldown when it fails for an outage again:
    fail over now instead of spending the retry budget on a route that just proved still down."""
    return bool(getattr(agent, "_primary_probe_pending", False)) and reason in _OUTAGE_FAILOVER_REASONS


def begin_primary_probe(agent) -> bool:
    """Called as the primary is reinstated: True when this turn is the probe after an outage cooldown."""
    probing = bool(getattr(agent, "_primary_outage_cooldown", False))
    agent._primary_outage_cooldown = False
    agent._primary_probe_pending = probing
    return probing


def primary_return_notice(agent, previous_model: str, previous_provider: str, probing: bool) -> str:
    """The status line for a primary reinstated after a real fallback. A probe after an outage
    cooldown is not a recovery yet, so it is not announced as one."""
    if probing:
        return (f"🔁 Trying primary model {agent.model} via {agent.provider} again after its outage cooldown; "
                f"fallback {previous_model} via {previous_provider} takes over if it is still down.")
    return (f"✅ Primary model restored: {agent.model} via {agent.provider}; "
            f"fallback {previous_model} via {previous_provider} is no longer active.")


def _arm_rate_limit_cooldown(
    agent, reason: FailoverReason | None, reset_at=None,
) -> int | None:
    """Arm the primary cooldown until the provider reset, or use exponential backoff.

    ``reset_at`` is an absolute wall-clock timestamp while ``_rate_limited_until`` is monotonic;
    convert through a duration so wall-clock epoch values never enter the monotonic comparison.
    Missing, invalid, or expired provider resets retain the 60s → 2m → ... → 4h fallback.
    Only arm when leaving the primary: chain-switching from an active fallback means the primary
    was not the failing source. Return the armed cooldown in seconds, or None when not armed.
    """
    if reason not in _RATE_LIMIT_FAILOVER_REASONS or not _leaving_primary(agent):
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
