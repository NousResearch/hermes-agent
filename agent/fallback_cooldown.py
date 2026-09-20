"""Primary rate-limit cooldown arming and per-session model rejection markers, shared by the
fallback walk (chat_completion_helpers) and restore_primary_runtime (agent_runtime_helpers)."""
import logging
import math
import time

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

_RATE_LIMIT_FAILOVER_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit})


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
    from agent.credential_pool import _parse_absolute_timestamp
    parsed_reset_at = _parse_absolute_timestamp(reset_at)
    provider_delay = parsed_reset_at - time.time() if parsed_reset_at is not None else None
    if provider_delay is not None and math.isfinite(provider_delay) and provider_delay > 0:
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
