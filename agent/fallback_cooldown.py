"""Primary rate-limit cooldown arming and per-session model rejection markers, shared by the
fallback walk (chat_completion_helpers) and restore_primary_runtime (agent_runtime_helpers)."""
import logging
import time

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

_RATE_LIMIT_FAILOVER_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit})


def _arm_rate_limit_cooldown(agent, reason: "FailoverReason | None") -> int | None:
    """Arm the primary's exponential cooldown (60s → 2m → ... → 4h cap) on CONSECUTIVE rate-limits;
    restore_primary_runtime resets the counter. Only when leaving the primary: chain-switching from
    an active fallback means the primary was not the 429 source, so its cooldown is left alone.
    Return the armed cooldown in seconds, or None when no cooldown was armed."""
    if reason not in _RATE_LIMIT_FAILOVER_REASONS:
        return None
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
    if getattr(agent, "_fallback_activated", False) and not (primary_provider and current_provider == primary_provider):
        return None
    backoff_count = getattr(agent, "_rate_limit_backoff_count", 0)
    agent._rate_limit_backoff_count = backoff_count + 1
    backoff_seconds = min(60 * (2 ** backoff_count), 14400)
    agent._rate_limited_until = time.monotonic() + backoff_seconds
    logging.info("Rate-limit backoff level %d: cooldown %d s (%.1f min, backoff#%d)", backoff_count, backoff_seconds, backoff_seconds / 60, backoff_count + 1)
    return backoff_seconds


# Codex ChatGPT-account entitlement 400 — the account can never use the named slug, so with
# nothing to rotate it is a config error, not a transient failure (#106475).
_CODEX_ACCOUNT_MODEL_ENTITLEMENT_MARKER = "model is not supported when using codex with a chatgpt account"


def _mark_entitlement_rejected_model(agent, api_error) -> bool:
    """Record a Codex ChatGPT-account 400 that rejects the current model for this account.

    With a single credential there is no pool to rotate (#71970 covers that case), so the
    (provider, model) pair is treated as dead for the session: the fallback walk skips it and
    restore_primary_runtime stops switching back — otherwise every turn re-fails on the primary,
    announces an unverified "Primary model restored", and oscillates forever (#106475).
    """
    if getattr(api_error, "status_code", None) != 400:
        return False
    pool = getattr(agent, "_credential_pool", None)
    if pool is not None and len(pool.entries()) > 1:
        return False  # another account in the pool may be entitled; leave rotation to it
    haystack = str(getattr(api_error, "message", "") or api_error).lower()
    if _CODEX_ACCOUNT_MODEL_ENTITLEMENT_MARKER not in haystack:
        return False
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    model = str(getattr(agent, "model", "") or "").strip()
    if not provider or not model:
        return False
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
    agent._buffer_status(
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
