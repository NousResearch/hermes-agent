"""Primary rate-limit cooldown arming and per-session model rejection markers, shared by the
fallback walk (chat_completion_helpers) and restore_primary_runtime (agent_runtime_helpers)."""
import contextlib
import logging
import math
import random
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


def _on_primary_route(agent) -> bool:
    """True when the agent's live route IS the ``_primary_runtime`` snapshot: provider, model and
    endpoint, the triple that keys a fallback entry (``_fallback_entry_key``). A provider name
    alone is not an identity: two ``custom`` endpoints (or one provider's other model) share it."""
    from hermes_cli.route_identity import normalize_route_base_url
    rt = getattr(agent, "_primary_runtime", None) or {}
    if not str(rt.get("provider") or "").strip():
        return False

    def ident(provider, model, base_url):
        return (str(provider or "").strip().lower(), str(model or "").strip(), normalize_route_base_url(base_url or ""))

    live = ident(getattr(agent, "provider", ""), getattr(agent, "model", ""), getattr(agent, "base_url", ""))
    return live == ident(rt.get("provider"), rt.get("model"), rt.get("base_url"))


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
    if reason not in _RATE_LIMIT_FAILOVER_REASONS:
        return None
    if getattr(agent, "_fallback_activated", False) and not _on_primary_route(agent):
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
    # Publish the window so other sessions/lanes don't probe the same closed model.
    try:
        from agent.model_health_ledger import record_model_window
        record_model_window(getattr(agent, "provider", ""), getattr(agent, "model", ""), backoff_seconds)
    except Exception:
        pass
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


DEFAULT_REPROBE_SECONDS = 300.0
# Sessions on the same interval would probe in lockstep — a self-inflicted herd against a
# fair-share limiter. Each deadline is spread over ±30% of the configured interval.
REPROBE_JITTER = 0.3
# A model throttled for hours must not be probed on the base interval forever. Each consecutive
# failed probe doubles the wait, capped at the ledger's own 4h ceiling.
MAX_REPROBE_SECONDS = 4 * 3600.0


def reprobe_backoff(interval: float, streak: int) -> float:
    """Next probe interval after ``streak`` consecutive failures (capped, never shortened)."""
    if interval <= 0:
        return 0.0
    factor = 2.0 ** max(0, min(int(streak), 7))
    return min(interval * factor, MAX_REPROBE_SECONDS)


def probe_primary_route(agent) -> tuple[bool, float | None]:
    """Spend one token to learn whether the primary route is back, before spending a turn's context.

    ``restore_primary_runtime`` only swaps clients — it cannot know the model is still 429ing. So
    every re-probe used to restore, then fail on the first real (possibly 100k-token) request. A
    tiny request answers the same question for a few tokens. Returns ``(available, retry_after_s)``;
    an inconclusive probe (no endpoint, non-chat mode, non-429 error) reports available so the
    restore path keeps its existing behaviour.
    """
    rt = getattr(agent, "_primary_runtime", None) or {}
    api_mode = str(rt.get("api_mode") or getattr(agent, "api_mode", "") or "chat_completions")
    model = str(rt.get("model") or "")
    base_url = str(rt.get("base_url") or "")
    if api_mode != "chat_completions" or not model or not base_url:
        return True, None
    try:
        import openai

        client = openai.OpenAI(base_url=base_url, api_key=str(rt.get("api_key") or "none"),
                               timeout=20.0, max_retries=0)
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=1,
        )
        return True, None
    except Exception as exc:  # any failure is evidence, not a crash
        retry_after = None
        headers = getattr(getattr(exc, "response", None), "headers", None)
        raw = headers.get("retry-after") if headers is not None else None
        try:
            retry_after = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            retry_after = None
        if getattr(exc, "status_code", None) == 429 or retry_after is not None:
            return False, retry_after
        logger.debug("Primary probe inconclusive (%s); leaving the decision to the restore path", exc)
        return True, None


def _reprobe_interval_seconds() -> float:
    """``fallback.reprobe_seconds`` (default 300; 0 disables the mid-turn re-probe)."""
    try:
        from hermes_cli.config import load_config
        raw = (load_config() or {}).get("fallback", {}).get("reprobe_seconds", DEFAULT_REPROBE_SECONDS)
        value = float(raw)
    except Exception:
        return DEFAULT_REPROBE_SECONDS
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return value


def reprobe_deadline(now: float | None = None) -> float:
    """Jittered monotonic deadline for the next mid-turn primary re-probe (``now`` when off)."""
    now = time.monotonic() if now is None else now
    interval = _reprobe_interval_seconds()
    if interval <= 0:
        return now
    return now + interval * random.uniform(1.0 - REPROBE_JITTER, 1.0 + REPROBE_JITTER)


def maybe_reprobe_primary(agent) -> bool:
    """Give the primary route another chance mid-turn once the fallback has held a while.

    Fallback is turn-scoped: ``restore_primary_runtime`` runs at the next turn start. A worker
    turn can run for hours, so a lane that fell back to a local model would otherwise stay there
    long after the primary's rate-limit window reopened. Once ``fallback.reprobe_seconds``
    elapses (and again every interval) this expires the armed cooldown and runs the same restore
    path; a primary that is still 429ing simply re-activates the fallback and re-arms the
    deadline, so the steady state costs one cheap failed call per interval.

    A passing probe restores only *tentatively*: it proves a 1-token request is accepted, not that
    a turn's worth of context will be. The restore is recorded with ``mark_probe_restore`` so a
    real 429 in the following minutes resumes the previous failure streak and the next probe waits
    longer, instead of the lane flapping back every base interval (observed: probe 11:51:47 OK →
    real call 11:51:50 refused).
    """
    if not getattr(agent, "_provider_fallback_active", False):
        return False
    if not getattr(agent, "_fallback_activated", False) or _on_primary_route(agent):
        return False
    interval = _reprobe_interval_seconds()
    if interval <= 0:
        return False
    now = time.monotonic()
    due_at = getattr(agent, "_fallback_reprobe_at", 0.0) or 0.0
    if now < due_at:
        return False
    # One shared window per model: if any session recorded a 429 for the primary, every
    # session waits it out instead of each firing its own probe at the same closed window.
    primary = getattr(agent, "_primary_runtime", None) or {}
    try:
        from agent.model_health_ledger import model_window_remaining
        shared_remaining = model_window_remaining(
            str(primary.get("provider") or ""), str(primary.get("model") or "")
        )
    except Exception:
        shared_remaining = 0.0
    if shared_remaining > 0:
        # Past the window plus a little spread; the next probe is the one that tests it.
        agent._fallback_reprobe_at = now + shared_remaining + random.uniform(0.0, min(15.0, shared_remaining * 0.1))
        return False
    provider = str(primary.get("provider") or "")
    model = str(primary.get("model") or "")
    try:
        from agent.model_health_ledger import (
            mark_probe_restore, model_failure_streak, record_model_window,
        )
        streak = model_failure_streak(provider, model)
    except Exception:
        mark_probe_restore = record_model_window = None  # type: ignore[assignment]
        streak = 0
    # One token now beats a whole turn's context a second later: a still-throttled primary
    # used to be discovered only after the full request was built and sent.
    available, retry_after = probe_primary_route(agent)
    if not available:
        wait = retry_after if retry_after and retry_after > 0 else interval
        backoff = reprobe_backoff(interval, streak + 1)
        agent._fallback_reprobe_at = now + backoff * random.uniform(1.0 - REPROBE_JITTER, 1.0 + REPROBE_JITTER)
        agent._rate_limited_until = now + wait
        if record_model_window is not None:
            with contextlib.suppress(Exception):
                record_model_window(provider, model, wait, probe_failure=True)
        logger.info(
            "Fallback re-probe: primary %s (%s) still rate-limited (retry_after=%s, streak=%d); "
            "next probe in ~%.0f s", model, provider, retry_after, streak + 1, backoff,
        )
        return False
    # Arm before the attempt: a restore that raises must not re-probe on every API iteration.
    agent._fallback_reprobe_at = now + reprobe_backoff(interval, streak) * random.uniform(1.0 - REPROBE_JITTER, 1.0 + REPROBE_JITTER)
    # The probe just proved the window is open; the armed cooldown would refuse the attempt
    # (restore_primary_runtime gates on ``_rate_limited_until``).
    agent._rate_limited_until = 0
    fallback_model, fallback_provider = agent.model, agent.provider
    try:
        from agent.agent_runtime_helpers import restore_primary_runtime
        restored = restore_primary_runtime(agent)
    except Exception as exc:
        logger.debug("Fallback re-probe failed: %s", exc, exc_info=True)
        return False
    if restored and mark_probe_restore is not None:
        with contextlib.suppress(Exception):
            mark_probe_restore(provider, model, streak)
    if restored:
        logger.info(
            "Fallback re-probe: primary %s (%s) is available again — leaving fallback %s (%s)",
            agent.model, agent.provider, fallback_model, fallback_provider,
        )
    return restored


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
