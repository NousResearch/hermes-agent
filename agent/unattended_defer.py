"""Defer-on-429 policy for UNATTENDED runs (cron fires, kanban workers).

Why this exists
---------------
``try_activate_fallback`` walks the chain the instant the primary returns a
rate-limit class error. For an INTERACTIVE session that is correct: a human is
waiting, and a degraded answer now beats no answer. For an UNATTENDED run
nobody is waiting — and on an all-one-provider chain the walk means
``primary -> cheaper rung -> cheaper rung -> paid floor`` on exactly the day the
subscription is capped. The work is not urgent; the quota is.

So: when the run is unattended, the failure is rate-limit class, the PRIMARY is
the thing that failed, and the provider told us *when* it resets, park the run
instead of walking. The caller (cron scheduler / kanban worker finalizer) turns
that into "skip this tick, retry after <reset>" or "re-queue the card with a
not-before timestamp".

Design notes
------------
* This module decides ONLY the policy and carries the reset timestamp. It never
  touches the chain, the client, or the board — the walk site asks it a
  question, and the terminal sites (cron ``run_job``, ``finalize_turn``) read the
  recorded deferral back out.
* Fails OPEN in every ambiguous case. No reset timestamp, unknown surface,
  unreadable config, non-rate-limit reason, already-on-a-fallback: return None
  and the existing walk happens byte-for-byte as before. A deferral that fires
  when it shouldn't silently stops work; a walk that fires when it shouldn't
  merely costs tokens. Prefer the loud failure.
* ``latency_critical`` is the escape hatch for unattended work that genuinely
  cannot wait (a monitor whose whole value is freshness). It forces ``walk``
  regardless of config.
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
from typing import Any, Optional

_logger = logging.getLogger(__name__)

# Memoized rate-limit FailoverReason set; see _rate_limit_reasons(). None = not yet
# resolved (the warning must fire at most once per process, not per 429).
_REASONS_CACHE: Optional[frozenset] = None

# Config value / env value vocabulary.
POLICY_DEFER = "defer"
POLICY_WALK = "walk"
_VALID_POLICIES = frozenset({POLICY_DEFER, POLICY_WALK})

# Per-run overrides. The cron scheduler and the kanban dispatcher set these for
# the child; a human shell never has them.
ENV_POLICY = "HERMES_UNATTENDED_ON_RATE_LIMIT"
ENV_LATENCY_CRITICAL = "HERMES_UNATTENDED_LATENCY_CRITICAL"
ENV_UNATTENDED = "HERMES_UNATTENDED_RUN"

# Cron fires run IN-PROCESS inside the gateway, several at a time. A per-job
# override therefore cannot live in ``os.environ`` — job B would read job A's
# value. ContextVars are per-task, which is the same reason ``_CronRunScope``
# uses them for session identity.
_CTX_LATENCY_CRITICAL: contextvars.ContextVar[Optional[bool]] = contextvars.ContextVar(
    "hermes_unattended_latency_critical", default=None,
)

# Attribute the walk site stamps on the agent and the terminal sites read back.
_DEFERRAL_ATTR = "_unattended_deferred_until"
_DEFERRAL_DETAIL_ATTR = "_unattended_deferral_detail"

# A reset further out than this is treated as unusable rather than parking a card
# for a week on one malformed header. 24h covers Anthropic 5-hour + weekly buckets
# and Ollama Cloud's 14-day bucket clamps to a daily retry instead of a fortnight.
MAX_DEFER_SECONDS = 24 * 60 * 60
# Below this a deferral is pointless churn — the ordinary retry/backoff path is
# cheaper than a board round-trip.
MIN_DEFER_SECONDS = 30


def _truthy(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def is_latency_critical() -> bool:
    """This unattended run must not be parked (freshness IS the deliverable).

    ContextVar wins over env so an in-process cron fire's per-job flag beats a
    process-wide default; ``None`` means "not set here" and falls through.
    """
    ctx = _CTX_LATENCY_CRITICAL.get()
    if ctx is not None:
        return bool(ctx)
    return _truthy(os.environ.get(ENV_LATENCY_CRITICAL))


def set_latency_critical(value: Optional[bool]):
    """Bind the per-run latency-critical flag; returns the token to reset with.

    Used by ``cron._CronRunScope`` (enter/exit) so a ``latency_critical: true`` job
    keeps walking the chain while its neighbours in the same process defer.
    """
    return _CTX_LATENCY_CRITICAL.set(value)


def reset_latency_critical(token) -> None:
    with_suppress = getattr(_CTX_LATENCY_CRITICAL, "reset", None)
    if token is not None and with_suppress is not None:
        try:
            _CTX_LATENCY_CRITICAL.reset(token)
        except (ValueError, RuntimeError):
            _CTX_LATENCY_CRITICAL.set(None)


def is_unattended_run() -> bool:
    """True for a dispatcher-spawned kanban worker, a cron fire, or an explicitly
    flagged unattended run. Interactive CLI/TUI/desktop/gateway chat is False.

    The cron surface is read through ``gateway.session_context`` because
    ``run_job`` binds it as a ContextVar, not an env var (parallel jobs in one
    process would clobber each other through ``os.environ``).
    """
    if os.environ.get("HERMES_KANBAN_TASK"):
        return True
    if _truthy(os.environ.get(ENV_UNATTENDED)):
        return True
    try:
        from gateway.session_context import get_session_env
        if str(get_session_env("HERMES_CRON_SESSION", "") or "").strip():
            return True
    except Exception:
        pass
    return False


def resolve_policy(config: Optional[dict] = None) -> str:
    """Effective policy: env override > ``fallback.unattended_on_rate_limit`` > walk.

    Default is ``walk`` so an un-migrated config behaves exactly as it does today.
    An unrecognized value is ignored (not an error): a typo must not silently
    park the fleet.
    """
    env_value = str(os.environ.get(ENV_POLICY, "") or "").strip().lower()
    if env_value in _VALID_POLICIES:
        return env_value
    try:
        if config is None:
            from hermes_cli.config import load_config
            config = load_config() or {}
        section = config.get("fallback")
        if isinstance(section, dict):
            value = str(section.get("unattended_on_rate_limit", "") or "").strip().lower()
            if value in _VALID_POLICIES:
                return value
    except Exception:
        pass
    return POLICY_WALK


def _primary_is_the_failure(agent: Any) -> bool:
    """Whether the backend that just 429'd is the PRIMARY.

    Mirrors ``_arm_rate_limit_cooldown``: once a fallback is active, a further
    rate-limit came from that rung, not the primary, so parking the whole run on
    the primary's reset would be wrong — keep walking.
    """
    if not getattr(agent, "_fallback_activated", False):
        return True
    current = str(getattr(agent, "provider", "") or "").strip().lower()
    primary = str((getattr(agent, "_primary_runtime", None) or {}).get("provider") or "").strip().lower()
    return bool(primary and current == primary)


def _pool_reset_at(agent: Any) -> Optional[float]:
    """Epoch seconds when the primary's credential pool re-enters rotation.

    This is the same signal the reset-aware restore gate in
    ``agent_runtime_helpers._primary_reset_gate_blocks`` consults, so a deferral
    and a restore agree on when the primary is usable again. The pool row is
    populated from the provider's own ``resets_at`` / ``retry-after`` /
    ``x-ratelimit-reset`` (see ``credential_pool`` normalization).
    """
    try:
        pool = getattr(agent, "_credential_pool", None)
        if pool is None:
            return None
        next_at = getattr(pool, "next_available_at", lambda: None)()
        return float(next_at) if next_at else None
    except Exception:
        return None


def _error_context_reset_at(error_context: Optional[dict]) -> Optional[float]:
    """Reset time carried by THIS error, when the pool has nothing.

    ``extract_api_error_context`` already normalizes ``resets_at`` / ``retry_after`` /
    ``Retry-After`` / ``x-ratelimit-reset`` / free-text "resets in 4h25m" into
    ``reset_at``; reuse its parser rather than re-deriving the shapes here.
    """
    if not isinstance(error_context, dict):
        return None
    try:
        from agent.credential_pool import _parse_absolute_timestamp
        return _parse_absolute_timestamp(error_context.get("reset_at"))
    except Exception:
        return None


def _rate_limit_reasons() -> frozenset:
    """The rate-limit-class FailoverReasons, wherever the constant currently lives.

    ``_RATE_LIMIT_FAILOVER_REASONS`` has moved between ``chat_completion_helpers``
    and ``fallback_cooldown`` across upstream refactors. Importing from ONE hardcoded
    module inside ``resolve_deferral``'s blanket ``except`` meant a relocation
    silently disabled this entire feature — every unattended run walked the chain
    again with no error, no log line, and a green test suite elsewhere. So: try both
    homes, and if neither resolves, say so ONCE at warning level rather than failing
    quiet. Empty set = no deferral ever = the pre-existing walk (still fails open,
    but audibly).
    """
    global _REASONS_CACHE
    if _REASONS_CACHE is not None:
        return _REASONS_CACHE
    for module_name in ("agent.fallback_cooldown", "agent.chat_completion_helpers"):
        try:
            module = __import__(module_name, fromlist=["_RATE_LIMIT_FAILOVER_REASONS"])
            reasons = getattr(module, "_RATE_LIMIT_FAILOVER_REASONS", None)
            if reasons:
                _REASONS_CACHE = frozenset(reasons)
                return _REASONS_CACHE
        except Exception:
            continue
    # Last resort: rebuild from the enum itself, so a module rename cannot mute us.
    try:
        from agent.error_classifier import FailoverReason
        _REASONS_CACHE = frozenset(
            r for name in ("rate_limit", "billing", "upstream_rate_limit")
            if (r := getattr(FailoverReason, name, None)) is not None
        )
        if _REASONS_CACHE:
            return _REASONS_CACHE
    except Exception:
        pass
    _logger.warning(
        "unattended_defer: could not resolve _RATE_LIMIT_FAILOVER_REASONS from any known "
        "module; defer-on-429 is INERT and unattended runs will walk the fallback chain."
    )
    _REASONS_CACHE = frozenset()
    return _REASONS_CACHE


def resolve_deferral(
    agent: Any,
    reason: Any,
    *,
    error_context: Optional[dict] = None,
    config: Optional[dict] = None,
) -> Optional[float]:
    """Epoch-seconds reset to park until, or None to walk the chain as usual.

    Every gate must pass, in cheapest-first order:

    1. policy is ``defer``  (default ``walk`` = today's behaviour, untouched)
    2. the run is unattended and not latency-critical
    3. the failure is rate-limit class (rate_limit / billing / upstream_rate_limit)
    4. the PRIMARY is what failed (not an already-active fallback rung)
    5. a usable reset timestamp exists, within ``MIN``..``MAX`` seconds from now

    Gate 5 is the load-bearing one: **no reset time means no deferral.** Parking a
    run with no idea when it can resume is worse than spending a cheap rung.
    """
    try:
        if resolve_policy(config) != POLICY_DEFER:
            return None
        if not is_unattended_run() or is_latency_critical():
            return None
        if reason not in _rate_limit_reasons():
            return None
        if not _primary_is_the_failure(agent):
            return None
        reset_at = _pool_reset_at(agent) or _error_context_reset_at(error_context)
        if not reset_at:
            return None
        delay = float(reset_at) - time.time()
        if delay < MIN_DEFER_SECONDS:
            return None
        if delay > MAX_DEFER_SECONDS:
            reset_at = time.time() + MAX_DEFER_SECONDS
        return float(reset_at)
    except Exception:
        # Fail open: any bug here degrades to the pre-existing walk.
        return None


def record_deferral(agent: Any, reset_at: float, *, provider: str = "", model: str = "") -> None:
    """Stamp the deferral on the agent for the terminal site to read back.

    Kept as plain attributes (not a return value) because the walk site is called
    from ~10 places across the turn loop, all of which only care about the
    boolean "did we switch"; threading a new return type through every one of
    them would be a far larger and riskier diff than a one-attribute handoff.
    """
    try:
        agent._unattended_deferred_until = float(reset_at)
        agent._unattended_deferral_detail = {
            "reset_at": float(reset_at),
            "provider": str(provider or getattr(agent, "provider", "") or ""),
            "model": str(model or getattr(agent, "model", "") or ""),
        }
    except Exception:
        pass


def get_deferral(agent: Any) -> Optional[float]:
    """The recorded reset timestamp, or None. Read-only; safe on any object."""
    value = getattr(agent, _DEFERRAL_ATTR, None)
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


def get_deferral_detail(agent: Any) -> dict:
    detail = getattr(agent, _DEFERRAL_DETAIL_ATTR, None)
    return dict(detail) if isinstance(detail, dict) else {}


def clear_deferral(agent: Any) -> None:
    for attr in (_DEFERRAL_ATTR, _DEFERRAL_DETAIL_ATTR):
        try:
            setattr(agent, attr, None)
        except Exception:
            pass


def format_reset(reset_at: float) -> str:
    """Local ISO-8601 to the second, for log lines and card reasons."""
    from datetime import datetime
    try:
        return datetime.fromtimestamp(float(reset_at)).isoformat(timespec="seconds")
    except Exception:
        return str(reset_at)


__all__ = [
    "POLICY_DEFER", "POLICY_WALK", "ENV_POLICY", "ENV_LATENCY_CRITICAL", "ENV_UNATTENDED",
    "MAX_DEFER_SECONDS", "MIN_DEFER_SECONDS",
    "is_unattended_run", "is_latency_critical", "set_latency_critical", "reset_latency_critical",
    "resolve_policy", "resolve_deferral",
    "record_deferral", "get_deferral", "get_deferral_detail", "clear_deferral", "format_reset",
]
