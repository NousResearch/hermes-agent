"""Automatic bounded re-runs for scheduled fires that never reached the model.

Inspired by Claude Cowork (desktop changelog v1.46388.1, 2026-09-04): "automatic re-runs
(after 5, 15, and 30 minutes) for a scheduled task that could not reach the model at all,
for example right after the computer wakes behind a VPN."

The class is deliberately narrow: the run must have FAILED with a transient network /
DNS error (``cron.scheduler_preflight._is_transient_provider_resolve_error``) AND the
agent must have completed zero API calls. Nothing was executed and nothing was spent, so
re-running cannot double a side effect — unlike a generic failure retry (see PR #16512),
which has to answer for one-shot dispatch accounting and mid-run side effects. Recurring
jobs only: finite one-shots are pre-claimed by ``claim_dispatch`` (at-most-times, #38758)
and must not regain a consumed dispatch here.

A second, narrower class rides the same ledger: a fire whose gateway refused it
(``GatewayUnavailableError`` — the owner was draining for an update/restart, still
starting, or gone) before the agent ran. Nothing was executed there either, and the
owner is normally back within a minute, so that class gets exactly ONE re-run
``GATEWAY_RETRY_DELAYS_SECONDS`` (2 minutes) out instead of the network ladder: an
unattended ``hermes update`` no longer costs the occurrence a full period.

While a retry is pending the failure notice is suppressed (Cowork re-runs silently); a
run that reaches the model — success or not — resets the ladder. Disable with
``cron.retry_unreachable: false`` (network ladder) or
``cron.retry_gateway_unavailable: false`` (gateway re-run) in config.yaml.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from cron.constants import is_recurring
from hermes_time import now as _hermes_now

logger = logging.getLogger("cron.scheduler")

# Cowork's ladder: re-run after 5, 15, then 30 minutes; then give up until the
# schedule's own next occurrence.
RETRY_DELAYS_SECONDS: tuple[int, ...] = (300, 900, 1800)

# Gateway-refused fires: one rung, two minutes out. An update drain restarts the owner in
# well under that, and a second miss is handled by the schedule's own next occurrence.
GATEWAY_RETRY_DELAYS_SECONDS: tuple[int, ...] = (120,)

# The ladder a parked retry belongs to, persisted beside the instant so switching classes
# starts that class's ladder at its first rung instead of inheriting the other's count.
KIND_MODEL = "model"
KIND_GATEWAY = "gateway"

# Persisted on the job while a retry cycle is active: {"attempt": <1-based count of
# retries already scheduled>, "at": <ISO instant of the pending retry>, "expr": <the cron
# expression it was planned under>, "kind": <ladder this rung belongs to>}. Cleared by any
# run that reached the model.
STATE_KEY = "unreachable_retry"


def _ladder_delays(kind: str) -> tuple[int, ...]:
    """The delay ladder for *kind*: the network ladder, or the single gateway rung."""
    return GATEWAY_RETRY_DELAYS_SECONDS if kind == KIND_GATEWAY else RETRY_DELAYS_SECONDS


def _cron_switch(key: str, cfg: Optional[dict] = None) -> bool:
    """One of the cron reliability switches, default ON. An unreadable config keeps the
    reliability default rather than silently disarming a retry."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
        except Exception:  # health: allow BLE001 -- a broken config must not disable the retry net; ON is the documented fallback
            return True
    cron_cfg = (cfg or {}).get("cron")
    if not isinstance(cron_cfg, dict):
        return True
    return cron_cfg.get(key) is not False


def retry_enabled(cfg: Optional[dict] = None) -> bool:
    """``cron.retry_unreachable`` — default ON (spend-neutral: only fires when zero
    model calls were made)."""
    return _cron_switch("retry_unreachable", cfg)


def gateway_retry_enabled(cfg: Optional[dict] = None) -> bool:
    """``cron.retry_gateway_unavailable`` — default ON, same spend-neutral guarantee as
    the network ladder (a refused fire never reached the agent)."""
    return _cron_switch("retry_gateway_unavailable", cfg)


def is_model_unreachable_failure(exc: BaseException, agent: Any = None) -> bool:
    """True when *exc* is a transient network/DNS failure and *agent* (may be ``None``)
    never completed a model call — the run consumed nothing and executed nothing."""
    if int(getattr(agent, "session_api_calls", 0) or 0) > 0:
        return False
    from cron.scheduler_preflight import _is_transient_provider_resolve_error

    return _is_transient_provider_resolve_error(exc)


def _is_gateway_refusal(exc: BaseException) -> bool:
    """True when *exc* (or anything in its cause chain) is the client's
    ``GatewayUnavailableError``: no ready owner to dial, so nothing was submitted."""
    from hermes_cli.gateway_client import GatewayUnavailableError

    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, GatewayUnavailableError):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def is_gateway_unavailable_failure(exc: BaseException, agent: Any = None) -> bool:
    """True when *exc* is a gateway refusal and *agent* (may be ``None``) never completed
    a model call — the run executed nothing and spent nothing, so one re-run is free."""
    if int(getattr(agent, "session_api_calls", 0) or 0) > 0:
        return False
    return _is_gateway_refusal(exc)


def _ladder_applies(job: dict[str, Any], kind: str = KIND_MODEL) -> bool:
    """The ladder's applicability gate: recurring, not paused, and enabled in config."""
    enabled = gateway_retry_enabled() if kind == KIND_GATEWAY else retry_enabled()
    return is_recurring(job) and job.get("state") != "paused" and enabled


def _parked_attempt(job: dict[str, Any], kind: str) -> int:
    """Rungs already scheduled on *kind*'s ladder. A cycle parked by the other class
    starts this one at its first rung (its failure is a different event)."""
    state = job.get(STATE_KEY) or {}
    if (state.get("kind") or KIND_MODEL) != kind:
        return 0
    return int(state.get("attempt") or 0)


def _ladder_instant(job: dict[str, Any], natural_next: Optional[datetime],
                    now: datetime, kind: str = KIND_MODEL) -> Optional[datetime]:
    """The ladder instant ``plan_retry`` parks for this flagged failure, or None when it
    parks nothing: not recurring, paused, disabled, ladder exhausted, or the schedule's
    own *natural_next* fires at or before the rung. The single source of that decision
    for ``plan_retry`` and ``will_retry``."""
    delays = _ladder_delays(kind)
    attempt = _parked_attempt(job, kind)
    if attempt >= len(delays) or not _ladder_applies(job, kind):
        return None
    # late: jobs imports this module's helpers
    from cron.jobs import _instant_at_or_before, _seconds_after

    retry_dt = _seconds_after(now, delays[attempt])
    if natural_next is not None and _instant_at_or_before(natural_next, retry_dt):
        return None  # the natural occurrence IS the retry
    return retry_dt


def will_retry(job: dict[str, Any], gateway: bool = False) -> bool:
    """True iff ``plan_retry`` will park a re-run for this flagged failure, so the scheduler
    may hold the interim notice. Called before ``mark_job_run``."""
    repeat = job.get("repeat") or {}
    times = repeat.get("times")
    counted = 0 if is_retry_run(job) else 1  # a ladder re-run does not count (``ladder_rung``)
    if times is not None and times > 0 and int(repeat.get("completed") or 0) + counted >= times:
        return False  # _advance_after_run completes the job; plan_retry never runs
    from cron.jobs import _parse_aware, compute_next_run

    now = _hermes_now()
    natural_next = _parse_aware(compute_next_run(
        job.get("schedule") or {}, now.isoformat(), timezone=job.get("timezone")))
    if natural_next is None:
        # The natural occurrence is uncomputable (e.g. croniter missing): _advance_after_run
        # leaves the record state=error — terminal — so plan_retry never runs either.
        return False
    kind = KIND_GATEWAY if gateway else KIND_MODEL
    return _ladder_instant(job, natural_next, now, kind) is not None


def clear_state(job: dict[str, Any]) -> None:
    """A run reached the model (any outcome): the ladder resets."""
    job.pop(STATE_KEY, None)


def is_retry_fire(job: dict[str, Any], next_run: str) -> bool:
    """True for the exact ladder instant parked by ``plan_retry`` (off the cron lattice).

    The expression fingerprint keeps a direct ``jobs.json`` schedule edit from inheriting the
    exception, as in ``cron.quota_hold.is_recovery_fire``.
    """
    state = job.get(STATE_KEY) or {}
    return state.get("at") == next_run and state.get("expr") == (job.get("schedule") or {}).get("expr")


def is_retry_run(job: dict[str, Any]) -> bool:
    """True when this dispatch snapshot fired the ladder instant parked by ``plan_retry``: a
    re-run of an occurrence that already counted toward ``repeat``, so it must not count again.
    Manual runs carry no ``_scheduled_instant`` and always count."""
    from cron.occurrences import scheduled_instant

    instant = job.get("_scheduled_instant")
    return instant is not None and scheduled_instant((job.get(STATE_KEY) or {}).get("at")) == instant


def plan_retry(job: dict[str, Any], gateway: bool = False) -> bool:
    """Called under the jobs lock AFTER ``_advance_after_run`` computed the schedule's
    natural ``next_run_at`` for a failed, flagged run. Pulls ``next_run_at`` earlier to
    the ladder instant when that is sooner than the natural occurrence; exhausted or
    inapplicable cycles clear state and leave the schedule untouched. Returns True when
    a retry was scheduled. *gateway* selects the single gateway rung instead of the
    network ladder."""
    from cron.jobs import _parse_aware  # late: jobs imports this module's helpers

    kind = KIND_GATEWAY if gateway else KIND_MODEL
    delays = _ladder_delays(kind)
    retry_dt = _ladder_instant(job, _parse_aware(job.get("next_run_at")), _hermes_now(), kind)
    attempt = _parked_attempt(job, kind)
    if retry_dt is None:
        if attempt >= len(delays) and _ladder_applies(job, kind):
            # Ladder exhausted: fall back to the natural schedule and reset so the NEXT
            # occurrence gets a fresh ladder if the condition is still there.
            logger.warning(
                "Job '%s': %s unreachable after %d automatic re-run(s) — waiting for the "
                "scheduled occurrence at %s",
                job.get("name", job.get("id", "?")),
                "gateway" if gateway else "model", attempt, job.get("next_run_at"))
        clear_state(job)
        return False
    delay = delays[attempt]
    retry_at = retry_dt.isoformat()
    job[STATE_KEY] = {"attempt": attempt + 1, "at": retry_at, "kind": kind,
                      "expr": job["schedule"].get("expr")}
    job["next_run_at"] = retry_at
    if job.get("state") != "paused":
        job["state"] = "scheduled"
    logger.info(
        "Job '%s': %s unreachable with zero API calls — automatic re-run %d/%d in %ds "
        "(at %s)",
        job.get("name", job.get("id", "?")), "gateway" if gateway else "model",
        attempt + 1, len(delays), delay, retry_at)
    return True
