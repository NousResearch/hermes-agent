"""Content-free cron service-health and execution telemetry projection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from agent.monitoring.events import CronExecutionEvent
from agent.monitoring.gateway_health import GatewayMetric, _contains_any, _safe_instance_id
from cron.jobs import (
    _compute_grace_seconds,
    get_catch_up_occurrence_count,
    get_ticker_heartbeat_age,
    get_ticker_success_age,
    load_jobs,
)
from cron.scheduler import get_running_job_ids
from hermes_time import now as _now

logger = logging.getLogger(__name__)
_KNOWN_STATUSES = {"claimed", "running", "completed", "failed", "unknown"}
_KNOWN_SOURCES = {"builtin", "direct", "external"}
_KNOWN_DELIVERY_OUTCOMES = {"delivered", "failed", "suppressed", "suppressed_acked", "not_configured"}
_TERMINAL_STATUSES = {"completed", "failed", "unknown"}


@dataclass(frozen=True, slots=True)
class CronHealthSnapshot:
    metrics: list[GatewayMetric]
    events: list[CronExecutionEvent]


# Opaque job key: same sha256:<24 hex> shape as service.instance.id — never the raw id.
_job_key = _safe_instance_id

_AUTH_RE = re.compile(
    r"\b(?:authentication|authenticated|authenticate|authorization|authorized|authorize|unauthorized|forbidden|bearer|401|403)\b"
    r"|\b(?:access|api|refresh) token\b"
)

# Ordered (predicate, class) rules; first match wins. Auth uses word boundaries so
# "oauth"/"tokenizer"/"HTTP 4015" do not false-positive.
_CRON_ERROR_RULES: tuple[tuple[Callable[[str], bool], str], ...] = (
    (lambda s: _AUTH_RE.search(s) is not None, "auth_failed"),
    (_contains_any("rate limit", "429", "quota"), "rate_limited"),
    (_contains_any("timeout", "timed out"), "timeout"),
    (_contains_any("network", "connection", "dns", "socket", "unreachable"), "network_error"),
    (_contains_any("dispatch", "executor"), "dispatch_failed"),
    (_contains_any("interrupt", "owner exited", "restarted"), "interrupted"),
    (_contains_any("empty response"), "empty_response"),
    (_contains_any("config", "missing", "invalid"), "invalid_config"),
)


def classify_cron_error(raw: Any) -> str:
    text = str(raw or "").lower()
    return next((label for match, label in _CRON_ERROR_RULES if match(text)), "unknown")


def _parse_time(raw: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(raw)) if raw else None
    except (TypeError, ValueError):
        return None


def _duration_ms(record: dict[str, Any]) -> Optional[int]:
    start = _parse_time(record.get("started_at")) or _parse_time(record.get("claimed_at"))
    finish = _parse_time(record.get("finished_at"))
    if start is None or finish is None:
        return None
    try:
        duration = int((finish - start).total_seconds() * 1000)
    except (TypeError, ValueError):
        return None
    return max(0, duration)


def project_execution_event(record: dict[str, Any], *, delivery_outcome: Optional[str] = None) -> CronExecutionEvent:
    status = str(record.get("status") or "unknown").lower()
    source = str(record.get("source") or "unknown").lower()
    outcome = str(delivery_outcome).lower() if delivery_outcome is not None else None
    return CronExecutionEvent(
        status=status if status in _KNOWN_STATUSES else "unknown",
        job_key=_job_key(record.get("job_id")),
        # Unknown non-empty sources are bucketed as "external" (not dropped to unknown).
        source=source if source in _KNOWN_SOURCES or source == "unknown" else "external",
        duration_ms=_duration_ms(record),
        delivery_outcome=outcome if outcome in _KNOWN_DELIVERY_OUTCOMES else None,
        error_class=classify_cron_error(record.get("error")) if status in {"failed", "unknown"} else None,
    )


def emit_execution_state(record: Optional[dict[str, Any]], *, delivery_outcome: Optional[str] = None) -> None:
    """Best-effort lifecycle emit; terminal states synchronously cross the queue barrier."""
    if not record:
        return
    try:
        from agent.monitoring import emitter
        event = project_execution_event(record, delivery_outcome=delivery_outcome)
        target = emitter.get_emitter()
        target.emit(event)
        if event.status in _TERMINAL_STATUSES:
            target.flush(timeout=1.0)
    except Exception:
        logger.debug("cron execution telemetry emit failed", exc_info=True)


def _is_overdue(job: dict[str, Any], now: datetime) -> bool:
    if not job.get("enabled", True):
        return False
    next_run = _parse_time(job.get("next_run_at"))
    schedule = job.get("schedule")
    if next_run is None or not isinstance(schedule, dict):
        return False
    try:
        if next_run.tzinfo is None and now.tzinfo is not None:
            next_run = next_run.replace(tzinfo=now.tzinfo)
        return (now - next_run).total_seconds() > _compute_grace_seconds(schedule)
    except (TypeError, ValueError):
        return False


def _job_metrics(metrics: list[GatewayMetric]) -> None:
    enabled = [job for job in load_jobs() if job.get("enabled", True)]
    metrics.append(GatewayMetric("hermes.cron.jobs.enabled", len(enabled), {}))
    metrics.append(GatewayMetric("hermes.cron.jobs.overdue", sum(1 for job in enabled if _is_overdue(job, _now())), {}))


def _freshness_metric(name: str, reader: Callable[[], Optional[float]]) -> Callable[[list[GatewayMetric]], None]:
    def build(metrics: list[GatewayMetric]) -> None:
        value = reader()
        if value is not None:
            metrics.append(GatewayMetric(name, max(0.0, float(value)), {}))

    return build


def _single_metric(name: str, reader: Callable[[], Any]) -> Callable[[list[GatewayMetric]], None]:
    return lambda metrics: metrics.append(GatewayMetric(name, reader(), {}))


# Each group is independently fail-open so one unavailable source never hides the rest.
# Readers are wrapped in lambdas so monkeypatching this module's names still takes effect.
_METRIC_GROUPS: tuple[tuple[Callable[[list[GatewayMetric]], None], str], ...] = (
    (_freshness_metric("hermes.cron.scheduler.heartbeat_age_seconds", lambda: get_ticker_heartbeat_age()), "cron freshness metric unavailable"),
    (_freshness_metric("hermes.cron.scheduler.last_success_age_seconds", lambda: get_ticker_success_age()), "cron freshness metric unavailable"),
    (_single_metric("hermes.cron.scheduler.catch_up_occurrences", lambda: get_catch_up_occurrence_count()), "cron catch-up metric unavailable"),
    (_job_metrics, "cron job metrics unavailable"),
    (_single_metric("hermes.cron.jobs.running", lambda: len(get_running_job_ids())), "cron running-job metric unavailable"),
)


def build_cron_health_snapshot() -> CronHealthSnapshot:
    metrics: list[GatewayMetric] = []
    for build, failure_msg in _METRIC_GROUPS:
        try:
            build(metrics)
        except Exception:
            logger.debug(failure_msg, exc_info=True)
    return CronHealthSnapshot(metrics=metrics, events=[])


__all__ = [
    "CronHealthSnapshot", "build_cron_health_snapshot", "classify_cron_error", "emit_execution_state",
    "project_execution_event",
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import hashlib  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'GatewayHealthSnapshot': ('agent.monitoring.gateway_health', 'GatewayHealthSnapshot'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
