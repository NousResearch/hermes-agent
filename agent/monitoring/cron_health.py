"""Content-free cron service-health and execution telemetry projection."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from agent.monitoring.events import CronExecutionEvent
from agent.monitoring.gateway_health import GatewayHealthSnapshot, GatewayMetric
from cron.jobs import (
    _compute_grace_seconds,
    get_ticker_heartbeat_age,
    get_ticker_success_age,
    load_jobs,
)
from cron.scheduler import get_running_job_ids
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)
_KNOWN_STATUSES = {"claimed", "running", "completed", "failed", "unknown"}
_KNOWN_SOURCES = {"builtin", "direct", "external"}
_KNOWN_DELIVERY_OUTCOMES = {"delivered", "failed", "suppressed", "not_configured"}


@dataclass(frozen=True, slots=True)
class CronHealthSnapshot:
    metrics: list[GatewayMetric]
    events: list[CronExecutionEvent]


def _now() -> datetime:
    return _hermes_now()


def _job_key(raw: Any) -> str:
    value = str(raw or "unknown").encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(value).hexdigest()[:24]}"


def classify_cron_error(raw: Any) -> str:
    text = str(raw or "").lower()
    if any(value in text for value in ("auth", "token", "unauthorized", "forbidden", "401", "403")):
        return "auth_failed"
    if "rate limit" in text or "429" in text or "quota" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(value in text for value in ("network", "connection", "dns", "socket", "unreachable")):
        return "network_error"
    if "dispatch" in text or "executor" in text:
        return "dispatch_failed"
    if "interrupt" in text or "owner exited" in text or "restarted" in text:
        return "interrupted"
    if "empty response" in text:
        return "empty_response"
    if any(value in text for value in ("config", "missing", "invalid")):
        return "invalid_config"
    return "unknown"


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


def project_execution_event(
    record: dict[str, Any], *, delivery_outcome: Optional[str] = None
) -> CronExecutionEvent:
    status = str(record.get("status") or "unknown").lower()
    source = str(record.get("source") or "unknown").lower()
    outcome = str(delivery_outcome).lower() if delivery_outcome is not None else None
    return CronExecutionEvent(
        status=status if status in _KNOWN_STATUSES else "unknown",
        job_key=_job_key(record.get("job_id")),
        source=source if source in _KNOWN_SOURCES else "unknown",
        duration_ms=_duration_ms(record),
        delivery_outcome=(
            outcome if outcome in _KNOWN_DELIVERY_OUTCOMES else None
        ),
        error_class=(
            classify_cron_error(record.get("error"))
            if status in {"failed", "unknown"}
            else None
        ),
    )


def emit_execution_state(
    record: Optional[dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Best-effort lifecycle emit; terminal states synchronously cross the queue barrier."""
    if not record:
        return
    try:
        from agent.monitoring import emitter

        event = project_execution_event(record, delivery_outcome=delivery_outcome)
        target = emitter.get_emitter()
        target.emit(event)
        if event.status in {"completed", "failed", "unknown"}:
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
        lateness = (now - next_run).total_seconds()
        return lateness > _compute_grace_seconds(schedule)
    except (TypeError, ValueError):
        return False


def build_cron_health_snapshot() -> CronHealthSnapshot:
    metrics: list[GatewayMetric] = []
    for name, reader in (
        ("hermes.cron.scheduler.heartbeat_age_seconds", get_ticker_heartbeat_age),
        ("hermes.cron.scheduler.last_success_age_seconds", get_ticker_success_age),
    ):
        try:
            value = reader()
            if value is not None:
                metrics.append(GatewayMetric(name, max(0.0, float(value)), {}))
        except Exception:
            logger.debug("cron freshness metric unavailable", exc_info=True)

    try:
        jobs = load_jobs()
        enabled = [job for job in jobs if job.get("enabled", True)]
        metrics.append(GatewayMetric("hermes.cron.jobs.enabled", len(enabled), {}))
        metrics.append(
            GatewayMetric(
                "hermes.cron.jobs.overdue",
                sum(1 for job in enabled if _is_overdue(job, _now())),
                {},
            )
        )
    except Exception:
        logger.debug("cron job metrics unavailable", exc_info=True)

    try:
        metrics.append(
            GatewayMetric("hermes.cron.jobs.running", len(get_running_job_ids()), {})
        )
    except Exception:
        logger.debug("cron running-job metric unavailable", exc_info=True)
    return CronHealthSnapshot(metrics=metrics, events=[])


__all__ = [
    "CronHealthSnapshot",
    "build_cron_health_snapshot",
    "classify_cron_error",
    "emit_execution_state",
    "project_execution_event",
]
