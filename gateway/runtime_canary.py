"""Runtime canary checks for unhealthy-but-running gateway conditions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_GATEWAY_STALE_SECONDS = 180
DEFAULT_QQ_STALE_SECONDS = 900
DEFAULT_SESSION_STUCK_SECONDS = 900
DEFAULT_BACKGROUND_STUCK_SECONDS = 1200
DEFAULT_PROVIDER_FAILURE_THRESHOLD = 3
DEFAULT_ALERT_THROTTLE_SECONDS = 1800


def _severity_rank(value: Any) -> int:
    normalized = str(value or "").strip().lower()
    if normalized == "critical":
        return 2
    if normalized == "warning":
        return 1
    return 0


def _status_from_issues(issues: list[dict[str, Any]]) -> str:
    max_rank = max((_severity_rank(issue.get("severity")) for issue in issues), default=0)
    if max_rank >= 2:
        return "critical"
    if max_rank >= 1:
        return "warning"
    return "healthy"


def _parse_iso_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc_now(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _summarize_issue(issue: dict[str, Any]) -> str:
    message = str(issue.get("message") or "").strip()
    if message:
        return message
    return str(issue.get("code") or "runtime_canary_issue")


def evaluate_runtime_health(
    runtime_status: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    gateway_stale_seconds: int = DEFAULT_GATEWAY_STALE_SECONDS,
    qq_stale_seconds: int = DEFAULT_QQ_STALE_SECONDS,
    session_stuck_seconds: int = DEFAULT_SESSION_STUCK_SECONDS,
    background_stuck_seconds: int = DEFAULT_BACKGROUND_STUCK_SECONDS,
    provider_failure_threshold: int = DEFAULT_PROVIDER_FAILURE_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate persisted runtime status and return operator-facing issues."""
    checked_at = _utc_now(now)
    issues: list[dict[str, Any]] = []

    if not isinstance(runtime_status, dict) or not runtime_status:
        issues.append(
            {
                "code": "runtime_status_missing",
                "severity": "critical",
                "message": "gateway runtime status is missing",
            }
        )
        return {
            "healthy": False,
            "status": "critical",
            "checked_at": checked_at.isoformat(),
            "issue_count": 1,
            "issues": issues,
            "summary": _summarize_issue(issues[0]),
        }

    gateway_state = str(runtime_status.get("gateway_state") or "").strip().lower()
    updated_at = _parse_iso_timestamp(runtime_status.get("updated_at"))
    if gateway_state == "running" and updated_at is not None:
        age_seconds = int((checked_at - updated_at).total_seconds())
        if age_seconds > int(gateway_stale_seconds):
            issues.append(
                {
                    "code": "gateway_runtime_stale",
                    "severity": "critical",
                    "age_seconds": age_seconds,
                    "message": f"gateway runtime snapshot stale for {age_seconds}s",
                }
            )

    platforms = runtime_status.get("platforms") or {}
    qq_state = platforms.get("qq_napcat") if isinstance(platforms, dict) else None
    if isinstance(qq_state, dict):
        qq_updated = _parse_iso_timestamp(qq_state.get("updated_at"))
        qq_state_name = str(qq_state.get("state") or "").strip().lower()
        if qq_updated is not None and qq_state_name in {"connected", "running", "ready", "ok"}:
            qq_age_seconds = int((checked_at - qq_updated).total_seconds())
            if qq_age_seconds > int(qq_stale_seconds):
                issues.append(
                    {
                        "code": "qq_connectivity_stale",
                        "severity": "critical",
                        "platform": "qq_napcat",
                        "age_seconds": qq_age_seconds,
                        "message": f"qq_napcat connectivity stale for {qq_age_seconds}s",
                    }
                )

    runtime_summary = runtime_status.get("runtime_summary") or {}
    if not isinstance(runtime_summary, dict):
        runtime_summary = {}

    for session in runtime_summary.get("active_sessions") or []:
        if not isinstance(session, dict):
            continue
        age_seconds = int(session.get("age_seconds") or 0)
        if age_seconds > int(session_stuck_seconds):
            platform = str(session.get("platform") or "unknown")
            chat_id = str(session.get("chat_id") or "unknown")
            current_tool = str(session.get("current_tool") or "").strip() or "unknown"
            issues.append(
                {
                    "code": "active_session_stuck",
                    "severity": "warning",
                    "platform": platform,
                    "chat_id": chat_id,
                    "age_seconds": age_seconds,
                    "current_tool": current_tool,
                    "message": (
                        f"active session stuck on {platform}:{chat_id} "
                        f"for {age_seconds}s (tool={current_tool})"
                    ),
                }
            )

    background = runtime_summary.get("background_jobs") or {}
    active_jobs = background.get("active") if isinstance(background, dict) else None
    for job in active_jobs or []:
        if not isinstance(job, dict):
            continue
        age_seconds = int(job.get("age_seconds") or 0)
        status = str(job.get("status") or "").strip().lower()
        if status in {"running", "queued", "cancelling"} and age_seconds > int(background_stuck_seconds):
            task_id = str(job.get("task_id") or "unknown")
            issues.append(
                {
                    "code": "background_work_stuck",
                    "severity": "warning",
                    "task_id": task_id,
                    "status": status,
                    "age_seconds": age_seconds,
                    "message": f"background work stuck for {age_seconds}s (task={task_id}, status={status})",
                }
            )

    model = runtime_summary.get("model") or {}
    if isinstance(model, dict):
        degraded_provider = str(model.get("degraded_provider") or "").strip()
        failure_count = int(model.get("degraded_failures") or 0)
        degraded_until = _parse_iso_timestamp(model.get("degraded_cooldown_until"))
        if (
            degraded_provider
            and failure_count >= int(provider_failure_threshold)
            and (degraded_until is None or degraded_until >= checked_at)
        ):
            degraded_model = str(model.get("degraded_model") or model.get("active_model") or "").strip()
            degraded_reason = str(model.get("degraded_reason") or "unknown").strip()
            issues.append(
                {
                    "code": "provider_degraded",
                    "severity": "warning",
                    "provider": degraded_provider,
                    "model": degraded_model,
                    "reason": degraded_reason,
                    "failure_count": failure_count,
                    "cooldown_until": degraded_until.isoformat() if degraded_until else None,
                    "message": (
                        f"provider degraded: {degraded_provider}/{degraded_model or 'unknown'} "
                        f"({degraded_reason}, failures={failure_count})"
                    ),
                }
            )

    summary = "; ".join(_summarize_issue(issue) for issue in issues) if issues else "runtime canary healthy"
    status = _status_from_issues(issues)
    return {
        "healthy": not issues,
        "status": status,
        "checked_at": checked_at.isoformat(),
        "issue_count": len(issues),
        "critical_count": sum(1 for issue in issues if _severity_rank(issue.get("severity")) >= 2),
        "warning_count": sum(1 for issue in issues if _severity_rank(issue.get("severity")) == 1),
        "issues": issues,
        "summary": summary,
    }


def format_canary_alert(evaluation: dict[str, Any]) -> str:
    """Render a compact operator-facing alert message."""
    issues = evaluation.get("issues") or []
    status = str(evaluation.get("status") or "warning").strip().lower() or "warning"
    lines = [f"Runtime canary alert ({status})", ""]
    for issue in issues:
        lines.append(f"- {_summarize_issue(issue)}")
    checked_at = str(evaluation.get("checked_at") or "").strip()
    if checked_at:
        lines.extend(["", f"Checked at: {checked_at}"])
    return "\n".join(lines)


def run_runtime_canary(
    *,
    runtime_status: dict[str, Any] | None,
    alert_state: dict[str, Any] | None = None,
    alert_target: str | None = None,
    now: datetime | None = None,
    throttle_seconds: int = DEFAULT_ALERT_THROTTLE_SECONDS,
    gateway_stale_seconds: int = DEFAULT_GATEWAY_STALE_SECONDS,
    qq_stale_seconds: int = DEFAULT_QQ_STALE_SECONDS,
    session_stuck_seconds: int = DEFAULT_SESSION_STUCK_SECONDS,
    background_stuck_seconds: int = DEFAULT_BACKGROUND_STUCK_SECONDS,
    provider_failure_threshold: int = DEFAULT_PROVIDER_FAILURE_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate runtime health and decide whether a throttled alert should fire."""
    checked_at = _utc_now(now)
    normalized_target = str(alert_target or "").strip() or None
    evaluation = evaluate_runtime_health(
        runtime_status,
        now=checked_at,
        gateway_stale_seconds=gateway_stale_seconds,
        qq_stale_seconds=qq_stale_seconds,
        session_stuck_seconds=session_stuck_seconds,
        background_stuck_seconds=background_stuck_seconds,
        provider_failure_threshold=provider_failure_threshold,
    )

    state = dict(alert_state or {})
    state["last_checked_at"] = checked_at.isoformat()
    last_alerts = dict(state.get("last_alerts") or {})
    current_codes = [str(issue.get("code") or "").strip() for issue in evaluation.get("issues") or []]
    current_codes = [code for code in current_codes if code]
    previous_codes = {
        str(code).strip()
        for code in last_alerts.keys()
        if str(code).strip()
    }
    resolved_codes = sorted(previous_codes - set(current_codes))
    for code in resolved_codes:
        last_alerts.pop(code, None)

    should_alert = False
    throttled = False
    if current_codes:
        for code in current_codes:
            previous = last_alerts.get(code) if isinstance(last_alerts.get(code), dict) else {}
            previous_sent = _parse_iso_timestamp(previous.get("sent_at"))
            previous_target = str(previous.get("target") or "").strip() or None
            if (
                previous_target != normalized_target
                or previous_sent is None
                or (checked_at - previous_sent).total_seconds() >= int(throttle_seconds)
            ):
                should_alert = True
        throttled = not should_alert and bool(normalized_target)
        should_deliver = should_alert and bool(normalized_target)
        if should_deliver:
            for issue in evaluation.get("issues") or []:
                code = str(issue.get("code") or "").strip()
                if code:
                    last_alerts[code] = {
                        "sent_at": checked_at.isoformat(),
                        "message": _summarize_issue(issue),
                        "target": normalized_target,
                    }
    else:
        state["last_healthy_at"] = checked_at.isoformat()

    state["last_alerts"] = last_alerts
    state["last_issue_codes"] = current_codes
    state["last_status"] = evaluation.get("status")
    state["last_alert_target"] = normalized_target

    return {
        "evaluation": evaluation,
        "should_alert": bool(current_codes) and should_alert and bool(normalized_target),
        "throttled": throttled,
        "alert_target": normalized_target,
        "alert_text": format_canary_alert(evaluation) if current_codes else None,
        "new_issue_codes": sorted(set(current_codes)),
        "resolved_issue_codes": resolved_codes,
        "alert_state": state,
    }
