"""Sanitized, persisted per-job retry and circuit health for cron jobs."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION = "syntheos.hermes_cron_job_health.v1"
MAX_RETRY_SECONDS = 7 * 24 * 3600
OPEN_THRESHOLD = 3
SERIES_WINDOW_SECONDS = 3600
PROBE_LEASE_SECONDS = 300

_STATES = {"unknown", "healthy", "degraded", "circuit_open", "half_open", "suppressed"}
_REASONS = {
    None,
    "provider_quota_exhausted",
    "provider_rate_limited",
    "auth",
    "timeout",
    "transport",
    "tool",
    "unknown",
}


def utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def profile_name(home: Path) -> str:
    return home.name if home.parent.name == "profiles" else "default"


def default_health(
    job_id: str, profile: str, observed_at: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "profile": profile,
        "observed_at": observed_at,
        "state": "unknown",
        "reason_code": None,
        "reason_summary": None,
        "failure_fingerprint": None,
        "consecutive_failures": 0,
        "series_started_at": None,
        "last_failure_at": None,
        "last_success_at": None,
        "retry_not_before": None,
        "retry_after_seconds": None,
        "retry_after_clamped": False,
        "circuit_opened_at": None,
        "probe_failures": 0,
        "probe_claim": None,
        "suppressed_runs": 0,
        "last_suppressed_at": None,
    }


def coerce_health(raw: Any, job_id: str, profile: str) -> Dict[str, Any]:
    """Fail safe to unknown when persisted health is absent or malformed."""
    expected = default_health(job_id, profile)
    state = raw.get("state") if isinstance(raw, dict) else None
    reason_code = raw.get("reason_code") if isinstance(raw, dict) else None
    if (
        not isinstance(raw, dict)
        or set(raw) != set(expected)
        or raw.get("schema_version") != SCHEMA_VERSION
        or not isinstance(raw.get("job_id"), str)
        or not raw.get("job_id")
        or len(raw["job_id"]) > 128
        or not isinstance(raw.get("profile"), str)
        or not raw.get("profile")
        or len(raw["profile"]) > 128
        or not isinstance(state, str)
        or state not in _STATES
        or (reason_code is not None and not isinstance(reason_code, str))
        or reason_code not in _REASONS
        or not _nullable_text(raw.get("reason_summary"), max_length=240)
        or not _fingerprint(raw.get("failure_fingerprint"))
        or not all(
            _nonnegative_int(raw.get(key))
            for key in ("consecutive_failures", "probe_failures", "suppressed_runs")
        )
        or not all(
            _nullable_timestamp(raw.get(key))
            for key in (
                "observed_at",
                "series_started_at",
                "last_failure_at",
                "last_success_at",
                "retry_not_before",
                "circuit_opened_at",
                "last_suppressed_at",
            )
        )
        or not _retry_seconds(raw.get("retry_after_seconds"))
        or not isinstance(raw.get("retry_after_clamped"), bool)
        or not _probe_claim(raw.get("probe_claim"))
    ):
        return default_health(job_id, profile)
    health = {key: raw[key] for key in expected}
    health["job_id"] = job_id
    health["profile"] = profile
    return health


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nullable_text(value: Any, *, max_length: int) -> bool:
    return value is None or (isinstance(value, str) and len(value) <= max_length)


def _nullable_timestamp(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.endswith("Z") and _parse_timestamp(value) is not None
    )


def _fingerprint(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _retry_seconds(value: Any) -> bool:
    return value is None or (_nonnegative_int(value) and value <= MAX_RETRY_SECONDS)


def _probe_claim(value: Any) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, dict)
        and set(value) == {"token", "claimed_at", "lease_expires_at"}
        and isinstance(value.get("token"), str)
        and 16 <= len(value["token"]) <= 128
        and _nullable_timestamp(value.get("claimed_at"))
        and value.get("claimed_at") is not None
        and _nullable_timestamp(value.get("lease_expires_at"))
        and value.get("lease_expires_at") is not None
    )


def _normalized_failure_text(text: str) -> str:
    summary = text.lower()
    summary = re.sub(
        r"\b(?:request|trace|correlation)[-_ ]?id\s*[:=]\s*\S+", "", summary
    )
    summary = re.sub(
        r"(?:retry(?:[- ]after)?\s*[:=]?\s*)\d+\s*s?",
        "retry deadline supplied",
        summary,
    )
    summary = re.sub(r"(?:[a-zA-Z]:)?[/\\][^\s]+", "[path]", summary)
    summary = re.sub(
        r"\b(?:sk|pk|api|token|key)[-_][a-z0-9_-]+\b", "[redacted]", summary
    )
    summary = re.sub(
        r"(?:^|[.;])[^.;]*(?:credential|api key|access token)[^.;]*[.;]?", " ", summary
    )
    summary = re.sub(r"\d{4}-\d{2}-\d{2}[t ][0-9:.+z-]+", "[timestamp]", summary)
    return " ".join(summary.split())[:240] or "terminal cron failure"


def _reason_summary(reason_code: str, retry_after: Optional[int]) -> str:
    summaries = {
        "provider_quota_exhausted": "provider quota exhausted",
        "provider_rate_limited": "provider rate limited",
        "auth": "provider authentication failed",
        "timeout": "provider request timed out",
        "transport": "provider transport failed",
        "tool": "tool execution failed",
        "unknown": "terminal cron failure",
    }
    summary = summaries.get(reason_code, "terminal cron failure")
    if retry_after is not None and reason_code in {
        "provider_quota_exhausted",
        "provider_rate_limited",
        "timeout",
        "transport",
    }:
        summary += "; retry deadline supplied"
    return summary


def _has_http_status(text: str, status: int) -> bool:
    code = str(status)
    return bool(
        re.search(
            rf"\b(?:http(?: status)?|status(?: code)?|response status|error code)\s*[:=]?\s*{code}\b",
            text,
        )
        or re.match(rf"^\s*{code}\b", text)
    )


def classify_failure(error: Optional[str]) -> Tuple[str, str, Optional[int]]:
    text = str(error or "")
    lower = text.lower()
    retry_match = re.search(r"retry(?:[- ]after| after)?\s*[:=]?\s*(\d+)\s*s?", lower)
    retry_after = int(retry_match.group(1)) if retry_match else None
    http_429 = _has_http_status(lower, 429)
    if ("quota" in lower or "usage limit" in lower or "usage_limit" in lower) and (
        http_429 or "exhaust" in lower or "reached" in lower or "exceed" in lower
    ):
        code = "provider_quota_exhausted"
    elif http_429 or "rate limit" in lower:
        code = "provider_rate_limited"
    elif _has_http_status(lower, 401) or _has_http_status(lower, 403) or any(
        token in lower
        for token in (
            "unauthorized",
            "forbidden",
            "authentication",
            "credentials expired",
            "no api key configured",
        )
    ):
        code = "auth"
    elif any(token in lower for token in ("timeout", "timed out", "deadline exceeded")):
        code = "timeout"
    elif any(
        token in lower
        for token in ("connection", "network", "dns", "socket", "transport")
    ):
        code = "transport"
    elif "tool" in lower:
        code = "tool"
    else:
        code = "unknown"
    return code, _reason_summary(code, retry_after), retry_after


def record_result(
    raw: Any,
    *,
    job_id: str,
    profile: str,
    success: bool,
    error: Optional[str],
    now: datetime,
    manual_run: bool = False,
    retry_after_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    health = coerce_health(raw, job_id, profile)
    if manual_run:
        return health
    now_utc = now.astimezone(timezone.utc)
    now_text = utc_iso(now_utc)
    health["observed_at"] = now_text
    health["probe_claim"] = None
    if success:
        last_reason = health.get("reason_code")
        last_failure = health.get("last_failure_at")
        health.update(default_health(job_id, profile, now_text))
        health.update({
            "state": "healthy",
            "last_success_at": now_text,
            "reason_code": last_reason,
            "last_failure_at": last_failure,
        })
        return health

    reason_code, summary, retry_after = classify_failure(error)
    if retry_after_seconds is not None:
        try:
            structured_retry = int(float(retry_after_seconds))
        except (TypeError, ValueError):
            structured_retry = 0
        if structured_retry > 0:
            retry_after = structured_retry
            summary = _reason_summary(reason_code, retry_after)
    fingerprint_reason = _normalized_failure_text(str(error or ""))
    fingerprint = hashlib.sha256(
        f"{reason_code}\n{fingerprint_reason}".encode()
    ).hexdigest()
    same_fingerprint = health.get("failure_fingerprint") == fingerprint
    previous_failure = _parse_timestamp(health.get("last_failure_at"))
    within_window = (
        previous_failure is not None
        and (now_utc - previous_failure).total_seconds() <= SERIES_WINDOW_SECONDS
    )
    was_probe = health.get("state") == "half_open"
    continuing_series = same_fingerprint and (within_window or was_probe)
    failures = (
        int(health.get("consecutive_failures") or 0) + 1 if continuing_series else 1
    )
    probe_failures = (
        int(health.get("probe_failures") or 0) + 1
        if was_probe and same_fingerprint
        else 0
    )
    open_now = (
        reason_code in {"provider_quota_exhausted", "auth"}
        or (continuing_series and failures >= OPEN_THRESHOLD)
        or (was_probe and same_fingerprint)
    )

    retry_seconds: Optional[int] = None
    clamped = False
    if open_now:
        if retry_after is not None and retry_after >= 300:
            retry_seconds = retry_after
            clamped = retry_seconds > MAX_RETRY_SECONDS
            retry_seconds = min(retry_seconds, MAX_RETRY_SECONDS)
        elif reason_code == "provider_quota_exhausted":
            retry_seconds = 3600
        elif reason_code == "auth":
            retry_seconds = 3600
        else:
            retry_seconds = min(300 * (2**probe_failures), 24 * 3600)

    health.update({
        "state": "circuit_open" if open_now else "degraded",
        "reason_code": reason_code,
        "reason_summary": summary,
        "failure_fingerprint": fingerprint,
        "consecutive_failures": failures,
        "series_started_at": health.get("series_started_at")
        if continuing_series
        else now_text,
        "last_failure_at": now_text,
        "retry_not_before": utc_iso(now_utc + timedelta(seconds=retry_seconds))
        if retry_seconds is not None
        else None,
        "retry_after_seconds": retry_seconds,
        "retry_after_clamped": clamped,
        "circuit_opened_at": (health.get("circuit_opened_at") or now_text)
        if open_now
        else None,
        "probe_failures": probe_failures,
    })
    return health


def suppress_or_claim(
    raw: Any,
    *,
    job_id: str,
    profile: str,
    now: datetime,
    claim_probe: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Return ``run``, ``suppress``, or ``probe`` and the updated health object."""
    health = coerce_health(raw, job_id, profile)
    now_utc = now.astimezone(timezone.utc)
    retry_at = _parse_timestamp(health.get("retry_not_before"))
    if (
        health.get("state") not in {"circuit_open", "suppressed", "half_open"}
        or retry_at is None
    ):
        return "run", health
    now_text = utc_iso(now_utc)
    if now_utc < retry_at:
        health.update({
            "observed_at": now_text,
            "state": "half_open"
            if health.get("state") == "half_open"
            else "suppressed",
            "suppressed_runs": int(health.get("suppressed_runs") or 0) + 1,
            "last_suppressed_at": now_text,
        })
        return "suppress", health
    claim = health.get("probe_claim")
    lease = (
        _parse_timestamp(claim.get("lease_expires_at"))
        if isinstance(claim, dict)
        else None
    )
    if lease is not None and lease > now_utc:
        health.update({
            "observed_at": now_text,
            "state": "half_open",
            "suppressed_runs": int(health.get("suppressed_runs") or 0) + 1,
            "last_suppressed_at": now_text,
        })
        return "suppress", health
    if not claim_probe:
        return "probe", health
    health.update({
        "observed_at": now_text,
        "state": "half_open",
        "probe_claim": {
            "token": uuid.uuid4().hex,
            "claimed_at": now_text,
            "lease_expires_at": utc_iso(
                now_utc + timedelta(seconds=PROBE_LEASE_SECONDS)
            ),
        },
    })
    return "probe", health


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None
