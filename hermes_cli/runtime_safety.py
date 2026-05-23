"""Runtime safety scanners for cron, gateway control, and resume state.

This module is intentionally display-safe: scanner result objects never carry raw
cron prompts or secret-bearing command text.  Callers may inspect source records
for classification, but status/doctor/preflight should only print the sanitized
fields exposed here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

ONESHOT_GRACE_SECONDS = 120
RESUME_PENDING_TTL_SECONDS = 10 * 60

_GATEWAY_SERVICE_RE = re.compile(r"\bhermes-gateway(?:\.service)?\b", re.IGNORECASE)
_READONLY_SYSTEMCTL_RE = re.compile(
    r"\bsystemctl\s+(?:--user\s+)?(?:status|show|is-active|is-enabled|list-units)\b[^\n;|&]*\bhermes-gateway(?:\.service)?\b",
    re.IGNORECASE,
)
_READONLY_HERMES_GATEWAY_RE = re.compile(r"\bhermes\s+gateway\s+status\b", re.IGNORECASE)
_READONLY_JOURNAL_RE = re.compile(
    r"\bjournalctl\b[^\n;|&]*\b(?:-u\s+)?hermes-gateway(?:\.service)?\b",
    re.IGNORECASE,
)
_MUTATING_GATEWAY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), reason)
    for pattern, reason in (
        (r"\bsystemctl\s+(?:--user\s+)?(?:restart|stop|kill|disable|mask|try-restart|reload-or-restart)\b[^\n;|&]*\bhermes-gateway(?:\.service)?\b", "systemctl_gateway_lifecycle"),
        (r"\bhermes\s+gateway\s+(?:stop|restart)\b", "hermes_gateway_lifecycle"),
        (r"\bhermes\s+update\b", "hermes_update_gateway_restart"),
        (r"\b(?:pkill|killall)\b[^\n]*\b(?:hermes(?:-gateway)?|gateway\.run|gateway/run\.py)\b", "name_based_gateway_kill"),
        (r"\bkill\b[^\n]*(?:-9|-KILL|-SIGKILL|-s\s+(?:9|KILL|SIGKILL))[^\n]*(?:\$\([^)]*hermes-gateway[^)]*\)|`[^`]*hermes-gateway[^`]*`)", "gateway_pid_sigkill"),
    )
)


@dataclass(frozen=True)
class CronRisk:
    job_id: str
    name: str
    enabled: bool
    state: str
    schedule_kind: str
    schedule: str
    next_run_at: str | None
    reason: str
    severity: str
    script: str | None = None
    no_agent: bool = False

    def to_display(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def _parse_dt(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_now().tzinfo)


def _safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _schedule_kind(job: dict[str, Any]) -> str:
    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        return _safe_text(schedule.get("kind"), "unknown") or "unknown"
    return "unknown"


def _schedule_display(job: dict[str, Any]) -> str:
    display = _safe_text(job.get("schedule_display")).strip()
    if display:
        return display
    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        for key in ("display", "expr", "run_at", "value"):
            text = _safe_text(schedule.get(key)).strip()
            if text:
                return text
    return "?"


def sanitize_script_path(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Path(text).name if not Path(text).is_absolute() else str(Path(text).name)
    except Exception:
        return text.split("/")[-1][-80:]


def sanitize_job_for_display(job: dict[str, Any], *, reason: str = "", severity: str = "info") -> dict[str, Any]:
    """Return a prompt-free, secret-minimized cron job display object."""
    return {
        "job_id": _safe_text(job.get("id"), "unknown"),
        "name": (_safe_text(job.get("name"), "cron job").strip() or "cron job")[:80],
        "enabled": bool(job.get("enabled", True)),
        "state": (_safe_text(job.get("state"), "scheduled").strip() or "scheduled")[:40],
        "schedule_kind": _schedule_kind(job),
        "schedule": _schedule_display(job)[:120],
        "next_run_at": _safe_text(job.get("next_run_at"), "") or None,
        "reason": reason,
        "severity": severity,
        "script": sanitize_script_path(job.get("script")),
        "no_agent": bool(job.get("no_agent", False)),
    }


def _script_body_for_scan(script: Any, *, max_chars: int = 20000) -> str:
    """Best-effort read of a cron script body for lifecycle-command scanning.

    Cron script paths are constrained to HERMES_HOME/scripts by the scheduler;
    mirror that boundary here and never return the body to callers. The text is
    only fed into classifiers so a harmless-looking script name cannot hide a
    `systemctl kill hermes-gateway.service` payload.
    """
    if not script:
        return ""
    try:
        scripts_dir = (get_hermes_home() / "scripts").resolve()
        raw = Path(str(script).strip()).expanduser()
        path = raw.resolve() if raw.is_absolute() else (scripts_dir / raw).resolve()
        path.relative_to(scripts_dir)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def classify_gateway_control_text(text: Any) -> str | None:
    """Classify text as mutating gateway control, adjacent, or safe/none."""
    if not text:
        return None
    raw = str(text)
    if not raw.strip():
        return None
    if (
        _READONLY_SYSTEMCTL_RE.search(raw)
        or _READONLY_HERMES_GATEWAY_RE.search(raw)
        or _READONLY_JOURNAL_RE.search(raw)
    ) and not any(pattern.search(raw) for pattern, _reason in _MUTATING_GATEWAY_PATTERNS):
        return None
    for pattern, reason in _MUTATING_GATEWAY_PATTERNS:
        if pattern.search(raw):
            return reason
    lowered = raw.lower()
    if "hermes-gateway" in lowered or "hermes gateway" in lowered:
        if any(word in lowered for word in ("restart", "stop", "kill", "sigkill", "disable", "mask")):
            return "gateway_control_adjacent"
    return None


def is_job_due(job: dict[str, Any], now: datetime | None = None) -> bool:
    if now is None:
        now = _now()
    if not job.get("enabled", True):
        return False
    next_run = _parse_dt(job.get("next_run_at"))
    return bool(next_run and next_run <= now)


def is_expired_oneshot(job: dict[str, Any], now: datetime | None = None, grace_seconds: int = ONESHOT_GRACE_SECONDS) -> bool:
    if now is None:
        now = _now()
    schedule = job.get("schedule") or {}
    if not isinstance(schedule, dict) or schedule.get("kind") != "once":
        return False
    if not job.get("enabled", True):
        return False
    if job.get("last_run_at"):
        return False
    run_at = _parse_dt(job.get("next_run_at")) or _parse_dt(schedule.get("run_at"))
    if run_at is None:
        return False
    return run_at < now - timedelta(seconds=grace_seconds)


def classify_cron_job(job: dict[str, Any], now: datetime | None = None) -> CronRisk | None:
    if now is None:
        now = _now()
    display = sanitize_job_for_display(job)
    if is_expired_oneshot(job, now):
        return CronRisk(**{**display, "reason": "expired_oneshot", "severity": "warning"})

    scan_text = "\n".join(
        _safe_text(job.get(key))
        for key in ("name", "script", "prompt")
    )
    reason = classify_gateway_control_text(scan_text)
    if not reason:
        return None

    enabled = bool(job.get("enabled", True))
    due = is_job_due(job, now)
    if enabled and due and reason != "gateway_control_adjacent":
        severity = "critical"
        public_reason = "unsafe_gateway_control_due"
    elif enabled and reason != "gateway_control_adjacent":
        severity = "warning"
        public_reason = "unsafe_gateway_control_scheduled"
    elif reason == "gateway_control_adjacent":
        severity = "info" if not enabled else "warning"
        public_reason = "gateway_control_adjacent"
    else:
        severity = "warning"
        public_reason = "unsafe_gateway_control_disabled"
    return CronRisk(**{**display, "reason": public_reason, "severity": severity})


def scan_cron_jobs(jobs: Iterable[dict[str, Any]], now: datetime | None = None) -> list[CronRisk]:
    if now is None:
        now = _now()
    risks: list[CronRisk] = []
    for job in jobs:
        try:
            risk = classify_cron_job(job, now)
        except Exception:
            risk = CronRisk(
                job_id=_safe_text(job.get("id"), "unknown") if isinstance(job, dict) else "unknown",
                name="cron job",
                enabled=False,
                state="unknown",
                schedule_kind="unknown",
                schedule="?",
                next_run_at=None,
                reason="safety_scan_error",
                severity="critical",
                script=None,
                no_agent=False,
            )
        if risk:
            risks.append(risk)
    return risks


def load_cron_jobs_for_scan() -> list[dict[str, Any]]:
    path = get_hermes_home() / "cron" / "jobs.json"
    try:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        jobs = data.get("jobs", [])
        return jobs if isinstance(jobs, list) else []
    except FileNotFoundError:
        return []
    except Exception:
        return []


def load_session_entries_for_scan() -> list[dict[str, Any]]:
    path = get_hermes_home() / "sessions" / "sessions.json"
    try:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        return [entry for entry in data.values() if isinstance(entry, dict)]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def scan_resume_pending_sessions(
    session_entries: Iterable[dict[str, Any]],
    now: datetime | None = None,
    ttl_seconds: int = RESUME_PENDING_TTL_SECONDS,
) -> list[dict[str, Any]]:
    if now is None:
        now = _now()
    stale: list[dict[str, Any]] = []
    for entry in session_entries:
        if not isinstance(entry, dict) or not entry.get("resume_pending"):
            continue
        marked = _parse_dt(entry.get("last_resume_marked_at"))
        if marked is None:
            age = None
            reason = "stale_resume_pending_missing_timestamp"
        else:
            age = max(0, int((now - marked).total_seconds()))
            if age <= ttl_seconds:
                continue
            reason = "stale_resume_pending"
        key = _safe_text(entry.get("session_key"), "")
        stale.append({
            "session_key_hash": _short_hash(key) if key else "unknown",
            "session_id": _safe_text(entry.get("session_id"), "unknown")[:80],
            "platform": _safe_text(entry.get("platform"), "unknown")[:40],
            "reason": reason,
            "age_seconds": age,
            "ttl_seconds": ttl_seconds,
        })
    return stale


def build_runtime_safety_report(now: datetime | None = None) -> dict[str, Any]:
    if now is None:
        now = _now()
    cron_risks = scan_cron_jobs(load_cron_jobs_for_scan(), now)
    stale_resume = scan_resume_pending_sessions(load_session_entries_for_scan(), now)
    counts = {
        "unsafe_gateway_control_due": sum(1 for r in cron_risks if r.reason == "unsafe_gateway_control_due"),
        "unsafe_gateway_control_scheduled": sum(1 for r in cron_risks if r.reason == "unsafe_gateway_control_scheduled"),
        "gateway_control_adjacent": sum(1 for r in cron_risks if r.reason == "gateway_control_adjacent"),
        "expired_oneshot": sum(1 for r in cron_risks if r.reason == "expired_oneshot"),
        "stale_resume_pending": len(stale_resume),
    }
    return {
        "ok": counts["unsafe_gateway_control_due"] == 0 and counts["stale_resume_pending"] == 0,
        "counts": counts,
        "cron_risks": [r.to_display() for r in cron_risks],
        "stale_resume_pending": stale_resume,
    }
