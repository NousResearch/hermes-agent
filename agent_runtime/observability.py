"""Read-only health/alert snapshots for the Agent Runtime daemon."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from . import db
from .dashboard_mirror import _safe_text

_SERVICE_NAME = "hermes-agent-runtime.service"
_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def _now(now: int | None = None) -> int:
    return int(time.time() if now is None else now)


def _status_get(status: Mapping[str, Any] | None, key: str, default: str = "") -> str:
    if not status:
        return default
    return str(status.get(key) or status.get(key.lower()) or default)


def _sanitize_extra(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_text(value, limit=500)
    return value


def _alert(severity: str, code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"severity": severity, "code": code, "message": _safe_text(message, limit=500)}
    payload.update({k: _sanitize_extra(v) for k, v in extra.items() if v is not None})
    return payload


def _systemctl_user_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a minimal env that can reach the user systemd bus when possible."""
    base = dict(environ if environ is not None else os.environ)
    if base.get("XDG_RUNTIME_DIR"):
        return base
    try:
        uid = os.getuid()
    except AttributeError:  # pragma: no cover - non-POSIX fallback
        return base
    candidate = Path(f"/run/user/{uid}")
    if candidate.is_dir():
        base["XDG_RUNTIME_DIR"] = str(candidate)
    return base


def probe_runtime_service(service_name: str = _SERVICE_NAME) -> dict[str, str]:
    """Probe the user systemd runtime service with fixed argv only.

    The probe is read-only.  It returns a normalized dict and never raises for a
    missing user bus/systemd; callers convert that into an alert.
    """
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                service_name,
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "NRestarts",
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env=_systemctl_user_env(),
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"ActiveState": "unknown", "SubState": "unknown", "NRestarts": "0", "probe_error": str(exc)}
    payload: dict[str, str] = {"ActiveState": "unknown", "SubState": "unknown", "NRestarts": "0"}
    for line in (result.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"ActiveState", "SubState", "NRestarts"}:
            payload[key] = value.strip()
    if result.returncode != 0:
        payload["probe_error"] = (result.stderr or result.stdout or "systemctl --user show failed").strip()
    return payload


def _service_snapshot(service_status: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    active_state = _status_get(service_status, "ActiveState", "unknown")
    sub_state = _status_get(service_status, "SubState", "unknown")
    try:
        restarts = int(_status_get(service_status, "NRestarts", "0") or 0)
    except ValueError:
        restarts = 0
    service = {
        "name": _SERVICE_NAME,
        "active_state": active_state,
        "sub_state": sub_state,
        "restarts": restarts,
        "probe_error": _status_get(service_status, "probe_error", "") or None,
    }
    alerts: list[dict[str, Any]] = []
    if active_state != "active":
        alerts.append(_alert("critical", "runtime_service_not_active", "Runtime daemon service is not active.", active_state=active_state, sub_state=sub_state))
    if service.get("probe_error"):
        alerts.append(_alert("warning", "runtime_service_probe_failed", "Runtime daemon service probe failed.", detail=service["probe_error"]))
    if restarts > 0:
        alerts.append(_alert("warning", "runtime_service_restarted", "Runtime daemon service has restarted.", restarts=restarts))
    return service, alerts


def _db_alerts(conn: sqlite3.Connection, *, now: int, stale_lease_seconds: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = db.doctor_status(conn)
    alerts: list[dict[str, Any]] = []
    leased_rows = conn.execute(
        """
        SELECT id, run_id, status, lease_owner, lease_expires_at, heartbeat_at
        FROM runtime_jobs
        WHERE status IN ('leased', 'running')
        ORDER BY lease_expires_at, id
        """
    ).fetchall()
    for row in leased_rows:
        lease_expires_at = row["lease_expires_at"]
        heartbeat_at = row["heartbeat_at"]
        if lease_expires_at is not None and int(lease_expires_at) <= now:
            alerts.append(
                _alert(
                    "critical",
                    "runtime_job_lease_expired",
                    "A Runtime job lease is expired and needs recovery.",
                    job_id=row["id"],
                    run_id=row["run_id"],
                    lease_owner=_safe_text(row["lease_owner"], limit=120),
                    lease_expires_at=lease_expires_at,
                )
            )
        elif heartbeat_at is not None and now - int(heartbeat_at) > stale_lease_seconds:
            alerts.append(
                _alert(
                    "warning",
                    "runtime_job_heartbeat_stale",
                    "A Runtime job heartbeat is stale.",
                    job_id=row["id"],
                    run_id=row["run_id"],
                    lease_owner=_safe_text(row["lease_owner"], limit=120),
                    heartbeat_at=heartbeat_at,
                )
            )
    open_high = int(
        conn.execute(
            "SELECT COUNT(*) FROM runtime_findings WHERE status='open' AND severity IN ('high', 'critical')"
        ).fetchone()[0]
    )
    if open_high:
        alerts.append(_alert("warning", "runtime_open_high_findings", "Runtime has open high/critical findings.", count=open_high))
    return counts, alerts


def _overall_status(alerts: list[dict[str, Any]]) -> str:
    if not alerts:
        return "ok"
    max_rank = max(_SEVERITY_RANK.get(str(alert.get("severity")), 0) for alert in alerts)
    return "critical" if max_rank >= 2 else "warning"


def build_health_snapshot(
    conn: sqlite3.Connection | None,
    *,
    service_status: Mapping[str, Any] | None = None,
    now: int | None = None,
    stale_lease_seconds: int = 900,
    db_missing: bool = False,
) -> dict[str, Any]:
    """Return a read-only health snapshot for alerting/UI consumption."""
    ts = _now(now)
    service, alerts = _service_snapshot(service_status or {})
    counts: dict[str, Any] | None = None
    if conn is None:
        if db_missing:
            alerts.append(_alert("warning", "runtime_db_missing", "Runtime DB does not exist yet."))
    else:
        db_counts, db_alerts = _db_alerts(conn, now=ts, stale_lease_seconds=int(stale_lease_seconds))
        counts = db_counts
        alerts.extend(db_alerts)
    status = _overall_status(alerts)
    return {
        "success": True,
        "status": status,
        "generated_at": ts,
        "service": service,
        "db": counts,
        "alerts": alerts,
        "read_only": True,
    }
