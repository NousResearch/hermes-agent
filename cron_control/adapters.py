"""Read-only adapters over the cron runtime artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cron.delivery_watchdog as delivery_watchdog
import cron.provider_recovery as provider_recovery
from hermes_constants import get_hermes_home

from .normalizer import (
    build_evidence,
    normalize_dead_man_switch_snapshot,
    normalize_delivery_event,
    normalize_execution_row,
    normalize_heartbeat_snapshot,
    normalize_job_row,
    normalize_provider_assessment,
    normalize_state_store_probe,
    utc_now_iso,
)


def _cron_paths() -> dict[str, Path]:
    home = get_hermes_home()
    cron_dir = home / "cron"
    import cron.jobs as cron_jobs
    import cron.executions as cron_executions

    return {
        "home": home,
        "cron_dir": cron_dir,
        "jobs_path": Path(cron_jobs.JOBS_FILE),
        "executions_path": Path(cron_executions.EXECUTIONS_FILE),
        "control_plane_path": cron_dir / "control-plane.db",
    }


def read_job_rows(jobs_path: Path | None = None, *, include_disabled: bool = True) -> list[dict[str, Any]]:
    path = jobs_path or _cron_paths()["jobs_path"]
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    jobs = raw.get("jobs", []) if isinstance(raw, dict) else raw
    if not isinstance(jobs, list):
        raise ValueError("jobs registry must contain a list")
    rows = [normalize_job_row(job) for job in jobs if isinstance(job, dict)]
    if not include_disabled:
        rows = [job for job in rows if job.get("enabled", True)]
    return rows


def _read_latest_executions(executions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for execution in executions:
        job_id = str(execution.get("job_id") or "").strip()
        if not job_id:
            continue
        current = latest.get(job_id)
        execution_key = (str(execution.get("claimed_at") or ""), str(execution.get("id") or ""))
        current_key = (
            str(current.get("claimed_at") or ""),
            str(current.get("id") or ""),
        ) if current else None
        if current_key is None or execution_key > current_key:
            latest[job_id] = execution
    return latest


def collect_job_registry_evidence(jobs_path: Path | None = None) -> list[dict[str, Any]]:
    jobs_path = jobs_path or _cron_paths()["jobs_path"]
    try:
        rows = read_job_rows(jobs_path, include_disabled=True)
    except Exception:
        return []
    latest = {}
    try:
        latest = _read_latest_executions(read_execution_rows())
    except Exception:
        latest = {}
    evidence: list[dict[str, Any]] = []
    for job in rows:
        value = dict(job)
        if job["id"] in latest:
            value["latest_execution"] = latest[job["id"]]
        evidence.append(
            build_evidence(
                incident_id=f"job:{job['id']}",
                job_id=job["id"],
                execution_id=str((value.get("latest_execution") or {}).get("id") or ""),
                kind="job_metadata",
                source="jobs.json",
                value=value,
                source_ref=f"jobs.json:{job['id']}",
                source_time=value.get("created_at") or utc_now_iso(),
                validation="valid",
            )
        )
    return evidence


def read_execution_rows(
    executions_path: Path | None = None,
    *,
    job_id: str | None = None,
    limit: int = 500,
    include_incomplete: bool = True,
) -> list[dict[str, Any]]:
    path = executions_path or _cron_paths()["executions_path"]
    if not path.exists():
        return []
    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=1.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            clauses = []
            params: list[Any] = []
            if job_id is not None:
                clauses.append("job_id=?")
                params.append(str(job_id))
            if not include_incomplete:
                clauses.append("status IN ('completed','failed','unknown')")
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            params.append(max(1, min(int(limit), 500)))
            rows = conn.execute(
                "SELECT * FROM executions" + where + " ORDER BY claimed_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
    except sqlite3.DatabaseError:
        return []
    return [normalize_execution_row(dict(row)) for row in rows]


def collect_execution_evidence(
    executions_path: Path | None = None,
    *,
    job_id: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    rows = read_execution_rows(executions_path, job_id=job_id, limit=limit)
    evidence: list[dict[str, Any]] = []
    for row in rows:
        evidence.append(
            build_evidence(
                incident_id=f"exec:{row['id']}",
                job_id=row["job_id"],
                execution_id=row["id"],
                kind="execution_state",
                source="executions.db",
                value=row,
                source_ref=f"executions.db:id={row['id']}",
                source_time=row.get("claimed_at") or utc_now_iso(),
                freshness_seconds=0,
                validation="valid",
            )
        )
    return evidence


def collect_delivery_evidence(
    jobs: list[dict[str, Any]] | None = None,
    executions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if jobs is None:
        jobs = read_job_rows()
    if executions is None:
        executions = read_execution_rows()
    raw_events = delivery_watchdog.classify_delivery_events(jobs, executions)
    evidence: list[dict[str, Any]] = []
    for event in raw_events:
        normalized = normalize_delivery_event(event)
        validation = {
            "missing_delivery_receipt": "stale",
            "delivery_failed": "conflicting",
            "delivery_uncertain": "conflicting",
            "execution_unknown": "unavailable",
            "unresolved_origin": "malformed",
            "job_runtime_error": "valid",
        }.get(normalized["event"], "valid")
        evidence.append(
            build_evidence(
                incident_id=f"delivery:{normalized['job_id']}",
                job_id=normalized["job_id"],
                execution_id=normalized["execution_id"],
                kind="delivery_receipt",
                source="delivery_watchdog",
                value=normalized,
                source_ref=f"delivery_watchdog:{normalized['job_id']}:{normalized['execution_id']}",
                source_time=utc_now_iso(),
                freshness_seconds=0,
                validation=validation,
            )
        )
    return evidence


def collect_provider_evidence(
    jobs: list[dict[str, Any]] | None = None,
    *,
    window_minutes: int = 60,
    min_consecutive: int = 3,
) -> list[dict[str, Any]]:
    if jobs is None:
        jobs = read_job_rows()
    providers = sorted({str(job.get("provider") or "").strip() for job in jobs if str(job.get("provider") or "").strip()})
    evidence: list[dict[str, Any]] = []
    for provider in providers:
        assessment = provider_recovery.scan_provider_failures(
            provider,
            window_minutes=window_minutes,
            min_consecutive=min_consecutive,
        )
        normalized = normalize_provider_assessment(assessment.__dict__)
        evidence.append(
            build_evidence(
                incident_id=f"provider:{provider}",
                job_id="provider",
                execution_id=provider,
                kind="provider_probe",
                source="provider_recovery",
                value=normalized,
                source_ref=f"provider_recovery:{provider}",
                source_time=utc_now_iso(),
                validation="valid",
            )
        )
    return evidence


def collect_heartbeat_evidence() -> list[dict[str, Any]]:
    from cron.jobs import get_ticker_heartbeat_age, get_ticker_last_error, get_ticker_success_age
    from gateway.status import get_process_start_time
    import os

    pid = os.getpid()
    snapshot = normalize_heartbeat_snapshot(
        {
            "ticker_heartbeat_age": get_ticker_heartbeat_age(),
            "ticker_success_age": get_ticker_success_age(),
            "ticker_last_error": get_ticker_last_error(),
            "process_id": pid,
            "process_start_time": get_process_start_time(pid),
        }
    )
    return [
        build_evidence(
            incident_id="heartbeat:ticker",
            job_id="ticker",
            execution_id=str(pid),
            kind="scheduler_heartbeat",
            source="cron/jobs.py",
            value=snapshot,
            source_ref="file:ticker_heartbeat",
            source_time=utc_now_iso(),
            validation="valid" if snapshot["ticker_heartbeat_age"] is not None else "unavailable",
            freshness_seconds=int(snapshot["ticker_heartbeat_age"] or 0),
        )
    ]


def _probe_json_file(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        return {"status": "ok", "kind": "json", "detail": f"keys={len(raw)}"}
    if isinstance(raw, list):
        return {"status": "ok", "kind": "json", "detail": f"items={len(raw)}"}
    return {"status": "malformed", "kind": "json", "detail": type(raw).__name__}


def _probe_sqlite_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "unavailable", "kind": "sqlite", "detail": "missing"}
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=1.0) as conn:
        conn.execute("PRAGMA query_only = ON")
        conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
    return {"status": "ok", "kind": "sqlite", "detail": "readable"}


def collect_state_store_evidence(paths: Iterable[Path] | None = None) -> list[dict[str, Any]]:
    if paths is None:
        p = _cron_paths()
        paths = (p["jobs_path"], p["executions_path"], p["control_plane_path"])
    evidence: list[dict[str, Any]] = []
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if path.suffix in {".json", ".yaml", ".yml"}:
            try:
                probe = _probe_json_file(path)
            except Exception as exc:
                probe = {"status": "unavailable", "kind": "json", "detail": f"{type(exc).__name__}"}
        elif path.suffix == ".db":
            try:
                probe = _probe_sqlite_file(path)
            except Exception as exc:
                probe = {"status": "unavailable", "kind": "sqlite", "detail": f"{type(exc).__name__}"}
        else:
            probe = {"status": "unavailable", "kind": "unknown", "detail": "unsupported"}
        normalized = normalize_state_store_probe(
            {
                "path": path.as_posix(),
                "kind": probe["kind"],
                "status": probe["status"],
                "detail": probe.get("detail"),
            }
        )
        validation = {
            "ok": "valid",
            "unavailable": "unavailable",
            "malformed": "malformed",
            "degraded": "stale",
        }.get(normalized["status"], "valid")
        evidence.append(
            build_evidence(
                incident_id=f"state-store:{path.name}",
                job_id=path.stem,
                execution_id=path.name,
                kind="state_store",
                source=normalized["kind"],
                value=normalized,
                source_ref=f"file:{path.as_posix()}",
                source_time=utc_now_iso(),
                freshness_seconds=0,
                validation=validation,
            )
        )
    return evidence


def collect_dead_man_switch_evidence() -> list[dict[str, Any]]:
    from gateway.shutdown_watchdog import get_loop_heartbeat_path, get_shutdown_watchdog_dump_path
    from gateway.systemd_notify import watchdog_interval_seconds

    heartbeat_path = get_loop_heartbeat_path()
    dump_path = get_shutdown_watchdog_dump_path()
    heartbeat_status = "unavailable"
    heartbeat_updated_at = None
    heartbeat_pid = 0
    heartbeat_detail = None
    if heartbeat_path.exists():
        try:
            raw = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            heartbeat_status = "ok"
            heartbeat_updated_at = raw.get("updated_at")
            heartbeat_pid = int(raw.get("pid", 0))
            heartbeat_detail = f"heartbeat present: pid={heartbeat_pid}"
        except Exception as exc:
            heartbeat_status = "malformed"
            heartbeat_detail = type(exc).__name__

    dump_status = "unavailable"
    dump_detail = None
    if dump_path.exists():
        try:
            text = dump_path.read_text(encoding="utf-8")
            first_line = text.splitlines()[0] if text else ""
            dump_status = "tripped"
            dump_detail = first_line or "watchdog dump present"
        except Exception as exc:
            dump_status = "malformed"
            dump_detail = type(exc).__name__

    snapshot = normalize_dead_man_switch_snapshot(
        {
            "heartbeat_path": heartbeat_path.as_posix(),
            "heartbeat_status": heartbeat_status,
            "heartbeat_updated_at": heartbeat_updated_at,
            "heartbeat_pid": heartbeat_pid,
            "dump_path": dump_path.as_posix(),
            "dump_status": dump_status,
            "dump_detail": dump_detail,
            "configured_watchdog_seconds": watchdog_interval_seconds(),
        }
    )
    validation = {
        "ok": "valid",
        "unavailable": "unavailable",
        "malformed": "malformed",
        "tripped": "conflicting",
    }.get(snapshot["heartbeat_status"], "valid")
    if snapshot["dump_status"] == "tripped":
        validation = "conflicting"
    return [
        build_evidence(
            incident_id="dead-man-switch:gateway",
            job_id="gateway",
            execution_id=str(snapshot["heartbeat_pid"] or "gateway"),
            kind="dead_man_switch",
            source="gateway.shutdown_watchdog",
            value=snapshot,
            source_ref=f"heartbeat:{snapshot['heartbeat_path']} dump:{snapshot['dump_path']}",
            source_time=snapshot["heartbeat_updated_at"] or utc_now_iso(),
            freshness_seconds=0,
            validation=validation,
        )
    ]
