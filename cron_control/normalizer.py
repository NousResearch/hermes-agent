"""Canonical normalization helpers for cron control-plane evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


_VALIDATION_VALUES = {"valid", "stale", "malformed", "unavailable", "conflicting"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _coerce_iso(value: Any, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    text = _coerce_text(value, "")
    if not text:
        return fallback
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except Exception:
        return fallback


def evidence_id_for(
    *,
    kind: str,
    source_ref: str,
    job_id: str,
    execution_id: str,
    value: Any,
) -> str:
    seed = canonical_json(
        {
            "kind": kind,
            "source_ref": source_ref,
            "job_id": job_id,
            "execution_id": execution_id,
            "value": value,
        }
    )
    return "ev_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def normalize_job_row(job: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(job)
    normalized["id"] = _coerce_text(normalized.get("id"), "unknown")
    normalized["name"] = _coerce_text(normalized.get("name"), normalized["id"])
    normalized["prompt"] = _coerce_text(normalized.get("prompt"), "")
    normalized["enabled"] = bool(normalized.get("enabled", True))
    state = _coerce_text(normalized.get("state"), "")
    normalized["state"] = state or ("scheduled" if normalized["enabled"] else "paused")
    normalized["deliver"] = _coerce_text(normalized.get("deliver"), "local")
    normalized["no_agent"] = bool(normalized.get("no_agent", False))
    normalized["provider"] = normalized.get("provider")
    normalized["model"] = normalized.get("model")
    normalized["schedule_display"] = _coerce_text(normalized.get("schedule_display"), "")
    schedule = normalized.get("schedule")
    if isinstance(schedule, dict) and not normalized["schedule_display"]:
        for key in ("display", "value", "expr", "run_at"):
            candidate = _coerce_text(schedule.get(key), "")
            if candidate:
                normalized["schedule_display"] = candidate
                break
    if not normalized["schedule_display"]:
        normalized["schedule_display"] = "?"
    normalized["created_at"] = _coerce_iso(normalized.get("created_at"))
    normalized["next_run_at"] = _coerce_iso(normalized.get("next_run_at"))
    normalized["last_run_at"] = _coerce_iso(normalized.get("last_run_at"))
    normalized["paused_at"] = _coerce_iso(normalized.get("paused_at"))
    return normalized


def normalize_execution_row(execution: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(execution)
    normalized["id"] = _coerce_text(normalized.get("id"), "unknown")
    normalized["job_id"] = _coerce_text(normalized.get("job_id"), "unknown")
    normalized["status"] = _coerce_text(normalized.get("status"), "unknown")
    normalized["claimed_at"] = _coerce_iso(normalized.get("claimed_at"))
    normalized["started_at"] = _coerce_iso(normalized.get("started_at"))
    normalized["finished_at"] = _coerce_iso(normalized.get("finished_at"))
    normalized["error"] = normalized.get("error")
    normalized["delivery_state"] = normalized.get("delivery_state")
    normalized["receipt_path"] = normalized.get("receipt_path")
    return normalized


def normalize_delivery_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(event)
    normalized["event"] = _coerce_text(normalized.get("event"), "unknown")
    normalized["job_id"] = _coerce_text(normalized.get("job_id"), "unknown")
    normalized["execution_id"] = _coerce_text(normalized.get("execution_id"), "unknown")
    normalized["delivery_state"] = normalized.get("delivery_state")
    return normalized


def normalize_provider_assessment(assessment: Any) -> dict[str, Any]:
    if isinstance(assessment, dict):
        normalized = dict(assessment)
    else:
        normalized = {}
    normalized["provider"] = _coerce_text(normalized.get("provider"), "unknown")
    normalized["triggered"] = bool(normalized.get("triggered", False))
    normalized["category"] = _coerce_text(getattr(normalized.get("category"), "value", normalized.get("category")), "other")
    normalized["failure_count"] = int(normalized.get("failure_count", 0))
    normalized["window_minutes"] = int(normalized.get("window_minutes", 0))
    normalized["min_consecutive"] = int(normalized.get("min_consecutive", 0))
    normalized["affected_job_ids"] = [
        _coerce_text(job_id, "")
        for job_id in (normalized.get("affected_job_ids") or [])
        if _coerce_text(job_id, "")
    ]
    normalized["details"] = _coerce_text(normalized.get("details"), "")
    return normalized


def normalize_heartbeat_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(snapshot)
    normalized["ticker_heartbeat_age"] = snapshot.get("ticker_heartbeat_age")
    normalized["ticker_success_age"] = snapshot.get("ticker_success_age")
    normalized["ticker_last_error"] = snapshot.get("ticker_last_error")
    normalized["process_id"] = int(snapshot.get("process_id", 0))
    normalized["process_start_time"] = snapshot.get("process_start_time")
    return normalized


def normalize_state_store_probe(probe: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(probe)
    normalized["path"] = _coerce_text(normalized.get("path"), "")
    normalized["kind"] = _coerce_text(normalized.get("kind"), "state_store")
    normalized["status"] = _coerce_text(normalized.get("status"), "unavailable")
    normalized["detail"] = normalized.get("detail")
    return normalized


def normalize_dead_man_switch_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(snapshot)
    normalized["heartbeat_path"] = _coerce_text(normalized.get("heartbeat_path"), "")
    normalized["heartbeat_status"] = _coerce_text(normalized.get("heartbeat_status"), "unavailable")
    normalized["heartbeat_updated_at"] = _coerce_iso(normalized.get("heartbeat_updated_at"))
    normalized["heartbeat_pid"] = int(normalized.get("heartbeat_pid", 0))
    normalized["dump_path"] = _coerce_text(normalized.get("dump_path"), "")
    normalized["dump_status"] = _coerce_text(normalized.get("dump_status"), "unavailable")
    normalized["dump_detail"] = normalized.get("dump_detail")
    normalized["configured_watchdog_seconds"] = normalized.get("configured_watchdog_seconds")
    return normalized


def build_evidence(
    *,
    incident_id: str,
    job_id: str,
    execution_id: str,
    kind: str,
    source: str,
    value: Any,
    source_ref: str,
    source_time: str | None = None,
    observed_at: str | None = None,
    validation: str = "valid",
    freshness_seconds: int = 0,
    content_hash: str | None = None,
    evidence_id: str | None = None,
) -> dict[str, Any]:
    if validation not in _VALIDATION_VALUES:
        raise ValueError(f"unsupported validation value: {validation!r}")
    observed = observed_at or utc_now_iso()
    source_at = source_time or observed
    payload = {
        "evidence_id": evidence_id
        or evidence_id_for(
            kind=kind,
            source_ref=source_ref,
            job_id=job_id,
            execution_id=execution_id,
            value=value,
        ),
        "incident_id": _coerce_text(incident_id, "unknown"),
        "job_id": _coerce_text(job_id, "unknown"),
        "execution_id": _coerce_text(execution_id, "unknown"),
        "kind": _coerce_text(kind, "unknown"),
        "source": _coerce_text(source, "unknown"),
        "observed_at": observed,
        "source_time": source_at,
        "value": value,
        "source_ref": _coerce_text(source_ref, ""),
        "content_hash": content_hash or f"sha256:{sha256_json(value)}",
        "freshness_seconds": max(0, int(freshness_seconds)),
        "validation": validation,
    }
    return payload
