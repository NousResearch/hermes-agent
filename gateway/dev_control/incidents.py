"""Advisory incident detection, rollback recommendations, and postmortems."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gateway.dev_control.ci_status import fetch_ci_status
from gateway.dev_control.production_signals import DevProductionSignalStore
from gateway.dev_control.signal_source import DEFAULT_WINDOW_DAYS, ProductSignalSource, SignalWindow
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_incidents (
    incident_id TEXT PRIMARY KEY,
    detected_at REAL NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    title TEXT NOT NULL,
    correlated_release TEXT NOT NULL,
    evidence_refs TEXT NOT NULL,
    clusters TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    postmortem TEXT NOT NULL,
    proposal_id TEXT,
    warnings TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    acknowledged_at REAL,
    resolved_at REAL
);

CREATE INDEX IF NOT EXISTS idx_dev_incidents_status
    ON dev_incidents(status, detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_incidents_detected_at
    ON dev_incidents(detected_at DESC);
"""

INCIDENT_STATUSES = {"detected", "acknowledged", "mitigated", "resolved"}


class DevIncidentStore:
    """Durable incident records for the advisory recovery loop."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="state.db")
        self._lock = threading.Lock()
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def create_incident(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_incidents (
                    incident_id, detected_at, severity, status, title,
                    correlated_release, evidence_refs, clusters, recommendation,
                    postmortem, proposal_id, warnings, created_at, updated_at,
                    acknowledged_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _incident_values(payload),
            )
        return self.get_incident(payload["incident_id"]) or payload

    def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_incident(incident_id)
        if not current:
            raise KeyError(f"Dev incident not found: {incident_id}")
        merged = {**current, **updates, "updated_at": time.time()}
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE dev_incidents
                SET detected_at = ?, severity = ?, status = ?, title = ?,
                    correlated_release = ?, evidence_refs = ?, clusters = ?,
                    recommendation = ?, postmortem = ?, proposal_id = ?,
                    warnings = ?, created_at = ?, updated_at = ?,
                    acknowledged_at = ?, resolved_at = ?
                WHERE incident_id = ?
                """,
                (*_incident_values(merged)[1:], incident_id),
            )
        return self.get_incident(incident_id) or merged

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_incidents WHERE incident_id = ?",
            (str(incident_id or "").strip(),),
        ).fetchone()
        return _incident_from_row(row) if row else None

    def list_incidents(self, *, status: Optional[str] = None, limit: int = 50) -> list[Dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if status:
            where = "WHERE status = ?"
            params.append(str(status))
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM dev_incidents
            {where}
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_incident_from_row(row) for row in rows]


def detect_incidents(
    *,
    incident_store: DevIncidentStore,
    product_event_store: Any,
    signal_store: Optional[DevProductionSignalStore] = None,
    current_release: Optional[Dict[str, Any]] = None,
    releases: Optional[list[Dict[str, Any]]] = None,
    repo: Optional[str] = None,
    window_days: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    persist: bool = True,
    ci_status_fetcher: Optional[Callable[..., Dict[str, Any]]] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Classify advisory incidents from product signals correlated with a recent release."""

    detected_at = float(now or time.time())
    window = SignalWindow.last_days(window_days or DEFAULT_WINDOW_DAYS, now=detected_at)
    filters = filters or {}
    warnings: list[str] = []
    clusters: list[Dict[str, Any]] = []
    try:
        source_result = ProductSignalSource(product_event_store).fetch_clusters(window, filters=filters)
        clusters = source_result.get("clusters") or []
        warnings.extend(source_result.get("warnings") or [])
    except Exception as exc:
        warnings.append(f"Incident product-signal analysis failed: {exc}")
    current_release = normalize_release(current_release)
    release_window_seconds = _env_float("HERMES_DEV_INCIDENT_RELEASE_WINDOW_HOURS", 48.0) * 3600
    incident_clusters = [
        cluster for cluster in clusters
        if _cluster_severity(cluster) != "none"
        and _correlates_with_release(cluster, current_release, detected_at=detected_at, window_seconds=release_window_seconds)
    ]
    ci_status_fetcher = ci_status_fetcher or fetch_ci_status
    incidents: list[Dict[str, Any]] = []
    for cluster in incident_clusters:
        severity = _cluster_severity(cluster)
        incident = {
            "object": "hermes.dev_incident",
            "incident_id": f"devinc-{uuid.uuid4().hex[:10]}",
            "detected_at": detected_at,
            "severity": severity,
            "status": "detected",
            "title": _incident_title(cluster, current_release),
            "correlated_release": current_release,
            "evidence_refs": cluster.get("evidence_refs") or [],
            "clusters": [cluster],
            "recommendation": rollback_recommendation(
                incident_cluster=cluster,
                current_release=current_release,
                releases=releases or [],
                repo=repo or _env_repo(),
                ci_status_fetcher=ci_status_fetcher,
            ),
            "postmortem": {},
            "proposal_id": None,
            "warnings": warnings.copy(),
            "created_at": detected_at,
            "updated_at": detected_at,
            "acknowledged_at": None,
            "resolved_at": None,
        }
        incidents.append(incident_store.create_incident(incident) if persist else incident)
    return {
        "ok": True,
        "object": "hermes.dev_incident_detection",
        "status": "completed_with_incidents" if incidents else "completed_empty",
        "incidents": incidents,
        "counts": {
            "cluster_count": len(clusters),
            "incident_count": len(incidents),
            "uncorrelated_severe_count": sum(1 for cluster in clusters if _cluster_severity(cluster) != "none") - len(incidents),
        },
        "warnings": warnings,
        "window": {"start": window.start, "end": window.end, "days": window.days},
    }


def acknowledge_incident(*, incident_store: DevIncidentStore, incident_id: str) -> Dict[str, Any]:
    now = time.time()
    return incident_store.update_incident(incident_id, {
        "status": "acknowledged",
        "acknowledged_at": now,
    })


def resolve_incident(
    *,
    incident_store: DevIncidentStore,
    signal_store: DevProductionSignalStore,
    incident_id: str,
    postmortem: Dict[str, Any],
) -> Dict[str, Any]:
    incident = incident_store.get_incident(incident_id)
    if not incident:
        raise KeyError(f"Dev incident not found: {incident_id}")
    normalized = normalize_postmortem(postmortem, incident)
    proposal = _create_postmortem_proposal(signal_store, incident, normalized)
    now = time.time()
    return incident_store.update_incident(incident_id, {
        "status": "resolved",
        "postmortem": normalized,
        "proposal_id": proposal["proposal_id"],
        "resolved_at": now,
    })


def rollback_recommendation(
    *,
    incident_cluster: Dict[str, Any],
    current_release: Dict[str, Any],
    releases: list[Dict[str, Any]],
    repo: str,
    ci_status_fetcher: Callable[..., Dict[str, Any]] = fetch_ci_status,
) -> Dict[str, Any]:
    """Build a manual rollback recommendation. This function never executes rollback actions."""

    current_commit = str(current_release.get("commit") or current_release.get("commit_sha") or "").strip()
    prior = _prior_releases(current_release, releases)
    checked: list[Dict[str, Any]] = []
    for release in prior:
        commit = str(release.get("commit") or release.get("commit_sha") or "").strip()
        if not commit or (current_commit and commit == current_commit):
            continue
        ci = ci_status_fetcher(repo=repo, ref=commit) if repo else {"state": "unknown", "warnings": ["repo unavailable"]}
        candidate = {**release, "ci_state": ci.get("state"), "ci_warnings": ci.get("warnings") or []}
        checked.append(candidate)
        if ci.get("state") == "success":
            return {
                "available": True,
                "target_version": release.get("version"),
                "target_commit": commit,
                "target_tag": release.get("tag") or _tag_for_version(release.get("version")),
                "target_ci_state": "success",
                "checked_releases": checked,
                "runbook_steps": _rollback_runbook(release),
                "rationale": _recommendation_rationale(incident_cluster, current_release, release),
                "manual_only": True,
            }
    return {
        "available": False,
        "target_version": None,
        "target_commit": None,
        "target_tag": None,
        "target_ci_state": None,
        "checked_releases": checked,
        "runbook_steps": [
            "Do not republish a rollback manifest until a prior CI-green release is identified.",
            "Inspect the release history and CI checks manually.",
            "If emergency mitigation is required, coordinate a human-approved fix-forward or manual release action.",
        ],
        "rationale": "No prior stable release with CI success was found; Hermes will not fabricate a rollback target.",
        "manual_only": True,
    }


def normalize_release(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    version = str(value.get("version") or value.get("app_version") or "").strip()
    return {
        "version": version,
        "commit": str(value.get("commit") or value.get("commit_sha") or "").strip(),
        "tag": str(value.get("tag") or _tag_for_version(version) or "").strip(),
        "released_at": _float_or_none(value.get("released_at") or value.get("published_at") or value.get("created_at")),
    }


def normalize_postmortem(value: Dict[str, Any], incident: Dict[str, Any]) -> Dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    keys = (
        "timeline",
        "signal_evidence",
        "suspected_release",
        "action_taken",
        "root_cause_hypothesis",
        "preventive_action",
    )
    postmortem = {key: value.get(key) for key in keys}
    postmortem["incident_id"] = incident.get("incident_id")
    postmortem["created_at"] = time.time()
    if not postmortem.get("signal_evidence"):
        postmortem["signal_evidence"] = incident.get("evidence_refs") or []
    if not postmortem.get("suspected_release"):
        postmortem["suspected_release"] = incident.get("correlated_release") or {}
    return postmortem


def _create_postmortem_proposal(
    signal_store: DevProductionSignalStore,
    incident: Dict[str, Any],
    postmortem: Dict[str, Any],
) -> Dict[str, Any]:
    now = time.time()
    incident_id = incident["incident_id"]
    title = f"Prevent recurrence: {incident.get('title') or incident_id}"
    proposal = {
        "proposal_id": f"devprop-{uuid.uuid4().hex[:10]}",
        "report_id": None,
        "cluster_key": f"incident:{incident_id}",
        "status": "proposed",
        "payload": {
            "title": title,
            "category": "incident_postmortem",
            "priority": "high" if incident.get("severity") == "urgent" else "medium",
            "impact": str(postmortem.get("action_taken") or "Incident resolved; preventive follow-up required."),
            "risk": "Proposal is advisory and must be reviewed before planning.",
            "reason": str(postmortem.get("root_cause_hypothesis") or "Postmortem captured an incident follow-up."),
            "suggested_change": str(postmortem.get("preventive_action") or "Clarify and implement a preventive fix."),
            "source": "incident_postmortem",
            "status": "proposed",
            "incident_id": incident_id,
        },
        "evidence_refs": incident.get("evidence_refs") or [],
        "query_descriptor": {"source": "incident", "incident_id": incident_id},
        "source_window": {"count": 1, "rate_per_day": 0},
        "seeded_clarification_id": None,
        "linked_plan_id": None,
        "outcome": {},
        "created_at": now,
        "updated_at": now,
        "reviewed_at": None,
        "promoted_at": None,
        "measured_at": None,
    }
    return signal_store.create_proposal(proposal)


def _cluster_severity(cluster: Dict[str, Any]) -> str:
    metrics = cluster.get("metrics") if isinstance(cluster.get("metrics"), dict) else {}
    event_type = str(metrics.get("product_event_type") or cluster.get("key") or "").lower()
    count = int(cluster.get("count") or 0)
    if any(part in event_type for part in ("product.crash", "product.unclean_shutdown", "product.uncaught_exception")):
        return "urgent" if count >= _env_int("HERMES_DEV_INCIDENT_CRASH_MIN_COUNT", 1) else "none"
    if count >= _env_int("HERMES_DEV_INCIDENT_ERROR_MIN_COUNT", 3):
        return "warn"
    return "none"


def _correlates_with_release(
    cluster: Dict[str, Any],
    release: Dict[str, Any],
    *,
    detected_at: float,
    window_seconds: float,
) -> bool:
    if not release:
        return False
    version = str(release.get("version") or "").strip()
    metrics = cluster.get("metrics") if isinstance(cluster.get("metrics"), dict) else {}
    versions = {str(item) for item in metrics.get("affected_versions") or []}
    if version and version in versions:
        return True
    released_at = _float_or_none(release.get("released_at"))
    if released_at is None:
        return False
    if detected_at - released_at > window_seconds:
        return False
    window_start = _float_or_none(metrics.get("window_start")) or detected_at
    window_end = _float_or_none(metrics.get("window_end")) or detected_at
    return window_end >= released_at and window_start <= released_at + window_seconds


def _prior_releases(current_release: Dict[str, Any], releases: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    current_version = str(current_release.get("version") or "")
    normalized = [normalize_release(release) for release in releases]
    prior = [release for release in normalized if release.get("version") != current_version]
    return sorted(prior, key=lambda release: float(release.get("released_at") or 0), reverse=True)


def _incident_title(cluster: Dict[str, Any], release: Dict[str, Any]) -> str:
    title = str(cluster.get("title") or "Production incident")
    version = release.get("version")
    return f"{title} after {version}" if version else title


def _recommendation_rationale(cluster: Dict[str, Any], current: Dict[str, Any], target: Dict[str, Any]) -> str:
    return (
        f"Incident signal `{cluster.get('key')}` is correlated with release "
        f"{current.get('version') or current.get('commit')}; {target.get('version') or target.get('commit')} "
        "is the most recent prior stable release with CI success."
    )


def _rollback_runbook(release: Dict[str, Any]) -> list[str]:
    version = release.get("version") or "<target-version>"
    commit = release.get("commit") or "<target-commit>"
    tag = release.get("tag") or _tag_for_version(version)
    return [
        "Confirm the incident, customer impact, and rollback approval with the human operator.",
        f"Inspect target release {version} ({tag}) at commit {commit}.",
        "Verify the target release artifacts and manifest contents manually.",
        f"Manually republish the stable manifest so it points at version {version}, tag {tag}, commit {commit}.",
        "Watch product events and CI after publication; resolve the incident only after mitigation is confirmed.",
    ]


def _tag_for_version(version: Any) -> Optional[str]:
    text = str(version or "").strip()
    return f"oryn-workspace-v{text}" if text else None


def _incident_values(payload: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload["incident_id"],
        float(payload.get("detected_at") or time.time()),
        str(payload.get("severity") or "warn"),
        str(payload.get("status") or "detected"),
        str(payload.get("title") or "Production incident"),
        _json(payload.get("correlated_release") or {}),
        _json(payload.get("evidence_refs") or []),
        _json(payload.get("clusters") or []),
        _json(payload.get("recommendation") or {}),
        _json(payload.get("postmortem") or {}),
        payload.get("proposal_id"),
        _json(payload.get("warnings") or []),
        float(payload.get("created_at") or time.time()),
        float(payload.get("updated_at") or time.time()),
        payload.get("acknowledged_at"),
        payload.get("resolved_at"),
    )


def _incident_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "object": "hermes.dev_incident",
        "incident_id": row["incident_id"],
        "detected_at": row["detected_at"],
        "severity": row["severity"],
        "status": row["status"],
        "title": row["title"],
        "correlated_release": _loads(row["correlated_release"], {}),
        "evidence_refs": _loads(row["evidence_refs"], []),
        "clusters": _loads(row["clusters"], []),
        "recommendation": _loads(row["recommendation"], {}),
        "postmortem": _loads(row["postmortem"], {}),
        "proposal_id": row["proposal_id"],
        "warnings": _loads(row["warnings"], []),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "acknowledged_at": row["acknowledged_at"],
        "resolved_at": row["resolved_at"],
    }


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_repo() -> str:
    return os.getenv("HERMES_DEV_INCIDENT_RELEASE_REPO") or os.getenv("HERMES_DEV_CI_REPO") or "Felippen/Oryn"
