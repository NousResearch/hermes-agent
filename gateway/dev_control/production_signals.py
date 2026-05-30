"""Canonical Hermes back-gate production signal reports and proposals."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from statistics import median
from typing import Any, Dict, Optional

from gateway.dev_control.clarifications import DevClarificationStore, start_clarification
from gateway.dev_control.signal_source import (
    DEFAULT_WINDOW_DAYS,
    DeterministicSignalSource,
    LaminarSignalSource,
    ProductSignalSource,
    ReliabilitySignalSource,
    SignalWindow,
    cluster_rate,
    default_thresholds,
)
from gateway.dev_control.reliability import DevReliabilityStore, measure_category_improvement
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_signal_reports (
    report_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,
    filters TEXT NOT NULL,
    clusters TEXT NOT NULL,
    counts TEXT NOT NULL,
    warnings TEXT NOT NULL,
    health_metrics TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dev_signal_reports_created_at
    ON dev_signal_reports(created_at DESC);

CREATE TABLE IF NOT EXISTS dev_backlog_proposals (
    proposal_id TEXT PRIMARY KEY,
    report_id TEXT,
    cluster_key TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    evidence_refs TEXT NOT NULL,
    query_descriptor TEXT NOT NULL,
    source_window TEXT NOT NULL,
    seeded_clarification_id TEXT,
    linked_plan_id TEXT,
    outcome TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    reviewed_at REAL,
    promoted_at REAL,
    measured_at REAL
);

CREATE INDEX IF NOT EXISTS idx_dev_backlog_proposals_status
    ON dev_backlog_proposals(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_backlog_proposals_cluster
    ON dev_backlog_proposals(cluster_key, created_at DESC);
"""


class DevProductionSignalStore:
    """Durable reports/proposals for production-signal feedback."""

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

    def create_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_signal_reports (
                    report_id, source, status, window_start, window_end, filters,
                    clusters, counts, warnings, health_metrics, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["report_id"],
                    payload["source"],
                    payload["status"],
                    float(payload["window"]["start"]),
                    float(payload["window"]["end"]),
                    _json(payload.get("filters") or {}),
                    _json(payload.get("clusters") or []),
                    _json(payload.get("counts") or {}),
                    _json(payload.get("warnings") or []),
                    _json(payload.get("health_metrics") or {}),
                    float(payload["created_at"]),
                    float(payload["updated_at"]),
                ),
            )
        return self.get_report(payload["report_id"]) or payload

    def update_report(self, report_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_report(report_id)
        if not current:
            raise KeyError(f"Dev signal report not found: {report_id}")
        merged = {**current, **updates, "updated_at": time.time()}
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE dev_signal_reports
                SET source = ?, status = ?, window_start = ?, window_end = ?,
                    filters = ?, clusters = ?, counts = ?, warnings = ?,
                    health_metrics = ?, created_at = ?, updated_at = ?
                WHERE report_id = ?
                """,
                (
                    merged["source"],
                    merged["status"],
                    float(merged["window"]["start"]),
                    float(merged["window"]["end"]),
                    _json(merged.get("filters") or {}),
                    _json(merged.get("clusters") or []),
                    _json(merged.get("counts") or {}),
                    _json(merged.get("warnings") or []),
                    _json(merged.get("health_metrics") or {}),
                    float(merged["created_at"]),
                    float(merged["updated_at"]),
                    report_id,
                ),
            )
        return self.get_report(report_id) or merged

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_signal_reports WHERE report_id = ?",
            (str(report_id or "").strip(),),
        ).fetchone()
        return _report_from_row(row) if row else None

    def list_reports(self, *, limit: int = 50) -> list[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM dev_signal_reports
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(int(limit or 50), 200)),),
        ).fetchall()
        return [_report_from_row(row) for row in rows]

    def create_proposal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_backlog_proposals (
                    proposal_id, report_id, cluster_key, status, payload,
                    evidence_refs, query_descriptor, source_window,
                    seeded_clarification_id, linked_plan_id, outcome,
                    created_at, updated_at, reviewed_at, promoted_at, measured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _proposal_values(payload),
            )
        return self.get_proposal(payload["proposal_id"]) or payload

    def update_proposal(self, proposal_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_proposal(proposal_id)
        if not current:
            raise KeyError(f"Dev backlog proposal not found: {proposal_id}")
        merged = {**current, **updates, "updated_at": time.time()}
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE dev_backlog_proposals
                SET report_id = ?, cluster_key = ?, status = ?, payload = ?,
                    evidence_refs = ?, query_descriptor = ?, source_window = ?,
                    seeded_clarification_id = ?, linked_plan_id = ?, outcome = ?,
                    created_at = ?, updated_at = ?, reviewed_at = ?,
                    promoted_at = ?, measured_at = ?
                WHERE proposal_id = ?
                """,
                (*_proposal_values(merged)[1:], proposal_id),
            )
        return self.get_proposal(proposal_id) or merged

    def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_backlog_proposals WHERE proposal_id = ?",
            (str(proposal_id or "").strip(),),
        ).fetchone()
        return _proposal_from_row(row) if row else None

    def find_proposal_by_cluster(self, cluster_key: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT *
            FROM dev_backlog_proposals
            WHERE cluster_key = ? AND status IN ('proposed', 'approved', 'promoted')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (cluster_key,),
        ).fetchone()
        return _proposal_from_row(row) if row else None

    def list_proposals(self, *, status: Optional[str] = None, limit: int = 50) -> list[Dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if status:
            where = "WHERE status = ?"
            params.append(status)
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM dev_backlog_proposals
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_proposal_from_row(row) for row in rows]


def run_signal_digest(
    *,
    signal_store: DevProductionSignalStore,
    event_store: Any,
    product_event_store: Any = None,
    reliability_store: Any = None,
    execution_store: Any = None,
    source: str = "deterministic",
    window_days: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    return generate_signal_report(
        signal_store=signal_store,
        event_store=event_store,
        product_event_store=product_event_store,
        reliability_store=reliability_store,
        source=source,
        window_days=window_days,
        filters=filters,
        persist=persist,
        create_proposals=True,
    )


def run_signal_digest_sources(
    *,
    signal_store: DevProductionSignalStore,
    event_store: Any,
    product_event_store: Any = None,
    reliability_store: Any = None,
    execution_store: Any = None,
    sources: Optional[list[str]] = None,
    window_days: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    selected_sources = [
        source.strip().lower()
        for source in (sources or ["deterministic", "product", "reliability"])
        if str(source or "").strip()
    ]
    reports = [
        run_signal_digest(
            signal_store=signal_store,
            event_store=event_store,
            product_event_store=product_event_store,
            reliability_store=reliability_store,
            execution_store=execution_store,
            source=source,
            window_days=window_days,
            filters=filters,
            persist=persist,
        )
        for source in selected_sources
    ]
    measurement_sweep = sweep_reliability_proposal_outcomes(
        signal_store=signal_store,
        reliability_store=reliability_store,
        execution_store=execution_store,
        window_days=window_days,
    )
    reliability_reports = [report for report in reports if report.get("source") == "reliability"]
    reliability_proposals = [
        proposal
        for report in reliability_reports
        for proposal in report.get("proposals") or []
    ]
    return {
        "ok": all(report.get("ok", False) for report in reports) and bool(measurement_sweep.get("ok")),
        "object": "hermes.dev_signal_digest_summary",
        "sources": selected_sources,
        "reports": reports,
        "measurement_sweep": measurement_sweep,
        "summary": {
            "report_count": len(reports),
            "cluster_count": sum(int((report.get("counts") or {}).get("cluster_count") or 0) for report in reports),
            "proposal_count": sum(int((report.get("counts") or {}).get("proposal_count") or 0) for report in reports),
            "reliability_proposal_count": len(reliability_proposals),
            "weakest_categories_targeted": [
                _reliability_category_from_proposal(proposal)
                for proposal in reliability_proposals
                if _reliability_category_from_proposal(proposal)
            ],
            "reliability_measurements": [
                {
                    "proposal_id": proposal.get("proposal_id"),
                    "category": (proposal.get("outcome") or {}).get("category"),
                    "before_score": (proposal.get("outcome") or {}).get("before_score"),
                    "after_score": (proposal.get("outcome") or {}).get("after_score"),
                    "status": (proposal.get("outcome") or {}).get("status"),
                    "before_sample_count": (proposal.get("outcome") or {}).get("before_sample_count"),
                    "after_sample_count": (proposal.get("outcome") or {}).get("after_sample_count"),
                }
                for proposal in measurement_sweep.get("measured") or []
            ],
        },
        "advisory_only": True,
    }


def generate_signal_report(
    *,
    signal_store: DevProductionSignalStore,
    event_store: Any,
    product_event_store: Any = None,
    reliability_store: Any = None,
    source: str = "deterministic",
    window_days: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    persist: bool = True,
    create_proposals: bool = True,
) -> Dict[str, Any]:
    created_at = time.time()
    window = SignalWindow.last_days(window_days, now=created_at)
    filters = filters or {}
    warnings: list[str] = []
    try:
        source_impl = _source_impl(
            source,
            event_store=event_store,
            product_event_store=product_event_store,
            reliability_store=reliability_store,
        )
        source_result = source_impl.fetch_clusters(window, filters=filters)
        clusters = source_result.get("clusters") or []
        warnings.extend(source_result.get("warnings") or [])
        status = "completed_with_clusters" if clusters else "completed_empty"
    except Exception as exc:
        clusters = []
        status = "analysis_failed"
        warnings.append(f"Signal analysis failed: {exc}")
        source_result = {"source": source, "analyzed_event_count": 0}
    report = {
        "ok": status != "analysis_failed",
        "object": "hermes.dev_signal_report",
        "report_id": f"devsig-{uuid.uuid4().hex[:10]}",
        "source": source_result.get("source") or source,
        "status": status,
        "window": {"start": window.start, "end": window.end, "days": window.days},
        "filters": filters,
        "clusters": clusters,
        "counts": {
            "cluster_count": len(clusters),
            "analyzed_event_count": int(source_result.get("analyzed_event_count") or 0),
            "proposal_count": 0,
        },
        "warnings": warnings,
        "health_metrics": {},
        "created_at": created_at,
        "updated_at": created_at,
        "proposals": [],
    }
    if persist:
        report = signal_store.create_report(report)
    proposals = []
    if create_proposals and status != "analysis_failed":
        proposals = [_create_or_reuse_proposal(signal_store, report, cluster, window) for cluster in clusters]
        report["proposals"] = proposals
        report["counts"]["proposal_count"] = len(proposals)
    report["health_metrics"] = signal_health(signal_store=signal_store, event_store=event_store)
    if persist:
        report = signal_store.update_report(report["report_id"], {
            "counts": report["counts"],
            "health_metrics": report["health_metrics"],
        })
        report["proposals"] = proposals
    return report


def list_signal_reports(*, signal_store: DevProductionSignalStore, limit: int = 50) -> Dict[str, Any]:
    data = signal_store.list_reports(limit=limit)
    return {"ok": True, "object": "list", "data": data, "total": len(data)}


def list_backlog_proposals(*, signal_store: DevProductionSignalStore, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    data = signal_store.list_proposals(status=status, limit=limit)
    return {"ok": True, "object": "list", "data": data, "total": len(data)}


def transition_backlog_proposal(
    *,
    signal_store: DevProductionSignalStore,
    clarification_store: Optional[DevClarificationStore],
    proposal_id: str,
    action: str,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    proposal = signal_store.get_proposal(proposal_id)
    if not proposal:
        raise KeyError(f"Dev backlog proposal not found: {proposal_id}")
    now = time.time()
    if action in {"approve", "dismiss"}:
        status = "approved" if action == "approve" else "dismissed"
        payload = {**proposal.get("payload", {}), "status": status}
        return signal_store.update_proposal(proposal_id, {
            "status": status,
            "payload": payload,
            "reviewed_at": now,
        })
    if action != "promote":
        raise ValueError(f"Unsupported proposal action: {action}")
    if clarification_store is None:
        raise ValueError("Clarification store is required to promote a proposal.")
    clarification = start_clarification(
        store=clarification_store,
        vision_brief=_promotion_brief(proposal),
        project_id=project_id,
        project_context={
            "project_name": project_id or "Oryn Workspace",
            "work_items": [proposal.get("payload", {}).get("title") or proposal.get("cluster_key")],
            "production_signal": {
                "proposal_id": proposal_id,
                "cluster_key": proposal.get("cluster_key"),
                "evidence_refs": proposal.get("evidence_refs") or [],
            },
        },
        max_questions=3,
    )
    payload = {**proposal.get("payload", {}), "status": "promoted"}
    return signal_store.update_proposal(proposal_id, {
        "status": "promoted",
        "payload": payload,
        "reviewed_at": proposal.get("reviewed_at") or now,
        "promoted_at": now,
        "seeded_clarification_id": clarification["clarification_id"],
    })


def measure_proposal_outcome(
    *,
    signal_store: DevProductionSignalStore,
    event_store: Any,
    product_event_store: Any = None,
    reliability_store: Any = None,
    proposal_id: str,
    window_days: Optional[float] = None,
    source: str = "deterministic",
) -> Dict[str, Any]:
    proposal = signal_store.get_proposal(proposal_id)
    if not proposal:
        raise KeyError(f"Dev backlog proposal not found: {proposal_id}")
    if _proposal_source(proposal) == "reliability" or str(source or "").lower() == "reliability":
        return measure_reliability_proposal_outcome(
            signal_store=signal_store,
            reliability_store=reliability_store,
            proposal_id=proposal_id,
            window_days=window_days,
        )
    now = time.time()
    after_window = SignalWindow.last_days(window_days or (proposal.get("source_window") or {}).get("days") or DEFAULT_WINDOW_DAYS, now=now)
    source_result = _source_impl(
        source,
        event_store=event_store,
        product_event_store=product_event_store,
        reliability_store=reliability_store,
    ).fetch_clusters(after_window, filters={})
    cluster_key = proposal.get("cluster_key")
    after_cluster = next((item for item in source_result.get("clusters") or [] if item.get("key") == cluster_key), None)
    source_window = proposal.get("source_window") or {}
    before_rate = float(source_window.get("rate_per_day") or cluster_rate({"count": source_window.get("count") or 0}, SignalWindow(
        start=float(source_window.get("start") or now - DEFAULT_WINDOW_DAYS * 86400),
        end=float(source_window.get("end") or now),
    )))
    after_rate = cluster_rate(after_cluster, after_window)
    outcome = {
        "before_rate": round(before_rate, 4),
        "after_rate": round(after_rate, 4),
        "before_count": int(source_window.get("count") or 0),
        "after_count": int((after_cluster or {}).get("count") or 0),
        "window": {"start": after_window.start, "end": after_window.end, "days": after_window.days},
        "warnings": source_result.get("warnings") or [],
        "status": _outcome_status(before_rate, after_rate),
        "measured_at": now,
    }
    return signal_store.update_proposal(proposal_id, {
        "outcome": outcome,
        "measured_at": now,
    })


def measure_reliability_proposal_outcome(
    *,
    signal_store: DevProductionSignalStore,
    reliability_store: Any = None,
    proposal_id: str,
    window_days: Optional[float] = None,
) -> Dict[str, Any]:
    proposal = signal_store.get_proposal(proposal_id)
    if not proposal:
        raise KeyError(f"Dev backlog proposal not found: {proposal_id}")
    reliability_store = reliability_store or DevReliabilityStore(signal_store.db_path)
    category = _reliability_category_from_proposal(proposal)
    if not category:
        raise ValueError("Reliability proposal does not include a category.")
    now = time.time()
    source_window = proposal.get("source_window") or {}
    before_start = float(source_window.get("start") or now - (window_days or DEFAULT_WINDOW_DAYS) * 86400)
    before_end = float(source_window.get("end") or now)
    days = float(window_days or source_window.get("days") or DEFAULT_WINDOW_DAYS)
    after_end = now
    after_start = after_end - max(days, 0.1) * 86400
    measurement = measure_category_improvement(
        store=reliability_store,
        category=category,
        before_start=before_start,
        before_end=before_end,
        after_start=after_start,
        after_end=after_end,
        proposal_id=proposal_id,
        plan_id=proposal.get("linked_plan_id"),
    )
    before_score = measurement.get("before_score")
    after_score = measurement.get("after_score")
    outcome = {
        "before_rate": before_score,
        "after_rate": after_score,
        "before_score": before_score,
        "after_score": after_score,
        "before_count": int(measurement.get("before_sample_count") or 0),
        "after_count": int(measurement.get("after_sample_count") or 0),
        "before_sample_count": int(measurement.get("before_sample_count") or 0),
        "after_sample_count": int(measurement.get("after_sample_count") or 0),
        "window": measurement.get("after_window") or {},
        "before_window": measurement.get("before_window") or {},
        "after_window": measurement.get("after_window") or {},
        "warnings": [
            *(measurement.get("warnings") or []),
            "Before/after score movement is correlation evidence, not causal proof.",
        ],
        "status": _score_outcome_status(before_score, after_score),
        "measured_at": measurement.get("measured_at") or now,
        "measurement_id": measurement.get("measurement_id"),
        "category": category,
    }
    return signal_store.update_proposal(proposal_id, {
        "outcome": outcome,
        "measured_at": outcome["measured_at"],
    })


def sweep_reliability_proposal_outcomes(
    *,
    signal_store: DevProductionSignalStore,
    reliability_store: Any = None,
    execution_store: Any = None,
    window_days: Optional[float] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    reliability_store = reliability_store or DevReliabilityStore(signal_store.db_path)
    proposals = [
        proposal for proposal in signal_store.list_proposals(limit=limit)
        if proposal.get("status") == "promoted"
        and _proposal_source(proposal) == "reliability"
        and not (proposal.get("outcome") or {}).get("measured_at")
    ]
    measured: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []
    for proposal in proposals:
        linked_plan_id = proposal.get("linked_plan_id")
        if linked_plan_id and execution_store is not None and not _linked_plan_is_terminal(execution_store, linked_plan_id):
            skipped.append({
                "proposal_id": proposal.get("proposal_id"),
                "reason": "linked plan is not terminal",
                "linked_plan_id": linked_plan_id,
            })
            continue
        if not linked_plan_id:
            skipped.append({
                "proposal_id": proposal.get("proposal_id"),
                "reason": "missing linked_plan_id",
            })
            continue
        measured.append(measure_reliability_proposal_outcome(
            signal_store=signal_store,
            reliability_store=reliability_store,
            proposal_id=proposal["proposal_id"],
            window_days=window_days,
        ))
    return {
        "ok": True,
        "object": "hermes.dev_reliability_outcome_sweep",
        "measured": measured,
        "skipped": skipped,
        "measured_count": len(measured),
        "skipped_count": len(skipped),
    }


def signal_health(*, signal_store: DevProductionSignalStore, event_store: Any = None) -> Dict[str, Any]:
    reports = signal_store.list_reports(limit=50)
    proposals = signal_store.list_proposals(limit=200)
    latest = reports[0] if reports else None
    status_counts = _count_by(proposals, "status")
    now = time.time()
    aging_days = int(default_thresholds()["proposal_aging_days"])
    aging = [
        proposal for proposal in proposals
        if proposal.get("status") in {"proposed", "approved"}
        and now - float(proposal.get("created_at") or now) > aging_days * 86400
    ]
    reviewed_durations = [
        float(proposal.get("reviewed_at")) - float(proposal.get("created_at"))
        for proposal in proposals
        if proposal.get("reviewed_at") and proposal.get("created_at")
    ]
    promoted = [proposal for proposal in proposals if proposal.get("status") == "promoted"]
    measured = [proposal for proposal in promoted if (proposal.get("outcome") or {}).get("measured_at")]
    improved = [proposal for proposal in measured if (proposal.get("outcome") or {}).get("status") == "improved"]
    regressed = [proposal for proposal in measured if (proposal.get("outcome") or {}).get("status") == "regressed"]
    no_change = [proposal for proposal in measured if (proposal.get("outcome") or {}).get("status") == "no_change"]
    analyzed = sum(int((report.get("counts") or {}).get("analyzed_event_count") or 0) for report in reports)
    clusters = sum(int((report.get("counts") or {}).get("cluster_count") or 0) for report in reports)
    return {
        "object": "hermes.dev_signal_health",
        "last_analyzed_at": latest.get("created_at") if latest else None,
        "last_analysis_status": latest.get("status") if latest else "never_run",
        "coverage": {
            "analyzed_event_count": analyzed,
            "cluster_count": clusters,
            "latest_window": latest.get("window") if latest else None,
        },
        "conversion_rate": round((len(proposals) / clusters), 3) if clusters else 0.0,
        "proposals_by_status": status_counts,
        "open_proposal_count": int(status_counts.get("proposed") or 0) + int(status_counts.get("approved") or 0),
        "aging_proposal_count": len(aging),
        "median_time_to_review_seconds": round(median(reviewed_durations), 3) if reviewed_durations else None,
        "outcome_coverage": {
            "promoted": len(promoted),
            "awaiting_measurement": max(len(promoted) - len(measured), 0),
            "measured": len(measured),
            "improved": len(improved),
            "no_change": len(no_change),
            "regressed": len(regressed),
        },
    }


def _source_impl(source: str, *, event_store: Any, product_event_store: Any = None, reliability_store: Any = None) -> Any:
    normalized = str(source or "").lower()
    if normalized == "laminar":
        return LaminarSignalSource()
    if normalized == "product":
        if product_event_store is None:
            from gateway.dev_control.product_events import DevProductEventStore
            db_path = getattr(event_store, "db_path", None)
            product_event_store = DevProductEventStore(db_path)
        return ProductSignalSource(product_event_store)
    if normalized == "reliability":
        if reliability_store is None:
            db_path = getattr(event_store, "db_path", None)
            reliability_store = DevReliabilityStore(db_path)
        return ReliabilitySignalSource(reliability_store)
    return DeterministicSignalSource(event_store)


def _create_or_reuse_proposal(signal_store: DevProductionSignalStore, report: Dict[str, Any], cluster: Dict[str, Any], window: SignalWindow) -> Dict[str, Any]:
    existing = signal_store.find_proposal_by_cluster(cluster["key"])
    if existing:
        return existing
    now = time.time()
    proposal = {
        "proposal_id": f"devprop-{uuid.uuid4().hex[:10]}",
        "report_id": report["report_id"],
        "cluster_key": cluster["key"],
        "status": "proposed",
        "payload": _proposal_payload(cluster),
        "evidence_refs": cluster.get("evidence_refs") or [],
        "query_descriptor": cluster.get("query_descriptor") or {"cluster_key": cluster["key"]},
        "source_window": {
            "start": window.start,
            "end": window.end,
            "days": window.days,
            "count": cluster.get("count") or 0,
            "rate_per_day": cluster.get("rate_per_day") or cluster_rate(cluster, window),
        },
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


def _proposal_payload(cluster: Dict[str, Any]) -> Dict[str, Any]:
    title = cluster.get("title") or cluster.get("key") or "Production signal"
    if str((cluster.get("query_descriptor") or {}).get("source") or "") == "reliability":
        return _reliability_proposal_payload(cluster, title)
    return {
        "title": title,
        "category": "production_signal",
        "priority": _priority_for_cluster(cluster),
        "impact": f"{cluster.get('count', 0)} matching production signal(s) in the analysis window.",
        "risk": "Proposal is advisory; review evidence before creating work.",
        "affected_components": [str(cluster.get("key") or "production_signal")],
        "evidence_refs": cluster.get("evidence_refs") or [],
        "reason": "Hermes observed repeated production signals.",
        "suggested_change": f"Review and clarify a fix for: {title}",
        "non_goals": ["Do not mutate runtime policy or auto-create execution work from this proposal."],
        "source": "production_signal",
        "status": "proposed",
    }


def _reliability_proposal_payload(cluster: Dict[str, Any], title: str) -> Dict[str, Any]:
    metrics = cluster.get("metrics") if isinstance(cluster.get("metrics"), dict) else {}
    descriptor = cluster.get("query_descriptor") if isinstance(cluster.get("query_descriptor"), dict) else {}
    category = str(descriptor.get("category") or str(cluster.get("key") or "").removeprefix("reliability:"))
    dominant = str(metrics.get("dominant_failure_mode") or "low reliability")
    guardrail_touching = _is_guardrail_touching_category(category)
    return {
        "title": title,
        "category": "reliability_improvement",
        "priority": "high" if int(metrics.get("escape_count") or 0) > 0 else "medium",
        "impact": (
            f"{category} is {metrics.get('tier') or 'unproven'} with "
            f"success_rate={metrics.get('success_rate')} across {metrics.get('sample_count') or 0} sample(s)."
        ),
        "risk": (
            "Guardrail-touching proposal; extra human scrutiny required."
            if guardrail_touching
            else "Proposal is advisory; review evidence before creating work."
        ),
        "affected_components": [category],
        "evidence_refs": cluster.get("evidence_refs") or [],
        "reason": f"Reliability scorecard identified {dominant} in {category}.",
        "suggested_change": _reliability_suggested_change(category, dominant),
        "non_goals": [
            "Do not auto-implement this proposal.",
            "Do not grant autonomy or mutate guardrails based on this tier.",
        ],
        "source": "reliability",
        "status": "proposed",
        "guardrail_touching": guardrail_touching,
        "target_category": category,
        "dominant_failure_mode": dominant,
    }


def _priority_for_cluster(cluster: Dict[str, Any]) -> str:
    count = int(cluster.get("count") or 0)
    key = str(cluster.get("key") or "")
    if "failed" in key or count >= 5:
        return "high"
    if "unverifiable" in key or "low_" in key or count >= 3:
        return "medium"
    return "low"


def _promotion_brief(proposal: Dict[str, Any]) -> str:
    payload = proposal.get("payload") or {}
    evidence = proposal.get("evidence_refs") or []
    return "\n".join([
        f"Production signal proposal: {payload.get('title') or proposal.get('cluster_key')}",
        "",
        str(payload.get("reason") or ""),
        str(payload.get("impact") or ""),
        "",
        "Suggested change:",
        str(payload.get("suggested_change") or ""),
        "",
        "Evidence refs:",
        json.dumps(evidence[:10], ensure_ascii=False),
    ]).strip()


def _outcome_status(before_rate: float, after_rate: float) -> str:
    if after_rate < before_rate:
        return "improved"
    if after_rate > before_rate:
        return "regressed"
    return "no_change"


def _score_outcome_status(before_score: Any, after_score: Any) -> str:
    if before_score is None or after_score is None:
        return "needs_more_data"
    before = float(before_score)
    after = float(after_score)
    if after > before:
        return "improved"
    if after < before:
        return "regressed"
    return "no_change"


def _proposal_source(proposal: Dict[str, Any]) -> str:
    payload = proposal.get("payload") if isinstance(proposal.get("payload"), dict) else {}
    return str(payload.get("source") or "").strip().lower()


def _reliability_category_from_proposal(proposal: Dict[str, Any]) -> str:
    payload = proposal.get("payload") if isinstance(proposal.get("payload"), dict) else {}
    descriptor = proposal.get("query_descriptor") if isinstance(proposal.get("query_descriptor"), dict) else {}
    cluster_key = str(proposal.get("cluster_key") or "")
    return str(
        payload.get("target_category")
        or descriptor.get("category")
        or (cluster_key.removeprefix("reliability:") if cluster_key.startswith("reliability:") else "")
    ).strip()


def _linked_plan_is_terminal(execution_store: Any, plan_id: str) -> bool:
    try:
        plan = execution_store.get_plan(plan_id)
    except Exception:
        plan = None
    if not plan:
        return False
    status = str(plan.get("status") or "").lower()
    if status in {"completed", "merged", "shipped", "failed", "cancelled", "needs_attention"}:
        return True
    tasks = plan.get("tasks") or []
    if tasks:
        return all(str(task.get("status") or "").lower() in {"completed", "failed", "cancelled", "needs_attention"} for task in tasks)
    return False


def _is_guardrail_touching_category(category: str) -> bool:
    lowered = category.lower()
    guardrail_terms = ("verify", "verification", "test", "ci", "review", "merge", "gate", "policy")
    return any(term in lowered for term in guardrail_terms)


def _reliability_suggested_change(category: str, dominant_failure_mode: str) -> str:
    if "verification" in dominant_failure_mode:
        return f"Improve the verification path for {category}: tighten emitted acceptance criteria, remove flaky checks, or fix the failing verifier setup."
    if "ci" in dominant_failure_mode:
        return f"Improve CI readiness for {category}: identify recurring red checks and fix the underlying build/test instability."
    if "review" in dominant_failure_mode:
        return f"Improve review readiness for {category}: address recurring code-review findings before merge readiness."
    if "escaped" in dominant_failure_mode:
        return f"Analyze escaped incidents for {category} and add a prevention check before similar work ships again."
    return f"Clarify and implement a reliability improvement for {category}, using the attached failing outcomes as evidence."


def _count_by(items: list[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _report_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "ok": row["status"] != "analysis_failed",
        "object": "hermes.dev_signal_report",
        "report_id": row["report_id"],
        "source": row["source"],
        "status": row["status"],
        "window": {"start": row["window_start"], "end": row["window_end"], "days": max((row["window_end"] - row["window_start"]) / 86400, 0.1)},
        "filters": _loads(row["filters"], {}),
        "clusters": _loads(row["clusters"], []),
        "counts": _loads(row["counts"], {}),
        "warnings": _loads(row["warnings"], []),
        "health_metrics": _loads(row["health_metrics"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _proposal_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "object": "hermes.dev_backlog_proposal",
        "proposal_id": row["proposal_id"],
        "report_id": row["report_id"],
        "cluster_key": row["cluster_key"],
        "status": row["status"],
        "payload": _loads(row["payload"], {}),
        "evidence_refs": _loads(row["evidence_refs"], []),
        "query_descriptor": _loads(row["query_descriptor"], {}),
        "source_window": _loads(row["source_window"], {}),
        "seeded_clarification_id": row["seeded_clarification_id"],
        "linked_plan_id": row["linked_plan_id"],
        "outcome": _loads(row["outcome"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "reviewed_at": row["reviewed_at"],
        "promoted_at": row["promoted_at"],
        "measured_at": row["measured_at"],
    }


def _proposal_values(payload: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload["proposal_id"],
        payload.get("report_id"),
        payload["cluster_key"],
        payload["status"],
        _json(payload.get("payload") or {}),
        _json(payload.get("evidence_refs") or []),
        _json(payload.get("query_descriptor") or {}),
        _json(payload.get("source_window") or {}),
        payload.get("seeded_clarification_id"),
        payload.get("linked_plan_id"),
        _json(payload.get("outcome") or {}),
        payload.get("created_at"),
        payload.get("updated_at"),
        payload.get("reviewed_at"),
        payload.get("promoted_at"),
        payload.get("measured_at"),
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value or _json(default))
    except Exception:
        return default
