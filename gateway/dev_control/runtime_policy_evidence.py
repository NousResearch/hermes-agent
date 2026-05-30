"""Benchmark-backed evidence for guarded runtime selection."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_state import DEFAULT_DB_PATH


MIN_COMPARABLE_RUNTIME_SAMPLES = 6
OPENHANDS_MAX_DELIVERY_FAILURE_RATE = 0.15
OPENHANDS_MIN_MARKER_PASS_RATE = 0.80
OPENHANDS_MIN_REQUIRED_EVIDENCE_PASS_RATE = 0.80
OPENHANDS_MAX_MEDIAN_SCORE_LAG = 0.10


def latest_runtime_policy_evidence(
    db_path: Optional[Path] = None,
    *,
    min_samples: int = MIN_COMPARABLE_RUNTIME_SAMPLES,
) -> Dict[str, Any]:
    """Return the latest live AO/OpenHands benchmark evidence for runtime policy."""

    db_path = db_path or DEFAULT_DB_PATH
    payload = _latest_live_benchmark_payload(db_path)
    if not payload:
        return {
            "status": "insufficient_evidence",
            "reason": "No live AO/OpenHands benchmark evidence is available.",
            "min_sample_count": int(min_samples),
            "benchmark_run_id": None,
            "runtimes": {},
        }

    runtime_results = {
        str(item.get("runtime") or ""): _runtime_health_fields(item)
        for item in payload.get("runtime_results") or []
        if item.get("runtime") in {"ao", "openhands"}
    }
    ao = runtime_results.get("ao")
    openhands = runtime_results.get("openhands")
    if not ao or not openhands:
        return {
            "status": "insufficient_evidence",
            "reason": "Latest live benchmark does not include both AO and OpenHands.",
            "min_sample_count": int(min_samples),
            "benchmark_run_id": payload.get("benchmark_run_id"),
            "runtimes": runtime_results,
        }

    if ao["sample_count"] < min_samples or openhands["sample_count"] < min_samples:
        return {
            "status": "insufficient_evidence",
            "reason": f"Latest live benchmark has fewer than {min_samples} comparable samples per runtime.",
            "min_sample_count": int(min_samples),
            "benchmark_run_id": payload.get("benchmark_run_id"),
            "runtimes": runtime_results,
        }

    degraded_reason = _openhands_degraded_reason(ao, openhands)
    if degraded_reason:
        return {
            "status": "degraded",
            "reason": degraded_reason,
            "min_sample_count": int(min_samples),
            "benchmark_run_id": payload.get("benchmark_run_id"),
            "runtimes": runtime_results,
        }

    return {
        "status": "healthy",
        "reason": "Latest live benchmark evidence keeps OpenHands eligible for read-only inspection.",
        "min_sample_count": int(min_samples),
        "benchmark_run_id": payload.get("benchmark_run_id"),
        "runtimes": runtime_results,
    }


def _latest_live_benchmark_payload(db_path: Path) -> Optional[Dict[str, Any]]:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'dev_harness_benchmark_runs'"
        ).fetchone()
        if not exists:
            return None
        rows = conn.execute(
            """
            SELECT payload
            FROM dev_harness_benchmark_runs
            WHERE live = 1 AND mode = 'live'
            ORDER BY completed_at DESC, created_at DESC
            LIMIT 10
            """
        ).fetchall()
    finally:
        conn.close()
    for row in rows:
        try:
            payload = json.loads(row["payload"] or "{}")
        except Exception:
            continue
        runtimes = {str(item.get("runtime") or "") for item in payload.get("runtime_results") or []}
        if {"ao", "openhands"}.issubset(runtimes):
            return payload
    return None


def _runtime_health_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "runtime": item.get("runtime"),
        "sample_count": int(item.get("case_count") or 0),
        "iteration_count": int(item.get("iteration_count") or 0),
        "median_score": _number(item.get("median_score")),
        "task_quality_score": _number(item.get("median_task_quality_score")),
        "contract_compliance_score": _number(item.get("median_contract_compliance_score")),
        "marker_pass_rate": _number(item.get("marker_pass_rate")),
        "required_evidence_pass_rate": _number(item.get("required_evidence_pass_rate")),
        "delivery_failure_rate": _number(item.get("delivery_failure_rate")),
        "average_duration_seconds": _number(item.get("average_duration_seconds")),
        "token_sample_count": int(item.get("token_sample_count") or 0),
        "cost_sample_count": int(item.get("cost_sample_count") or 0),
        "total_tokens": int(item.get("total_tokens") or 0),
        "total_cost_usd": _number(item.get("total_cost_usd")),
    }


def _openhands_degraded_reason(ao: Dict[str, Any], openhands: Dict[str, Any]) -> Optional[str]:
    if openhands["delivery_failure_rate"] > OPENHANDS_MAX_DELIVERY_FAILURE_RATE:
        return "OpenHands benchmark delivery failure rate is above the guarded threshold."
    if openhands["marker_pass_rate"] < OPENHANDS_MIN_MARKER_PASS_RATE:
        return "OpenHands benchmark marker pass rate is below the guarded threshold."
    if openhands["required_evidence_pass_rate"] < OPENHANDS_MIN_REQUIRED_EVIDENCE_PASS_RATE:
        return "OpenHands benchmark required-evidence pass rate is below the guarded threshold."
    if openhands["median_score"] < ao["median_score"] - OPENHANDS_MAX_MEDIAN_SCORE_LAG:
        return "OpenHands benchmark median score trails AO by more than the guarded threshold."
    return None


def _number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
