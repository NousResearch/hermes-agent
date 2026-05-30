"""Advisory Dev reliability scorecards and trust tiers."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_reliability_outcomes (
    outcome_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    category TEXT NOT NULL,
    profile_id TEXT,
    risk_level TEXT,
    terminal_status TEXT,
    merged INTEGER NOT NULL DEFAULT 0,
    verification_verdict TEXT,
    ci_state TEXT,
    code_review_verdict TEXT,
    output_contract_score REAL,
    rework_count INTEGER NOT NULL DEFAULT 0,
    escaped INTEGER NOT NULL DEFAULT 0,
    escape_refs TEXT NOT NULL,
    source_refs TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL,
    merged_at REAL,
    UNIQUE(plan_id, task_id)
);

CREATE INDEX IF NOT EXISTS idx_dev_reliability_outcomes_category
    ON dev_reliability_outcomes(category, completed_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_reliability_outcomes_completed
    ON dev_reliability_outcomes(completed_at DESC);

CREATE TABLE IF NOT EXISTS dev_reliability_improvements (
    measurement_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    proposal_id TEXT,
    plan_id TEXT,
    before_score REAL,
    after_score REAL,
    before_sample_count INTEGER NOT NULL,
    after_sample_count INTEGER NOT NULL,
    before_window TEXT NOT NULL,
    after_window TEXT NOT NULL,
    measured_at REAL NOT NULL,
    warnings TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dev_reliability_improvements_category
    ON dev_reliability_improvements(category, measured_at DESC);
"""

SUCCESS_VERIFICATION_VERDICTS = {"verified", "passed"}
SUCCESS_CI_STATES = {"success"}
SUCCESS_REVIEW_VERDICTS = {"approved"}
UNMEASURED_GATE_STATES = {"", "unknown", "not_measured", "not-measured", "not measured", "unavailable"}
MERGED_STATUSES = {"merged", "shipped", "completed"}
TERMINAL_STATUSES = MERGED_STATUSES | {"failed", "cancelled", "needs_attention", "needs_review"}
TIER_RANK = {"unproven": 0, "observed": 1, "trusted": 2}
DEFAULT_EXCLUDED_OUTCOME_IDS = {"devrel-out-042c159df0"}


@dataclass(frozen=True)
class ReliabilityConfig:
    window_days: float
    escape_window_days: float
    observed_min_samples: int
    observed_success_rate: float
    trusted_min_samples: int
    trusted_success_rate: float
    trusted_min_window_days: float


def reliability_config_from_env() -> ReliabilityConfig:
    return ReliabilityConfig(
        window_days=_env_float("HERMES_DEV_RELIABILITY_WINDOW_DAYS", 30.0),
        escape_window_days=_env_float("HERMES_DEV_RELIABILITY_ESCAPE_WINDOW_DAYS", 14.0),
        observed_min_samples=_env_int("HERMES_DEV_RELIABILITY_OBSERVED_MIN_SAMPLES", 10),
        observed_success_rate=_env_float("HERMES_DEV_RELIABILITY_OBSERVED_SUCCESS_RATE", 0.90),
        trusted_min_samples=_env_int("HERMES_DEV_RELIABILITY_TRUSTED_MIN_SAMPLES", 30),
        trusted_success_rate=_env_float("HERMES_DEV_RELIABILITY_TRUSTED_SUCCESS_RATE", 0.95),
        trusted_min_window_days=_env_float("HERMES_DEV_RELIABILITY_TRUSTED_MIN_WINDOW_DAYS", 30.0),
    )


class DevReliabilityStore:
    """SQLite persistence for normalized task outcomes and measurements."""

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

    def upsert_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_outcome(outcome)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_reliability_outcomes (
                    outcome_id, plan_id, task_id, category, profile_id, risk_level,
                    terminal_status, merged, verification_verdict, ci_state,
                    code_review_verdict, output_contract_score, rework_count,
                    escaped, escape_refs, source_refs, payload, created_at,
                    updated_at, completed_at, merged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id, task_id) DO UPDATE SET
                    category = excluded.category,
                    profile_id = excluded.profile_id,
                    risk_level = excluded.risk_level,
                    terminal_status = excluded.terminal_status,
                    merged = excluded.merged,
                    verification_verdict = excluded.verification_verdict,
                    ci_state = excluded.ci_state,
                    code_review_verdict = excluded.code_review_verdict,
                    output_contract_score = excluded.output_contract_score,
                    rework_count = excluded.rework_count,
                    escaped = excluded.escaped,
                    escape_refs = excluded.escape_refs,
                    source_refs = excluded.source_refs,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at,
                    completed_at = excluded.completed_at,
                    merged_at = excluded.merged_at
                """,
                _outcome_values(normalized),
            )
        return self.get_outcome(plan_id=normalized["plan_id"], task_id=normalized["task_id"]) or normalized

    def get_outcome(self, *, plan_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT *
            FROM dev_reliability_outcomes
            WHERE plan_id = ? AND task_id = ?
            """,
            (str(plan_id or "").strip(), str(task_id or "").strip()),
        ).fetchone()
        return _outcome_from_row(row) if row else None

    def list_outcomes(
        self,
        *,
        category: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        limit: int = 500,
    ) -> list[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if category:
            clauses.append("category = ?")
            params.append(str(category).strip())
        if start is not None:
            clauses.append("COALESCE(completed_at, updated_at) >= ?")
            params.append(float(start))
        if end is not None:
            clauses.append("COALESCE(completed_at, updated_at) <= ?")
            params.append(float(end))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 500), 5000)))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM dev_reliability_outcomes
            {where}
            ORDER BY COALESCE(completed_at, updated_at) DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_outcome_from_row(row) for row in rows]

    def persist_improvement_measurement(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "measurement_id": str(measurement.get("measurement_id") or f"devrel-measure-{uuid.uuid4().hex[:10]}"),
            "category": str(measurement.get("category") or "").strip(),
            "proposal_id": _optional_text(measurement.get("proposal_id")),
            "plan_id": _optional_text(measurement.get("plan_id")),
            "before_score": _optional_float(measurement.get("before_score")),
            "after_score": _optional_float(measurement.get("after_score")),
            "before_sample_count": int(measurement.get("before_sample_count") or 0),
            "after_sample_count": int(measurement.get("after_sample_count") or 0),
            "before_window": measurement.get("before_window") or {},
            "after_window": measurement.get("after_window") or {},
            "measured_at": float(measurement.get("measured_at") or time.time()),
            "warnings": measurement.get("warnings") or [],
        }
        if not payload["category"]:
            raise ValueError("category is required for reliability measurement")
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_reliability_improvements (
                    measurement_id, category, proposal_id, plan_id, before_score,
                    after_score, before_sample_count, after_sample_count,
                    before_window, after_window, measured_at, warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["measurement_id"],
                    payload["category"],
                    payload["proposal_id"],
                    payload["plan_id"],
                    payload["before_score"],
                    payload["after_score"],
                    payload["before_sample_count"],
                    payload["after_sample_count"],
                    _json(payload["before_window"]),
                    _json(payload["after_window"]),
                    payload["measured_at"],
                    _json(payload["warnings"]),
                ),
            )
        return payload

    def list_improvement_measurements(self, *, category: Optional[str] = None, limit: int = 50) -> list[Dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if category:
            where = "WHERE category = ?"
            params.append(str(category).strip())
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM dev_reliability_improvements
            {where}
            ORDER BY measured_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_measurement_from_row(row) for row in rows]


def normalize_outcome(outcome: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    profile_id = str(outcome.get("profile_id") or "unknown").strip() or "unknown"
    risk_level = normalize_risk_level(outcome.get("risk_level"))
    category = str(outcome.get("category") or f"{profile_id}/{risk_level}").strip()
    completed_at = _optional_float(outcome.get("completed_at")) or _optional_float(outcome.get("updated_at")) or now
    merged_at = _optional_float(outcome.get("merged_at"))
    normalized = {
        "object": "hermes.dev_reliability_outcome",
        "outcome_id": str(outcome.get("outcome_id") or f"devrel-out-{uuid.uuid4().hex[:10]}"),
        "plan_id": str(outcome.get("plan_id") or "").strip(),
        "task_id": str(outcome.get("task_id") or "").strip(),
        "category": category,
        "profile_id": profile_id,
        "risk_level": risk_level,
        "terminal_status": str(outcome.get("terminal_status") or outcome.get("status") or "").strip().lower(),
        "merged": bool(outcome.get("merged")),
        "verification_verdict": _optional_lower(outcome.get("verification_verdict")),
        "ci_state": _optional_lower(outcome.get("ci_state")),
        "code_review_verdict": _optional_lower(outcome.get("code_review_verdict")),
        "output_contract_score": _optional_float(outcome.get("output_contract_score")),
        "rework_count": max(0, int(outcome.get("rework_count") or 0)),
        "escaped": bool(outcome.get("escaped")),
        "escape_refs": outcome.get("escape_refs") or [],
        "source_refs": outcome.get("source_refs") or {},
        "created_at": float(outcome.get("created_at") or now),
        "updated_at": float(outcome.get("updated_at") or now),
        "completed_at": completed_at,
        "merged_at": merged_at,
    }
    if not normalized["plan_id"] or not normalized["task_id"]:
        raise ValueError("plan_id and task_id are required for reliability outcome")
    normalized["success"] = outcome_success(normalized)
    return normalized


def compose_task_outcome(
    *,
    plan: Dict[str, Any],
    task: Dict[str, Any],
    verification: Optional[Dict[str, Any]] = None,
    pr_state: Optional[Dict[str, Any]] = None,
    code_review: Optional[Dict[str, Any]] = None,
    readiness: Optional[Dict[str, Any]] = None,
    incidents: Optional[Iterable[Dict[str, Any]]] = None,
    product_events: Optional[Iterable[Dict[str, Any]]] = None,
    config: Optional[ReliabilityConfig] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    config = config or reliability_config_from_env()
    now_value = float(now or time.time())
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    source_refs = {
        "plan_id": plan.get("plan_id"),
        "task_id": task.get("task_id"),
    }
    profile_id = str(task.get("profile_id") or payload.get("profile_id") or "unknown").strip() or "unknown"
    risk_level = normalize_risk_level(task.get("risk_level") or payload.get("risk_level") or payload.get("risk"))
    terminal_status = str(task.get("status") or plan.get("status") or "").strip().lower()
    verification_verdict = _optional_lower((verification or {}).get("verdict"))
    ci_state = _optional_lower((pr_state or {}).get("ci_state") or ((pr_state or {}).get("ci_status") or {}).get("state"))
    code_review_verdict = _optional_lower((code_review or {}).get("verdict"))
    if not code_review_verdict and readiness:
        code_review_verdict = _optional_lower((readiness.get("code_review") or {}).get("verdict"))
    merged = _merged_from_sources(terminal_status, pr_state=pr_state, readiness=readiness)
    completed_at = _optional_float(task.get("updated_at")) or _optional_float(plan.get("updated_at")) or now_value
    merged_at = _optional_float((pr_state or {}).get("merged_at")) or (completed_at if merged else None)
    head_sha = str((pr_state or {}).get("head_sha") or (readiness or {}).get("head_sha") or payload.get("head_sha") or "").strip()
    escape_refs = correlate_escapes(
        head_sha=head_sha,
        merged_at=merged_at,
        incidents=incidents or [],
        product_events=product_events or [],
        escape_window_days=config.escape_window_days,
    )
    outcome = {
        "plan_id": plan.get("plan_id"),
        "task_id": task.get("task_id"),
        "category": f"{profile_id}/{risk_level}",
        "profile_id": profile_id,
        "risk_level": risk_level,
        "terminal_status": terminal_status,
        "merged": merged,
        "verification_verdict": verification_verdict,
        "ci_state": ci_state,
        "code_review_verdict": code_review_verdict,
        "output_contract_score": _first_float(
            payload.get("output_contract_score"),
            task.get("output_contract_score"),
        ),
        "rework_count": rework_count_from_sources(task=task, code_review=code_review),
        "escaped": bool(escape_refs),
        "escape_refs": escape_refs,
        "source_refs": source_refs,
        "created_at": _optional_float(task.get("created_at")) or _optional_float(plan.get("created_at")) or now_value,
        "updated_at": now_value,
        "completed_at": completed_at,
        "merged_at": merged_at,
    }
    return normalize_outcome(outcome)


def correlate_escapes(
    *,
    head_sha: Optional[str],
    merged_at: Optional[float],
    incidents: Iterable[Dict[str, Any]],
    product_events: Iterable[Dict[str, Any]],
    escape_window_days: float,
) -> list[Dict[str, Any]]:
    sha = str(head_sha or "").strip()
    if not sha or not merged_at:
        return []
    end = float(merged_at) + max(0.0, float(escape_window_days)) * 86400
    refs: list[Dict[str, Any]] = []
    for incident in incidents or []:
        detected_at = _optional_float(incident.get("detected_at") or incident.get("created_at"))
        if detected_at is None or detected_at < float(merged_at) or detected_at > end:
            continue
        release = incident.get("correlated_release") if isinstance(incident.get("correlated_release"), dict) else {}
        if _matches_sha(sha, release.get("commit") or release.get("commit_sha") or release.get("target_commit")):
            refs.append({
                "type": "incident",
                "incident_id": incident.get("incident_id"),
                "detected_at": detected_at,
            })
    for event in product_events or []:
        seen_at = _optional_float(event.get("last_seen_at") or event.get("received_at") or event.get("client_ts"))
        if seen_at is None or seen_at < float(merged_at) or seen_at > end:
            continue
        context = event.get("context") if isinstance(event.get("context"), dict) else {}
        event_sha = context.get("commit") or context.get("commit_sha") or context.get("release_commit")
        if _matches_sha(sha, event_sha):
            refs.append({
                "type": "product_event",
                "event_id": event.get("event_id"),
                "signature": event.get("signature"),
                "seen_at": seen_at,
            })
    return refs


def scorecard(
    outcomes: Iterable[Dict[str, Any]],
    *,
    now: Optional[float] = None,
    config: Optional[ReliabilityConfig] = None,
) -> Dict[str, Any]:
    config = config or reliability_config_from_env()
    now_value = float(now or time.time())
    window_seconds = max(1.0, config.window_days * 86400)
    current_start = now_value - window_seconds
    previous_start = current_start - window_seconds
    normalized = [normalize_outcome(item) for item in outcomes if not outcome_excluded(item)]
    categories = sorted({item["category"] for item in normalized})
    category_rows = []
    for category in categories:
        current = [
            item for item in normalized
            if item["category"] == category and current_start <= _outcome_time(item) <= now_value
        ]
        previous = [
            item for item in normalized
            if item["category"] == category and previous_start <= _outcome_time(item) < current_start
        ]
        if not current:
            continue
        row = category_scorecard(
            category=category,
            outcomes=current,
            previous_outcomes=previous,
            config=config,
            window={"start": current_start, "end": now_value, "days": config.window_days},
        )
        category_rows.append(row)
    category_rows.sort(key=lambda item: (TIER_RANK.get(item["tier"], 0), item["success_rate"] or 0.0, -item["escape_rate"]))
    return {
        "ok": True,
        "object": "hermes.dev_reliability_scorecard",
        "window": {"start": current_start, "end": now_value, "days": config.window_days},
        "config": {
            "escape_window_days": config.escape_window_days,
            "observed_min_samples": config.observed_min_samples,
            "observed_success_rate": config.observed_success_rate,
            "trusted_min_samples": config.trusted_min_samples,
            "trusted_success_rate": config.trusted_success_rate,
            "trusted_min_window_days": config.trusted_min_window_days,
        },
        "categories": category_rows,
        "weakest": weakest_categories(category_rows, limit=5),
        "summary": {
            "category_count": len(category_rows),
            "sample_count": sum(item["sample_count"] for item in category_rows),
            "trusted_count": sum(1 for item in category_rows if item["tier"] == "trusted"),
            "observed_count": sum(1 for item in category_rows if item["tier"] == "observed"),
            "unproven_count": sum(1 for item in category_rows if item["tier"] == "unproven"),
        },
        "advisory_only": True,
    }


def category_scorecard(
    *,
    category: str,
    outcomes: list[Dict[str, Any]],
    previous_outcomes: Optional[list[Dict[str, Any]]] = None,
    config: Optional[ReliabilityConfig] = None,
    window: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or reliability_config_from_env()
    previous_outcomes = previous_outcomes or []
    sample_count = len(outcomes)
    success_count = sum(1 for item in outcomes if outcome_success(item))
    escape_count = sum(1 for item in outcomes if item.get("escaped"))
    rework_count = sum(1 for item in outcomes if int(item.get("rework_count") or 0) > 0)
    success_rate = _rate(success_count, sample_count)
    previous_success_rate = _rate(
        sum(1 for item in previous_outcomes if outcome_success(item)),
        len(previous_outcomes),
    )
    scores = [
        float(item["output_contract_score"])
        for item in outcomes
        if item.get("output_contract_score") is not None
    ]
    row = {
        "object": "hermes.dev_reliability_category",
        "category": category,
        "sample_count": sample_count,
        "success_count": success_count,
        "failure_count": max(sample_count - success_count, 0),
        "success_rate": success_rate,
        "rework_rate": _rate(rework_count, sample_count),
        "escape_rate": _rate(escape_count, sample_count),
        "escape_count": escape_count,
        "median_output_contract_score": _median(scores),
        "trend": _trend(success_rate, previous_success_rate),
        "previous_success_rate": previous_success_rate,
        "tier": "unproven",
        "tier_reasons": [],
        "window": window or {},
        "advisory_only": True,
    }
    tier, reasons = classify_trust_tier(row, config=config)
    row["tier"] = tier
    row["tier_reasons"] = reasons
    return row


def classify_trust_tier(category: Dict[str, Any], *, config: Optional[ReliabilityConfig] = None) -> tuple[str, list[str]]:
    config = config or reliability_config_from_env()
    sample_count = int(category.get("sample_count") or 0)
    success_rate = float(category.get("success_rate") or 0.0)
    escape_count = int(category.get("escape_count") or 0)
    window_days = float((category.get("window") or {}).get("days") or config.window_days)
    reasons: list[str] = []
    if escape_count > 0:
        return "unproven", [f"{escape_count} escape(s) in window"]
    if sample_count < config.observed_min_samples:
        return "unproven", [f"{sample_count} sample(s), need {config.observed_min_samples} for observed"]
    if (
        sample_count >= config.trusted_min_samples
        and success_rate >= config.trusted_success_rate
        and window_days >= config.trusted_min_window_days
    ):
        return "trusted", []
    if success_rate >= config.observed_success_rate:
        if sample_count < config.trusted_min_samples:
            reasons.append(f"{sample_count} sample(s), need {config.trusted_min_samples} for trusted")
        if success_rate < config.trusted_success_rate:
            reasons.append(f"success_rate {success_rate:.2f}, need {config.trusted_success_rate:.2f} for trusted")
        if window_days < config.trusted_min_window_days:
            reasons.append(f"window {window_days:.0f}d, need {config.trusted_min_window_days:.0f}d for trusted")
        return "observed", reasons
    return "unproven", [f"success_rate {success_rate:.2f}, need {config.observed_success_rate:.2f} for observed"]


def weakest_categories(categories: Iterable[Dict[str, Any]], *, limit: int = 5) -> list[Dict[str, Any]]:
    rows = list(categories or [])
    rows.sort(key=lambda item: (
        TIER_RANK.get(str(item.get("tier") or "unproven"), 0),
        float(item.get("success_rate") or 0.0),
        -float(item.get("escape_rate") or 0.0),
        -int(item.get("sample_count") or 0),
        str(item.get("category") or ""),
    ))
    return rows[:max(1, min(int(limit or 5), 50))]


def measure_category_improvement(
    *,
    store: DevReliabilityStore,
    category: str,
    before_start: float,
    before_end: float,
    after_start: float,
    after_end: float,
    proposal_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    config: Optional[ReliabilityConfig] = None,
) -> Dict[str, Any]:
    config = config or reliability_config_from_env()
    before = store.list_outcomes(category=category, start=before_start, end=before_end, limit=5000)
    after = store.list_outcomes(category=category, start=after_start, end=after_end, limit=5000)
    before_row = category_scorecard(
        category=category,
        outcomes=before,
        config=config,
        window={"start": before_start, "end": before_end, "days": (before_end - before_start) / 86400},
    )
    after_row = category_scorecard(
        category=category,
        outcomes=after,
        previous_outcomes=before,
        config=config,
        window={"start": after_start, "end": after_end, "days": (after_end - after_start) / 86400},
    )
    return store.persist_improvement_measurement({
        "category": category,
        "proposal_id": proposal_id,
        "plan_id": plan_id,
        "before_score": before_row.get("success_rate"),
        "after_score": after_row.get("success_rate"),
        "before_sample_count": before_row.get("sample_count"),
        "after_sample_count": after_row.get("sample_count"),
        "before_window": before_row.get("window"),
        "after_window": after_row.get("window"),
        "warnings": [] if before and after else ["One or both measurement windows have no samples."],
    })


def recompute_reliability_outcomes(
    *,
    reliability_store: DevReliabilityStore,
    execution_store: Any,
    verification_store: Any = None,
    scm_store: Any = None,
    incident_store: Any = None,
    product_event_store: Any = None,
    project_id: Optional[str] = None,
    limit: int = 200,
    now: Optional[float] = None,
    config: Optional[ReliabilityConfig] = None,
) -> Dict[str, Any]:
    config = config or reliability_config_from_env()
    plans = execution_store.list_plans(limit=limit, project_id=project_id)
    incidents = incident_store.list_incidents(limit=500) if incident_store else []
    product_events = product_event_store.list_events(limit=1000) if product_event_store else []
    outcomes: list[Dict[str, Any]] = []
    warnings: list[str] = []
    for plan in plans:
        for task in plan.get("tasks") or []:
            status = str(task.get("status") or plan.get("status") or "").strip().lower()
            if status not in TERMINAL_STATUSES:
                continue
            plan_id = str(plan.get("plan_id") or "")
            task_id = str(task.get("task_id") or "")
            verification = _latest_verification(verification_store, plan_id=plan_id, task_id=task_id)
            pr_state = _latest_pr_state(scm_store, plan_id=plan_id, task_id=task_id)
            code_review = _latest_code_review_for_state(scm_store, pr_state=pr_state, plan_id=plan_id, task_id=task_id)
            readiness = _latest_readiness(scm_store, plan_id=plan_id, task_id=task_id)
            try:
                outcomes.append(reliability_store.upsert_outcome(compose_task_outcome(
                    plan=plan,
                    task=task,
                    verification=verification,
                    pr_state=pr_state,
                    code_review=code_review,
                    readiness=readiness,
                    incidents=incidents,
                    product_events=product_events,
                    config=config,
                    now=now,
                )))
            except Exception as exc:
                warnings.append(f"{plan_id}/{task_id}: {exc}")
    return {
        "ok": True,
        "object": "hermes.dev_reliability_recompute",
        "outcomes": outcomes,
        "count": len(outcomes),
        "warnings": warnings,
    }


def outcome_success(outcome: Dict[str, Any]) -> bool:
    source_refs = outcome.get("source_refs") if isinstance(outcome.get("source_refs"), dict) else {}
    if bool(source_refs.get("draft_pr_only")):
        return (
            str(outcome.get("terminal_status") or "").lower() in {"completed", "ready", "draft_pr_ready"}
            and bool(source_refs.get("draft_pr_ready"))
            and str(outcome.get("verification_verdict") or "").lower() in SUCCESS_VERIFICATION_VERDICTS
            and _draft_gate_success_or_unmeasured(outcome, gate="ci", success_states=SUCCESS_CI_STATES)
            and _draft_gate_success_or_unmeasured(outcome, gate="review", success_states=SUCCESS_REVIEW_VERDICTS)
            and not bool(outcome.get("escaped"))
        )
    return (
        bool(outcome.get("merged"))
        and str(outcome.get("terminal_status") or "").lower() in MERGED_STATUSES
        and str(outcome.get("verification_verdict") or "").lower() in SUCCESS_VERIFICATION_VERDICTS
        and str(outcome.get("ci_state") or "").lower() in SUCCESS_CI_STATES
        and str(outcome.get("code_review_verdict") or "").lower() in SUCCESS_REVIEW_VERDICTS
        and not bool(outcome.get("escaped"))
    )


def _draft_gate_success_or_unmeasured(outcome: Dict[str, Any], *, gate: str, success_states: set[str]) -> bool:
    source_refs = outcome.get("source_refs") if isinstance(outcome.get("source_refs"), dict) else {}
    gates = source_refs.get("gates") if isinstance(source_refs.get("gates"), dict) else {}
    outcome_key = "ci_state" if gate == "ci" else "code_review_verdict"
    state = str(outcome.get(outcome_key) or "").strip().lower()
    gate_state = str(gates.get(gate) or state).strip().lower()
    if gate_state in UNMEASURED_GATE_STATES and state in UNMEASURED_GATE_STATES:
        return True
    return state in success_states


def outcome_excluded(outcome: Dict[str, Any]) -> bool:
    """Return true for outcomes that must not count as reliability evidence."""

    source_refs = outcome.get("source_refs") if isinstance(outcome.get("source_refs"), dict) else {}
    if bool(source_refs.get("exclude_from_scorecard") or source_refs.get("invalid")):
        return True
    outcome_id = str(outcome.get("outcome_id") or "").strip()
    if outcome_id and outcome_id in _excluded_outcome_ids():
        return True
    return False


def _excluded_outcome_ids() -> set[str]:
    configured = os.getenv("HERMES_DEV_RELIABILITY_EXCLUDED_OUTCOME_IDS")
    values = set(DEFAULT_EXCLUDED_OUTCOME_IDS)
    if configured:
        values.update(part.strip() for part in configured.split(",") if part.strip())
    return values


def normalize_risk_level(value: Any) -> str:
    text = str(value or "low").strip().lower().replace("_", "-")
    if text in {"high", "medium", "low"}:
        return text
    if text in {"med", "moderate"}:
        return "medium"
    return "low"


def rework_count_from_sources(*, task: Dict[str, Any], code_review: Optional[Dict[str, Any]]) -> int:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    count = 0
    for key in ("rework_count", "retry_count", "follow_up_count", "changes_requested_count"):
        try:
            count += int(payload.get(key) or task.get(key) or 0)
        except Exception:
            pass
    if str((code_review or {}).get("verdict") or "").lower() == "changes_requested":
        count += 1
    return max(0, count)


def _latest_verification(store: Any, *, plan_id: str, task_id: str) -> Dict[str, Any]:
    if not store:
        return {}
    try:
        return store.latest_for_task(plan_id=plan_id, task_id=task_id) or {}
    except Exception:
        return {}


def _latest_pr_state(store: Any, *, plan_id: str, task_id: str) -> Dict[str, Any]:
    if not store or not hasattr(store, "_conn"):
        return {}
    row = store._conn.execute(
        """
        SELECT *
        FROM dev_pr_states
        WHERE plan_id = ? AND task_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (plan_id, task_id),
    ).fetchone()
    if not row:
        return {}
    state = dict(row)
    state["mergeable"] = None if state.get("mergeable") is None else bool(state.get("mergeable"))
    for key in ("ci_status", "warnings", "raw"):
        state[key] = _loads(state.get(key), [] if key == "warnings" else {})
    return state


def _latest_code_review_for_state(store: Any, *, pr_state: Dict[str, Any], plan_id: str, task_id: str) -> Dict[str, Any]:
    if not store or not hasattr(store, "_conn"):
        return {}
    if pr_state.get("repo") and pr_state.get("pr_number"):
        try:
            return store.latest_code_review(
                repo=pr_state["repo"],
                pr_number=int(pr_state["pr_number"]),
                head_sha=pr_state.get("head_sha"),
            ) or {}
        except Exception:
            pass
    row = store._conn.execute(
        """
        SELECT *
        FROM dev_code_review_runs
        WHERE plan_id = ? AND task_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (plan_id, task_id),
    ).fetchone()
    if not row:
        return {}
    review = dict(row)
    for key in ("findings", "evidence_refs", "warnings"):
        review[key] = _loads(review.get(key), [])
    return review


def _latest_readiness(store: Any, *, plan_id: str, task_id: str) -> Dict[str, Any]:
    if not store or not hasattr(store, "_conn"):
        return {}
    row = store._conn.execute(
        """
        SELECT *
        FROM dev_merge_readiness
        WHERE plan_id = ? AND task_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (plan_id, task_id),
    ).fetchone()
    if not row:
        return {}
    readiness = dict(row)
    readiness["ready"] = bool(readiness.get("ready"))
    for key in ("blocked_by", "gates", "pr_state", "verification", "code_review"):
        readiness[key] = _loads(readiness.get(key), [] if key == "blocked_by" else {})
    return readiness


def _merged_from_sources(status: str, *, pr_state: Optional[Dict[str, Any]], readiness: Optional[Dict[str, Any]]) -> bool:
    if status in {"merged", "shipped"}:
        return True
    merge_state = str((pr_state or {}).get("merge_state") or "").lower()
    raw = (pr_state or {}).get("raw") if isinstance((pr_state or {}).get("raw"), dict) else {}
    if merge_state == "merged" or raw.get("merged") is True:
        return True
    gates = (readiness or {}).get("gates") if isinstance((readiness or {}).get("gates"), dict) else {}
    return bool((readiness or {}).get("ready") and gates.get("mergeable") is True and status == "completed")


def _outcome_values(outcome: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        outcome["outcome_id"],
        outcome["plan_id"],
        outcome["task_id"],
        outcome["category"],
        outcome.get("profile_id"),
        outcome.get("risk_level"),
        outcome.get("terminal_status"),
        1 if outcome.get("merged") else 0,
        outcome.get("verification_verdict"),
        outcome.get("ci_state"),
        outcome.get("code_review_verdict"),
        outcome.get("output_contract_score"),
        int(outcome.get("rework_count") or 0),
        1 if outcome.get("escaped") else 0,
        _json(outcome.get("escape_refs") or []),
        _json(outcome.get("source_refs") or {}),
        _json({k: v for k, v in outcome.items() if k not in {"payload"}}),
        float(outcome.get("created_at") or time.time()),
        float(outcome.get("updated_at") or time.time()),
        outcome.get("completed_at"),
        outcome.get("merged_at"),
    )


def _outcome_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    payload = _loads(row["payload"], {})
    outcome = {
        **payload,
        "object": "hermes.dev_reliability_outcome",
        "outcome_id": row["outcome_id"],
        "plan_id": row["plan_id"],
        "task_id": row["task_id"],
        "category": row["category"],
        "profile_id": row["profile_id"],
        "risk_level": row["risk_level"],
        "terminal_status": row["terminal_status"],
        "merged": bool(row["merged"]),
        "verification_verdict": row["verification_verdict"],
        "ci_state": row["ci_state"],
        "code_review_verdict": row["code_review_verdict"],
        "output_contract_score": row["output_contract_score"],
        "rework_count": int(row["rework_count"] or 0),
        "escaped": bool(row["escaped"]),
        "escape_refs": _loads(row["escape_refs"], []),
        "source_refs": _loads(row["source_refs"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
        "merged_at": row["merged_at"],
    }
    outcome["success"] = outcome_success(outcome)
    return outcome


def _measurement_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "object": "hermes.dev_reliability_improvement",
        "measurement_id": row["measurement_id"],
        "category": row["category"],
        "proposal_id": row["proposal_id"],
        "plan_id": row["plan_id"],
        "before_score": row["before_score"],
        "after_score": row["after_score"],
        "before_sample_count": int(row["before_sample_count"] or 0),
        "after_sample_count": int(row["after_sample_count"] or 0),
        "before_window": _loads(row["before_window"], {}),
        "after_window": _loads(row["after_window"], {}),
        "measured_at": row["measured_at"],
        "warnings": _loads(row["warnings"], []),
    }


def _matches_sha(expected: str, candidate: Any) -> bool:
    candidate_text = str(candidate or "").strip()
    expected_text = str(expected or "").strip()
    if not candidate_text or not expected_text:
        return False
    return candidate_text.startswith(expected_text) or expected_text.startswith(candidate_text)


def _outcome_time(outcome: Dict[str, Any]) -> float:
    return float(outcome.get("completed_at") or outcome.get("updated_at") or outcome.get("created_at") or 0)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _trend(current: Optional[float], previous: Optional[float]) -> str:
    if current is None or previous is None:
        return "flat"
    delta = current - previous
    if delta >= 0.02:
        return "improving"
    if delta <= -0.02:
        return "regressing"
    return "flat"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _optional_lower(value: Any) -> Optional[str]:
    text = _optional_text(value)
    return text.lower() if text else None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_float(*values: Any) -> Optional[float]:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default
