"""Observe-only Dev harness component inventory and experience reports."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gateway.dev_execution import (
    DevExecutionStore,
    derive_execution_plan_status,
    list_launch_profiles,
    list_policy_profiles,
)
from gateway.dev_worker_runtimes import list_worker_runtimes
from gateway.subagent_events import SubagentEventStore
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


HERMES_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_harness_reports (
    report_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    scope TEXT,
    component_hashes TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dev_harness_reports_created_at
    ON dev_harness_reports(created_at DESC);
"""


@dataclass
class DevHarnessObservabilityStore:
    """Persistence for lightweight harness report metadata and JSON payloads."""

    db_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.db_path = self.db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="state.db")
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def persist_report(self, report: Dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_harness_reports (
                    report_id, created_at, scope, component_hashes, payload
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report["report_id"],
                    float(report["created_at"]),
                    _canonical_json(report.get("scope") or {}),
                    _canonical_json(report.get("component_hashes") or {}),
                    json.dumps(report, ensure_ascii=False),
                ),
            )

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT payload
            FROM dev_harness_reports
            WHERE report_id = ?
            """,
            (str(report_id or "").strip(),),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["payload"])

    def list_reports(self, *, limit: int = 50) -> list[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT report_id, created_at, scope, component_hashes
            FROM dev_harness_reports
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(int(limit or 50), 200)),),
        ).fetchall()
        reports = []
        for row in rows:
            try:
                scope = json.loads(row["scope"] or "{}")
            except Exception:
                scope = {}
            try:
                component_hashes = json.loads(row["component_hashes"] or "{}")
            except Exception:
                component_hashes = {}
            reports.append({
                "report_id": row["report_id"],
                "created_at": float(row["created_at"]),
                "scope": scope,
                "component_hashes": component_hashes,
            })
        return reports


def list_harness_components(
    *,
    store: Optional[DevExecutionStore] = None,
) -> list[Dict[str, Any]]:
    """Return the active Dev harness components and deterministic version hashes."""

    dynamic_runbooks = []
    if store is not None:
        try:
            dynamic_runbooks = store.list_runbooks(limit=200)
        except Exception:
            dynamic_runbooks = []

    specs = [
        _file_component(
            component_id="runtime-selection-policy",
            kind="policy",
            label="Runtime Selection Policy",
            source="gateway/dev_control/runtime_selection.py",
            description="Chooses AO, OpenHands, or fixture for Dev tasks when runtime is automatic.",
        ),
        _dynamic_component(
            component_id="launch-profiles",
            kind="profile_set",
            label="Dev Launch Profiles",
            source="gateway/dev_execution.py:list_launch_profiles",
            value=list_launch_profiles(),
            description="Named task profiles that bind projects, permissions, agents, models, and default runtime.",
        ),
        _dynamic_component(
            component_id="runbooks-policy-profiles",
            kind="policy_profile_set",
            label="Runbooks and Policy Profiles",
            source="gateway/dev_execution.py:list_policy_profiles + state.db:dev_execution_runbooks",
            value={
                "builtin_policy_profiles": list_policy_profiles(),
                "project_runbooks": dynamic_runbooks,
            },
            description="Project policies and attempt limits used by Dev supervisor decisions.",
        ),
        _file_component(
            component_id="supervisor-review",
            kind="supervisor",
            label="Supervisor and Review Logic",
            source="gateway/dev_execution.py",
            description="Status, synthesis, review, apply-review, approvals, and supervisor loop logic.",
        ),
        _file_component(
            component_id="summary-quality-classifier",
            kind="classifier",
            label="Summary Quality Classifier",
            source="gateway/dev_execution.py",
            description="Weak-summary, prompt-echo, missing-marker, and verification-evidence detection.",
        ),
        _dynamic_component(
            component_id="runtime-adapters",
            kind="runtime_capability_set",
            label="Runtime Adapters and Capabilities",
            source="gateway/dev_worker_runtimes.py:list_worker_runtimes",
            value=list_worker_runtimes(),
            description="AO, OpenHands, and fixture runtime capability discovery.",
        ),
        _file_component(
            component_id="subagent-event-schema",
            kind="event_schema",
            label="Subagent Event Schema",
            source="gateway/dev_control/events.py",
            description="Normalized subagent event construction and compatibility shape.",
        ),
        _file_component(
            component_id="worker-contract-template",
            kind="prompt_template",
            label="Worker Prompt Contract Template",
            source="tools/ao_delegate_tool.py",
            description="AO worker prompt and delegation contract framing used for production AO launches.",
        ),
        _file_component(
            component_id="dev-tool-contracts",
            kind="tool_schema",
            label="Dev Tool Contracts",
            source="tools/dev_execution_tools.py",
            description="Dev-facing tool schemas for plans, runtime routing, supervisor, approvals, and harness reports.",
        ),
    ]
    return specs


def generate_harness_report(
    *,
    store: DevExecutionStore,
    event_store: SubagentEventStore,
    plan_ids: Optional[list[str]] = None,
    project_id: Optional[str] = None,
    limit: int = 25,
    since: Optional[float] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Build and optionally persist an observe-only Dev harness experience report."""

    created_at = time.time()
    report_id = f"devharness-{uuid.uuid4().hex[:10]}"
    components = list_harness_components(store=store)
    component_hashes = {component["component_id"]: component["version_hash"] for component in components}
    plans = _select_plans(store=store, plan_ids=plan_ids, project_id=project_id, limit=limit, since=since)

    plan_observations: list[Dict[str, Any]] = []
    evidence: list[Dict[str, Any]] = []
    runtime_stats: Dict[str, Dict[str, Any]] = {}
    pattern_counts: Dict[str, Dict[str, Any]] = {}
    summary = {
        "plan_count": 0,
        "task_count": 0,
        "by_plan_status": {},
        "by_task_status": {},
        "by_runtime": {},
        "by_summary_quality": {},
        "by_review_status": {},
        "by_recommended_action": {},
        "weak_summary_count": 0,
        "missing_marker_count": 0,
        "prompt_echo_count": 0,
        "fallback_count": 0,
        "follow_up_count": 0,
        "repair_retry_count": 0,
        "approval_pending_count": 0,
        "human_review_count": 0,
    }

    for plan in plans:
        status_payload = derive_execution_plan_status(
            store=store,
            plan_id=plan["plan_id"],
            event_store=event_store,
        )
        tasks = status_payload.get("tasks") or []
        summary["plan_count"] += 1
        _inc(summary["by_plan_status"], status_payload.get("status") or "unknown")
        _inc(summary["by_review_status"], status_payload.get("review_status") or "unknown")
        _inc(summary["by_recommended_action"], status_payload.get("recommended_action") or "none")
        if status_payload.get("review_status") == "human_review_required":
            summary["human_review_count"] += 1
        if (status_payload.get("plan") or {}).get("supervisor_approval_status") == "pending":
            summary["approval_pending_count"] += 1

        task_observations = []
        for task in tasks:
            summary["task_count"] += 1
            runtime = str(task.get("runtime") or "unknown")
            task_status = str(task.get("status") or "unknown")
            quality = str(task.get("summary_quality") or "unknown")
            _inc(summary["by_task_status"], task_status)
            _inc(summary["by_runtime"], runtime)
            _inc(summary["by_summary_quality"], quality)

            runtime_entry = runtime_stats.setdefault(runtime, {
                "runtime": runtime,
                "tasks": 0,
                "completed": 0,
                "needs_review": 0,
                "failed": 0,
                "warnings": 0,
                "follow_ups": 0,
                "fallbacks": 0,
            })
            runtime_entry["tasks"] += 1
            if task_status in runtime_entry:
                runtime_entry[task_status] += 1
            if quality == "warning":
                runtime_entry["warnings"] += 1
                summary["weak_summary_count"] += 1

            if task.get("runtime_fallback_reason"):
                summary["fallback_count"] += 1
                runtime_entry["fallbacks"] += 1
                _record_pattern(pattern_counts, evidence, "runtime_fallback", task, "Runtime fallback occurred.")

            recent_action = str(task.get("recent_action") or "")
            if recent_action == "follow-up":
                summary["follow_up_count"] += 1
                runtime_entry["follow_ups"] += 1
            if recent_action == "repair-retry":
                summary["repair_retry_count"] += 1

            warning = str(task.get("summary_warning") or "")
            _classify_warning(
                warning=warning,
                task=task,
                summary=summary,
                pattern_counts=pattern_counts,
                evidence=evidence,
            )
            task_observations.append(_task_observation(task))

        plan_observations.append({
            "plan_id": status_payload.get("plan_id"),
            "title": (status_payload.get("plan") or {}).get("title"),
            "status": status_payload.get("status"),
            "review_status": status_payload.get("review_status"),
            "recommended_action": status_payload.get("recommended_action"),
            "next_step": status_payload.get("next_step"),
            "policy_profile": status_payload.get("policy_profile"),
            "task_count": len(tasks),
            "tasks": task_observations,
        })

    failure_patterns = [
        {
            "pattern": pattern,
            "count": entry["count"],
            "description": entry["description"],
            "evidence_refs": entry["evidence_refs"][:10],
        }
        for pattern, entry in sorted(pattern_counts.items())
    ]

    scope = {
        "plan_ids": plan_ids or None,
        "project_id": project_id,
        "limit": limit,
        "since": since,
    }
    report = {
        "ok": True,
        "object": "hermes.dev_harness_report",
        "report_id": report_id,
        "created_at": created_at,
        "scope": scope,
        "components": components,
        "component_hashes": component_hashes,
        "summary": summary,
        "runtime_observations": list(runtime_stats.values()),
        "failure_patterns": failure_patterns,
        "plan_observations": plan_observations,
        "evidence": evidence,
    }
    if persist:
        report_store = DevHarnessObservabilityStore(store.db_path)
        try:
            report_store.persist_report(report)
        finally:
            report_store.close()
    return report


def _select_plans(
    *,
    store: DevExecutionStore,
    plan_ids: Optional[list[str]],
    project_id: Optional[str],
    limit: int,
    since: Optional[float],
) -> list[Dict[str, Any]]:
    if plan_ids:
        plans = [plan for plan_id in plan_ids if (plan := store.get_plan(str(plan_id)))]
    else:
        plans = store.list_plans(limit=limit)
    if project_id:
        plans = [
            plan for plan in plans
            if project_id in {str(task.get("project_id") or "") for task in plan.get("tasks") or []}
        ]
    if since is not None:
        plans = [plan for plan in plans if float(plan.get("updated_at") or 0) >= float(since)]
    return plans[: max(1, min(int(limit or 25), 100))]


def _task_observation(task: Dict[str, Any]) -> Dict[str, Any]:
    last_event = task.get("last_event") if isinstance(task.get("last_event"), dict) else {}
    output_tail = str((last_event or {}).get("output_tail") or "")
    return {
        "plan_id": task.get("plan_id"),
        "task_id": task.get("task_id"),
        "goal": task.get("goal"),
        "status": task.get("status"),
        "status_reason": task.get("status_reason"),
        "runtime": task.get("runtime"),
        "runtime_session_id": task.get("runtime_session_id") or task.get("ao_session_id"),
        "selected_runtime": task.get("selected_runtime"),
        "runtime_selection_reason": task.get("runtime_selection_reason"),
        "runtime_fallback_reason": task.get("runtime_fallback_reason"),
        "summary_quality": task.get("summary_quality"),
        "summary_warning": task.get("summary_warning"),
        "recent_action": task.get("recent_action"),
        "recent_action_status": task.get("recent_action_status"),
        "files_read_count": len(task.get("files_read") or []),
        "files_written_count": len(task.get("files_written") or []),
        "verification_evidence_count": len(task.get("verification_evidence") or []),
        "has_output_tail": bool(output_tail),
        "output_tail_chars": len(output_tail),
        "last_event_id": (last_event or {}).get("event_id"),
    }


def _classify_warning(
    *,
    warning: str,
    task: Dict[str, Any],
    summary: Dict[str, Any],
    pattern_counts: Dict[str, Dict[str, Any]],
    evidence: list[Dict[str, Any]],
) -> None:
    lowered = warning.lower()
    if not lowered:
        return
    if "marker" in lowered:
        summary["missing_marker_count"] += 1
        _record_pattern(pattern_counts, evidence, "missing_completion_marker", task, warning)
    if "echo" in lowered or "contract" in lowered:
        summary["prompt_echo_count"] += 1
        _record_pattern(pattern_counts, evidence, "prompt_echo_summary", task, warning)
    if "too short" in lowered or "did not provide a summary" in lowered or "weak" in lowered:
        _record_pattern(pattern_counts, evidence, "weak_or_missing_summary", task, warning)


def _record_pattern(
    pattern_counts: Dict[str, Dict[str, Any]],
    evidence: list[Dict[str, Any]],
    pattern: str,
    task: Dict[str, Any],
    description: str,
) -> None:
    evidence_id = f"evidence-{len(evidence) + 1}"
    evidence.append({
        "evidence_id": evidence_id,
        "pattern": pattern,
        "plan_id": task.get("plan_id"),
        "task_id": task.get("task_id"),
        "runtime": task.get("runtime"),
        "runtime_session_id": task.get("runtime_session_id") or task.get("ao_session_id"),
        "summary_quality": task.get("summary_quality"),
        "summary_warning": task.get("summary_warning"),
        "runtime_selection_reason": task.get("runtime_selection_reason"),
        "runtime_fallback_reason": task.get("runtime_fallback_reason"),
        "recent_action": task.get("recent_action"),
        "recent_action_status": task.get("recent_action_status"),
        "last_event_id": (task.get("last_event") or {}).get("event_id")
        if isinstance(task.get("last_event"), dict) else None,
    })
    entry = pattern_counts.setdefault(pattern, {
        "count": 0,
        "description": description,
        "evidence_refs": [],
    })
    entry["count"] += 1
    entry["evidence_refs"].append(evidence_id)


def _file_component(
    *,
    component_id: str,
    kind: str,
    label: str,
    source: str,
    description: str,
) -> Dict[str, Any]:
    path = HERMES_ROOT / source
    version_hash = _file_hash(path)
    updated_at = path.stat().st_mtime if path.exists() else None
    return {
        "component_id": component_id,
        "kind": kind,
        "label": label,
        "source": source,
        "version_hash": version_hash,
        "updated_at": updated_at,
        "description": description,
    }


def _dynamic_component(
    *,
    component_id: str,
    kind: str,
    label: str,
    source: str,
    value: Any,
    description: str,
) -> Dict[str, Any]:
    return {
        "component_id": component_id,
        "kind": kind,
        "label": label,
        "source": source,
        "version_hash": _hash_value(value),
        "updated_at": time.time(),
        "description": description,
    }


def _file_hash(path: Path) -> str:
    if not path.exists():
        return _hash_value({"missing": str(path)})
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _inc(counter: Dict[str, int], key: Any) -> None:
    text = str(key or "unknown")
    counter[text] = int(counter.get(text) or 0) + 1
