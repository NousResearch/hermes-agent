"""Durable Dev planning artifacts generated from clarification sessions."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agent.auxiliary_client import call_llm
from gateway.dev_control.acceptance_criteria import (
    ACCEPTANCE_CRITERION_JSON_SCHEMA,
    ALLOWED_VERIFICATION_COMMAND_SHAPES,
    acceptance_criteria_to_strings,
    normalize_acceptance_criteria,
    validate_and_downgrade_criteria,
)
from gateway.dev_control.clarifications import DevClarificationStore, get_clarification
from gateway.dev_control.project_scope import resolve_project_id
from gateway.dev_control.worker_output_contract import append_worker_output_contract
from gateway.dev_execution import DevExecutionStore
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_plan_artifacts (
    plan_artifact_id TEXT PRIMARY KEY,
    clarification_id TEXT NOT NULL,
    project_id TEXT,
    session_id TEXT,
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    markdown TEXT NOT NULL,
    payload TEXT NOT NULL,
    revision_history TEXT NOT NULL,
    superseded_by TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    approved_at REAL,
    cancelled_at REAL
);

CREATE INDEX IF NOT EXISTS idx_dev_plan_artifacts_clarification
    ON dev_plan_artifacts(clarification_id, version DESC);

CREATE INDEX IF NOT EXISTS idx_dev_plan_artifacts_project_status
    ON dev_plan_artifacts(project_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_plan_artifacts_updated_at
    ON dev_plan_artifacts(updated_at DESC);

CREATE TABLE IF NOT EXISTS dev_plan_artifact_builds (
    build_id TEXT PRIMARY KEY,
    plan_artifact_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    task_count INTEGER NOT NULL,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dev_plan_artifact_builds_artifact
    ON dev_plan_artifact_builds(plan_artifact_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_plan_artifact_builds_plan
    ON dev_plan_artifact_builds(plan_id);
"""

ARTIFACT_STATUSES = {"draft", "reviewing", "approved", "superseded", "cancelled"}
ARTIFACT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title",
        "overview",
        "product_intent",
        "scope",
        "non_goals",
        "assumptions",
        "user_workflow",
        "implementation_slices",
        "validation_slices",
        "acceptance_criteria",
        "risks",
        "open_questions",
        "recommended_next_action",
    ],
    "properties": {
        "title": {"type": "string"},
        "overview": {"type": "string"},
        "product_intent": {"type": "string"},
        "scope": {"type": "array", "items": {"type": "string"}},
        "non_goals": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "user_workflow": {"type": "array", "items": {"type": "string"}},
        "implementation_slices": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "description"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "validation_slices": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "description"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "acceptance_criteria": {"type": "array", "items": ACCEPTANCE_CRITERION_JSON_SCHEMA},
        "risks": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "recommended_next_action": {"type": "string"},
    },
}

EXECUTION_TASKS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tasks"],
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["goal", "prompt", "profile_id", "permissions", "acceptance_criteria"],
                "properties": {
                    "goal": {"type": "string"},
                    "prompt": {"type": "string"},
                    "profile_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "permissions": {"type": "string"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


@dataclass
class DevPlanArtifactStore:
    """Persistence for durable Dev planning artifacts."""

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

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = float(payload.get("created_at") or time.time())
        payload = dict(payload)
        payload["created_at"] = now
        payload["updated_at"] = now
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_plan_artifacts (
                    plan_artifact_id, clarification_id, project_id, session_id,
                    status, version, source, title, markdown, payload,
                    revision_history, superseded_by, created_at, updated_at,
                    approved_at, cancelled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(payload),
            )
        return self.get(payload["plan_artifact_id"]) or payload

    def update(self, plan_artifact_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get(plan_artifact_id)
        if not current:
            raise KeyError(f"Plan artifact not found: {plan_artifact_id}")
        payload = {**current, **updates, "updated_at": time.time()}
        with self._conn:
            self._conn.execute(
                """
                UPDATE dev_plan_artifacts
                SET clarification_id = ?, project_id = ?, session_id = ?,
                    status = ?, version = ?, source = ?, title = ?, markdown = ?,
                    payload = ?, revision_history = ?, superseded_by = ?,
                    created_at = ?, updated_at = ?, approved_at = ?, cancelled_at = ?
                WHERE plan_artifact_id = ?
                """,
                (*_row_values(payload)[1:], plan_artifact_id),
            )
        return self.get(plan_artifact_id) or payload

    def get(self, plan_artifact_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_plan_artifacts WHERE plan_artifact_id = ?",
            (str(plan_artifact_id or "").strip(),),
        ).fetchone()
        return _row_to_payload(row) if row else None

    def list(
        self,
        *,
        clarification_id: Optional[str] = None,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if clarification_id:
            clauses.append("clarification_id = ?")
            params.append(str(clarification_id).strip())
        if project_id:
            clauses.append("project_id = ?")
            params.append(str(project_id).strip())
        if status:
            clauses.append("status = ?")
            params.append(str(status).strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM dev_plan_artifacts
            {where}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_payload(row) for row in rows]

    def next_version(self, clarification_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM dev_plan_artifacts WHERE clarification_id = ?",
            (str(clarification_id or "").strip(),),
        ).fetchone()
        return int(row["version"] or 0) + 1

    def create_build(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(payload)
        payload["created_at"] = float(payload.get("created_at") or time.time())
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_plan_artifact_builds (
                    build_id, plan_artifact_id, plan_id, status, source,
                    task_count, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["build_id"],
                    payload["plan_artifact_id"],
                    payload["plan_id"],
                    payload["status"],
                    payload["source"],
                    int(payload.get("task_count") or 0),
                    float(payload["created_at"]),
                    json.dumps(payload.get("payload") or {}, ensure_ascii=False),
                ),
            )
        return self.get_build(payload["build_id"]) or payload

    def get_build(self, build_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_plan_artifact_builds WHERE build_id = ?",
            (str(build_id or "").strip(),),
        ).fetchone()
        return _build_row_to_payload(row) if row else None

    def list_builds(self, plan_artifact_id: str, *, limit: int = 25) -> list[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM dev_plan_artifact_builds
            WHERE plan_artifact_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (str(plan_artifact_id or "").strip(), max(1, min(int(limit or 25), 100))),
        ).fetchall()
        return [_build_row_to_payload(row) for row in rows]


def create_plan_artifact(
    *,
    store: DevPlanArtifactStore,
    clarification_store: DevClarificationStore,
    clarification_id: str,
) -> Dict[str, Any]:
    clarification = get_clarification(store=clarification_store, clarification_id=clarification_id)
    if clarification.get("status") != "completed":
        raise ValueError("Plan artifacts can only be created from completed clarifications")
    artifact = _generate_artifact(clarification)
    version = store.next_version(clarification["clarification_id"])
    payload = _artifact_payload(
        clarification=clarification,
        artifact=artifact,
        version=version,
        status="draft",
        revision_history=[],
    )
    return store.create(payload)


def list_plan_artifacts(
    *,
    store: DevPlanArtifactStore,
    clarification_id: Optional[str] = None,
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    data = store.list(
        clarification_id=clarification_id,
        project_id=project_id,
        status=status,
        limit=limit,
    )
    return {"object": "list", "data": data, "total": len(data)}


def get_plan_artifact(*, store: DevPlanArtifactStore, plan_artifact_id: str) -> Dict[str, Any]:
    artifact = store.get(plan_artifact_id)
    if not artifact:
        raise KeyError(f"Plan artifact not found: {plan_artifact_id}")
    return artifact


def revise_plan_artifact(
    *,
    store: DevPlanArtifactStore,
    clarification_store: DevClarificationStore,
    plan_artifact_id: str,
    feedback: str,
) -> Dict[str, Any]:
    current = get_plan_artifact(store=store, plan_artifact_id=plan_artifact_id)
    if current.get("status") in {"cancelled", "superseded"}:
        raise ValueError(f"Plan artifact is {current.get('status')} and cannot be revised")
    instruction = str(feedback or "").strip()
    if not instruction:
        raise ValueError("revision feedback is required")
    clarification = get_clarification(
        store=clarification_store,
        clarification_id=current["clarification_id"],
    )
    artifact = _generate_artifact(clarification, previous=current, feedback=instruction)
    revision_history = list(current.get("revision_history") or [])
    revision_history.append({
        "from_plan_artifact_id": current["plan_artifact_id"],
        "from_version": current["version"],
        "feedback": instruction,
        "revised_at": time.time(),
    })
    payload = _artifact_payload(
        clarification=clarification,
        artifact=artifact,
        version=store.next_version(current["clarification_id"]),
        status="draft",
        revision_history=revision_history,
    )
    created = store.create(payload)
    store.update(current["plan_artifact_id"], {
        "status": "superseded",
        "superseded_by": created["plan_artifact_id"],
    })
    return created


def approve_plan_artifact(*, store: DevPlanArtifactStore, plan_artifact_id: str) -> Dict[str, Any]:
    current = get_plan_artifact(store=store, plan_artifact_id=plan_artifact_id)
    if current.get("status") in {"cancelled", "superseded"}:
        raise ValueError(f"Plan artifact is {current.get('status')} and cannot be approved")
    return store.update(plan_artifact_id, {
        "status": "approved",
        "approved_at": time.time(),
    })


def cancel_plan_artifact(
    *,
    store: DevPlanArtifactStore,
    plan_artifact_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    current = get_plan_artifact(store=store, plan_artifact_id=plan_artifact_id)
    if current.get("status") == "superseded":
        raise ValueError("Superseded plan artifacts cannot be cancelled")
    history = list(current.get("revision_history") or [])
    if reason:
        history.append({"cancel_reason": str(reason).strip(), "cancelled_at": time.time()})
    return store.update(plan_artifact_id, {
        "status": "cancelled",
        "cancelled_at": time.time(),
        "revision_history": history,
    })


def create_execution_plan_from_artifact(
    *,
    artifact_store: DevPlanArtifactStore,
    execution_store: DevExecutionStore,
    plan_artifact_id: str,
) -> Dict[str, Any]:
    artifact = get_plan_artifact(store=artifact_store, plan_artifact_id=plan_artifact_id)
    if artifact.get("status") != "approved":
        raise ValueError(f"Plan artifact is {artifact.get('status')} and is not approved for build")
    drafted = _generate_execution_tasks(artifact)
    build_id = f"devbuild-{uuid.uuid4().hex[:10]}"
    plan = execution_store.create_plan(
        title=f"Build: {artifact.get('title') or 'Planning artifact'}",
        vision_brief=_artifact_vision_brief(artifact),
        tasks=drafted["tasks"],
    )
    build = {
        "object": "hermes.dev_plan_artifact_build",
        "build_id": build_id,
        "plan_artifact_id": artifact["plan_artifact_id"],
        "plan_id": plan["plan_id"],
        "status": "created",
        "source": drafted["source"],
        "task_count": len(plan.get("tasks") or []),
        "created_at": time.time(),
        "payload": {
            "artifact_title": artifact.get("title"),
            "artifact_version": artifact.get("version"),
            "draft_warning": drafted.get("warning"),
            "tasks": drafted["tasks"],
        },
        "plan": plan,
    }
    execution_store.create_draft_review(
        plan_id=plan["plan_id"],
        plan_artifact_id=artifact["plan_artifact_id"],
        build_id=build_id,
        source=drafted["source"],
        payload={"draft_warning": drafted.get("warning")},
    )
    persisted = artifact_store.create_build(build)
    persisted["plan"] = execution_store.get_plan(plan["plan_id"]) or plan
    return persisted


def list_plan_artifact_builds(
    *,
    store: DevPlanArtifactStore,
    plan_artifact_id: str,
    limit: int = 25,
) -> Dict[str, Any]:
    get_plan_artifact(store=store, plan_artifact_id=plan_artifact_id)
    data = store.list_builds(plan_artifact_id, limit=limit)
    return {"object": "list", "data": data, "total": len(data)}


def get_execution_plan_draft_review(
    *,
    execution_store: DevExecutionStore,
    plan_id: str,
) -> Dict[str, Any]:
    plan = execution_store.get_plan(plan_id)
    if not plan:
        raise KeyError(f"Dev execution plan not found: {plan_id}")
    review = execution_store.get_draft_review(plan_id)
    if not review:
        raise KeyError(f"Draft review not found for Dev execution plan: {plan_id}")
    return {**review, "plan": plan}


def revise_execution_plan_draft(
    *,
    artifact_store: DevPlanArtifactStore,
    execution_store: DevExecutionStore,
    plan_id: str,
    feedback: str,
) -> Dict[str, Any]:
    review = get_execution_plan_draft_review(execution_store=execution_store, plan_id=plan_id)
    status = review.get("draft_status")
    if status == "cancelled":
        raise ValueError("Cancelled draft plans cannot be revised")
    if status == "approved_for_launch":
        raise ValueError("Approved draft plans cannot be revised")
    instruction = str(feedback or "").strip()
    if not instruction:
        raise ValueError("revision feedback is required")
    if execution_store.plan_has_launched_tasks(plan_id):
        raise ValueError("Draft plan tasks cannot be revised after any task has launched")
    artifact = get_plan_artifact(store=artifact_store, plan_artifact_id=review["plan_artifact_id"])
    current_plan = execution_store.get_plan(plan_id)
    drafted = _generate_execution_tasks(
        artifact,
        current_tasks=(current_plan or {}).get("tasks") or [],
        feedback=instruction,
    )
    previous_tasks = [
        _task_revision_snapshot(task)
        for task in ((current_plan or {}).get("tasks") or [])
    ]
    plan = execution_store.replace_plan_tasks(plan_id=plan_id, tasks=drafted["tasks"])
    revision_history = list(review.get("revision_history") or [])
    revision_history.append({
        "version": review.get("version"),
        "feedback": instruction,
        "source": drafted["source"],
        "draft_warning": drafted.get("warning"),
        "revised_at": time.time(),
        "tasks": previous_tasks,
    })
    updated = execution_store.update_draft_review(plan_id, {
        "draft_status": "revision_requested",
        "version": int(review.get("version") or 1) + 1,
        "revision_history": revision_history,
        "source": drafted["source"],
        "payload": {
            **(review.get("payload") or {}),
            "last_feedback": instruction,
            "draft_warning": drafted.get("warning"),
        },
    })
    return {**updated, "plan": plan}


def approve_execution_plan_draft(
    *,
    execution_store: DevExecutionStore,
    plan_id: str,
) -> Dict[str, Any]:
    review = get_execution_plan_draft_review(execution_store=execution_store, plan_id=plan_id)
    if review.get("draft_status") == "cancelled":
        raise ValueError("Cancelled draft plans cannot be approved")
    updated = execution_store.update_draft_review(plan_id, {
        "draft_status": "approved_for_launch",
        "approved_at": time.time(),
    })
    return {**updated, "plan": execution_store.get_plan(plan_id)}


def cancel_execution_plan_draft(
    *,
    execution_store: DevExecutionStore,
    plan_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    review = get_execution_plan_draft_review(execution_store=execution_store, plan_id=plan_id)
    if review.get("draft_status") == "approved_for_launch":
        raise ValueError("Approved draft plans cannot be cancelled")
    history = list(review.get("revision_history") or [])
    if reason:
        history.append({"cancel_reason": str(reason).strip(), "cancelled_at": time.time()})
    updated = execution_store.update_draft_review(plan_id, {
        "draft_status": "cancelled",
        "cancelled_at": time.time(),
        "revision_history": history,
    })
    return {**updated, "plan": execution_store.get_plan(plan_id)}


def _artifact_project_id(artifact: Dict[str, Any], *extra: Any) -> str:
    return resolve_project_id(artifact.get("project_id"), *extra)


def _artifact_payload(
    *,
    clarification: Dict[str, Any],
    artifact: Dict[str, Any],
    version: int,
    status: str,
    revision_history: list[Dict[str, Any]],
) -> Dict[str, Any]:
    structured = artifact["payload"]
    return {
        "object": "hermes.dev_plan_artifact",
        "plan_artifact_id": f"devpart-{uuid.uuid4().hex[:10]}",
        "clarification_id": clarification["clarification_id"],
        "project_id": clarification.get("project_id"),
        "session_id": clarification.get("session_id"),
        "status": status,
        "version": version,
        "source": artifact["source"],
        "title": structured["title"],
        "markdown": _render_markdown(structured),
        "payload": structured,
        "revision_history": revision_history,
        "superseded_by": None,
        "approved_at": None,
        "cancelled_at": None,
        "warning": artifact.get("warning"),
    }


def _generate_artifact(
    clarification: Dict[str, Any],
    *,
    previous: Optional[Dict[str, Any]] = None,
    feedback: Optional[str] = None,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for _ in range(2):
        try:
            payload = _generate_artifact_with_llm(clarification, previous=previous, feedback=feedback)
            validated = _validate_artifact_payload(payload)
            validated, warnings = _validate_artifact_criteria(validated, clarification)
            warning = _criteria_warning_text(warnings)
            if warning:
                validated["warning"] = warning
            return {"source": "llm", "payload": validated, "warning": warning}
        except Exception as exc:
            last_error = exc
    try:
        raise last_error or ValueError("unknown artifact generation error")
    except Exception as exc:
        fallback = _fallback_artifact_payload(clarification, previous=previous, feedback=feedback)
        fallback["warning"] = f"LLM plan artifact generation failed; using deterministic fallback artifact: {exc}"
        return {"source": "fallback", "payload": fallback, "warning": fallback["warning"]}


def _generate_artifact_with_llm(
    clarification: Dict[str, Any],
    *,
    previous: Optional[Dict[str, Any]],
    feedback: Optional[str],
) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "Generate a rich but non-executing implementation planning artifact. "
                "Return only valid JSON matching the provided schema. "
                "Do not create worker tasks, do not say work has started, and do not approve execution. "
                "Make the plan specific to the vision and clarification answers. "
                "Artifact-level acceptance_criteria must be structured verification objects. "
                "verification_detail for any machine_checkable: true criterion MUST match one of these exact "
                f"command shapes: {'; '.join(ALLOWED_VERIFICATION_COMMAND_SHAPES)}. "
                "Reference only files/paths present in the provided repository grounding; do not invent file paths. "
                "Prefer a whole-suite or directory-level command when unsure of an exact file. "
                "If no allowlisted command fits, set machine_checkable: false and describe a manual check instead. "
                "Do not change execution task acceptance criteria; task drafting still uses string criteria later."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "clarification": _clarification_context(clarification),
                "previous_artifact": previous.get("payload") if previous else None,
                "revision_feedback": feedback,
                "repository_grounding_paths": _grounding_paths(clarification.get("grounding_provenance")),
            }, ensure_ascii=False),
        },
    ]
    kwargs = {
        "task": "dev_plan_artifact",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 3200,
        "timeout": 60,
    }
    try:
        response = call_llm(
            **kwargs,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dev_plan_artifact",
                        "schema": ARTIFACT_JSON_SCHEMA,
                        "strict": True,
                    },
                },
            },
        )
    except Exception as exc:
        error_text = str(exc).lower()
        if "response_format" not in error_text and "json_schema" not in error_text and "unsupported" not in error_text:
            raise
        response = call_llm(**kwargs)
    content = str(response.choices[0].message.content or "").strip()
    return _extract_json(content)


def _generate_execution_tasks(
    artifact: Dict[str, Any],
    *,
    current_tasks: Optional[list[Dict[str, Any]]] = None,
    feedback: Optional[str] = None,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for _ in range(2):
        try:
            payload = _generate_execution_tasks_with_llm(artifact, current_tasks=current_tasks, feedback=feedback)
            return {"source": "llm", "tasks": _validate_execution_tasks(payload, artifact), "warning": None}
        except Exception as exc:
            last_error = exc
    tasks = _fallback_execution_tasks(artifact)
    return {
        "source": "fallback",
        "tasks": tasks,
        "warning": f"LLM task drafting failed; using deterministic fallback tasks: {last_error}",
    }


def _generate_execution_tasks_with_llm(
    artifact: Dict[str, Any],
    *,
    current_tasks: Optional[list[Dict[str, Any]]],
    feedback: Optional[str],
) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "Convert an approved planning artifact into a draft Dev execution plan task list. "
                "Return only valid JSON matching the provided schema. "
                "Do not launch workers, do not say implementation has started, and do not ask for approvals. "
                "Use workspace.implement for implementation tasks and workspace.inspect for validation or inspection tasks."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "artifact": {
                    "plan_artifact_id": artifact.get("plan_artifact_id"),
                    "title": artifact.get("title"),
                    "project_id": _artifact_project_id(artifact),
                    "markdown": artifact.get("markdown"),
                    "payload": artifact.get("payload") or {},
                },
                "current_tasks": [
                    _task_revision_snapshot(task)
                    for task in (current_tasks or [])
                ],
                "revision_feedback": feedback,
                "task_shape": {
                    "fields": ["goal", "prompt", "profile_id", "project_id", "permissions", "dependencies", "acceptance_criteria"],
                    "constraints": [
                        "Every prompt must include product intent, relevant slice, acceptance criteria, non-goals, risks, and Worker Output Contract v2 instruction.",
                        "Implementation tasks use profile_id workspace.implement and permissions edit.",
                        "Validation tasks use profile_id workspace.inspect and permissions read_only.",
                    ],
                },
            }, ensure_ascii=False),
        },
    ]
    kwargs = {
        "task": "dev_plan_artifact_execution_tasks",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 3200,
        "timeout": 60,
    }
    try:
        response = call_llm(
            **kwargs,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dev_execution_tasks_from_artifact",
                        "schema": EXECUTION_TASKS_JSON_SCHEMA,
                        "strict": True,
                    },
                },
            },
        )
    except Exception as exc:
        error_text = str(exc).lower()
        if "response_format" not in error_text and "json_schema" not in error_text and "unsupported" not in error_text:
            raise
        response = call_llm(**kwargs)
    content = str(response.choices[0].message.content or "").strip()
    return _extract_json(content)


def _validate_execution_tasks(payload: Dict[str, Any], artifact: Dict[str, Any]) -> list[Dict[str, Any]]:
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("Execution task draft requires tasks array")
    tasks: list[Dict[str, Any]] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue
        goal = str(item.get("goal") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not goal or not prompt:
            continue
        profile_id = str(item.get("profile_id") or "workspace.implement").strip()
        permissions = str(item.get("permissions") or ("read_only" if profile_id.endswith(".inspect") else "edit")).strip()
        task = {
            "goal": goal[:180],
            "prompt": _task_prompt_with_contract(prompt, artifact),
            "profile_id": profile_id,
            "project_id": resolve_project_id(item.get("project_id"), artifact.get("project_id")),
            "permissions": permissions,
            "dependencies": _string_list(item.get("dependencies")),
            "acceptance_criteria": _string_list(item.get("acceptance_criteria")) or _artifact_acceptance(artifact),
            "source_plan_artifact_id": artifact.get("plan_artifact_id"),
        }
        tasks.append(task)
    if not tasks:
        raise ValueError("Execution task draft produced no usable tasks")
    return tasks[:8]


def _fallback_execution_tasks(artifact: Dict[str, Any]) -> list[Dict[str, Any]]:
    payload = artifact.get("payload") or {}
    tasks: list[Dict[str, Any]] = []
    for index, item in enumerate(_slice_list(payload.get("implementation_slices")), start=1):
        tasks.append(_task_from_slice(
            artifact,
            item,
            index=index,
            profile_id="workspace.implement",
            permissions="edit",
            prefix="Implement",
        ))
    dependency_goals = [task["goal"] for task in tasks]
    for index, item in enumerate(_slice_list(payload.get("validation_slices")), start=1):
        task = _task_from_slice(
            artifact,
            item,
            index=index,
            profile_id="workspace.inspect",
            permissions="read_only",
            prefix="Validate",
        )
        task["dependencies"] = dependency_goals
        tasks.append(task)
    if not tasks:
        tasks.append({
            "goal": f"Implement {artifact.get('title') or 'approved planning artifact'}"[:180],
            "prompt": _task_prompt_with_contract(_artifact_context_prompt(artifact, "Implement the approved planning artifact."), artifact),
            "profile_id": "workspace.implement",
            "project_id": _artifact_project_id(artifact),
            "permissions": "edit",
            "dependencies": [],
            "acceptance_criteria": _artifact_acceptance(artifact),
            "source_plan_artifact_id": artifact.get("plan_artifact_id"),
        })
    return tasks[:8]


def _task_from_slice(
    artifact: Dict[str, Any],
    item: Dict[str, str],
    *,
    index: int,
    profile_id: str,
    permissions: str,
    prefix: str,
) -> Dict[str, Any]:
    title = item.get("title") or f"{prefix} slice {index}"
    description = item.get("description") or title
    prompt = _artifact_context_prompt(
        artifact,
        f"{prefix} this slice:\n\n{title}\n\n{description}",
    )
    return {
        "goal": f"{prefix}: {title}"[:180],
        "prompt": _task_prompt_with_contract(prompt, artifact),
        "profile_id": profile_id,
        "project_id": _artifact_project_id(artifact),
        "permissions": permissions,
        "dependencies": [],
        "acceptance_criteria": _artifact_acceptance(artifact),
        "source_plan_artifact_id": artifact.get("plan_artifact_id"),
    }


def _artifact_context_prompt(artifact: Dict[str, Any], task_instruction: str) -> str:
    payload = artifact.get("payload") or {}
    sections = [
        f"Approved planning artifact: {artifact.get('title')}",
        f"Artifact id: {artifact.get('plan_artifact_id')}",
        f"Product intent: {payload.get('product_intent') or payload.get('overview') or artifact.get('title')}",
        f"Task instruction: {task_instruction}",
    ]
    if payload.get("scope"):
        sections.append("Scope:\n" + "\n".join(f"- {item}" for item in _string_list(payload.get("scope"))))
    if payload.get("non_goals"):
        sections.append("Non-goals:\n" + "\n".join(f"- {item}" for item in _string_list(payload.get("non_goals"))))
    if payload.get("risks"):
        sections.append("Risks:\n" + "\n".join(f"- {item}" for item in _string_list(payload.get("risks"))))
    criteria = _artifact_acceptance(artifact)
    if criteria:
        sections.append("Acceptance criteria:\n" + "\n".join(f"- {item}" for item in criteria))
    return "\n\n".join(sections)


def _task_prompt_with_contract(prompt: str, artifact: Dict[str, Any]) -> str:
    base = str(prompt or "").strip()
    if "Approved planning artifact:" not in base:
        base = _artifact_context_prompt(artifact, base)
    return append_worker_output_contract(base)


def _task_revision_snapshot(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "goal": task.get("goal"),
        "prompt": task.get("prompt"),
        "profile_id": task.get("profile_id"),
        "project_id": task.get("project_id"),
        "dependencies": task.get("dependencies") or [],
        "acceptance_criteria": task.get("acceptance_criteria") or [],
        "status": task.get("status"),
    }


def _artifact_acceptance(artifact: Dict[str, Any]) -> list[str]:
    payload = artifact.get("payload") or {}
    return acceptance_criteria_to_strings(payload.get("acceptance_criteria")) or ["Complete the assigned slice and report verification evidence."]


def _artifact_vision_brief(artifact: Dict[str, Any]) -> str:
    payload = artifact.get("payload") or {}
    return "\n\n".join([
        str(payload.get("overview") or artifact.get("title") or "Approved planning artifact"),
        "",
        str(artifact.get("markdown") or "").strip(),
    ]).strip()


def _clarification_context(clarification: Dict[str, Any]) -> Dict[str, Any]:
    answers = []
    questions_by_id = {
        item.get("question_id"): item
        for item in clarification.get("questions") or []
    }
    for item in clarification.get("answers") or []:
        question = questions_by_id.get(item.get("question_id")) or {}
        answers.append({
            "question": item.get("question_prompt") or question.get("prompt"),
            "answer": item.get("answer_text") or item.get("option_label"),
            "skipped": bool(item.get("skipped")),
        })
    return {
        "project_id": clarification.get("project_id"),
        "project_context": clarification.get("project_context") or {},
        "vision_brief": clarification.get("vision_brief"),
        "clarified_brief": clarification.get("clarified_brief"),
        "grounding": clarification.get("grounding") or {},
        "grounding_provenance": clarification.get("grounding_provenance") or [],
        "answers": answers,
    }


def _extract_json(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Plan artifact JSON root must be an object")
    return parsed


def _validate_artifact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    title = str(payload.get("title") or "").strip()
    if not title:
        title = _title_from_vision(
            str(
                payload.get("overview")
                or payload.get("product_intent")
                or payload.get("recommended_next_action")
                or "Planning artifact"
            )
        )
    normalized = {
        "title": title,
        "overview": _required_text(payload, "overview"),
        "product_intent": _required_text(payload, "product_intent"),
        "scope": _string_list(payload.get("scope")),
        "non_goals": _string_list(payload.get("non_goals")),
        "assumptions": _string_list(payload.get("assumptions")),
        "user_workflow": _string_list(payload.get("user_workflow")),
        "implementation_slices": _slice_list(payload.get("implementation_slices")),
        "validation_slices": _slice_list(payload.get("validation_slices")),
        "acceptance_criteria": normalize_acceptance_criteria(payload.get("acceptance_criteria")),
        "risks": _string_list(payload.get("risks")),
        "open_questions": _string_list(payload.get("open_questions")),
        "recommended_next_action": _required_text(payload, "recommended_next_action"),
    }
    if not normalized["implementation_slices"]:
        raise ValueError("Plan artifact requires implementation_slices")
    if not normalized["acceptance_criteria"]:
        raise ValueError("Plan artifact requires acceptance_criteria")
    return normalized


def _validate_artifact_criteria(
    payload: Dict[str, Any],
    clarification: Dict[str, Any],
) -> tuple[Dict[str, Any], list[str]]:
    criteria, warnings = validate_and_downgrade_criteria(
        payload.get("acceptance_criteria"),
        repo_roots=_repo_roots_from_project_context(clarification.get("project_context") or {}),
    )
    if not warnings:
        return payload, []
    updated = dict(payload)
    updated["acceptance_criteria"] = criteria
    return updated, warnings


def _criteria_warning_text(warnings: list[str]) -> Optional[str]:
    if not warnings:
        return None
    return "Acceptance criteria downgraded: " + " ".join(warnings)


def _fallback_artifact_payload(
    clarification: Dict[str, Any],
    *,
    previous: Optional[Dict[str, Any]],
    feedback: Optional[str],
) -> Dict[str, Any]:
    context = _clarification_context(clarification)
    answers = [item for item in context["answers"] if not item.get("skipped")]
    skipped = [item for item in context["answers"] if item.get("skipped")]
    title = _title_from_vision(str(context.get("vision_brief") or "Planning artifact"))
    project_context = context.get("project_context") or {}
    scope = [
        f"{item.get('question')}: {item.get('answer')}"
        for item in answers
        if item.get("question") and item.get("answer")
    ]
    if project_context.get("vision"):
        scope.insert(0, f"Project vision: {project_context.get('vision')}")
    if feedback:
        scope.append(f"Revision feedback to incorporate: {feedback}")
    assumptions = []
    if project_context.get("project_name"):
        assumptions.append(f"Project: {project_context.get('project_name')}")
    assumptions.extend(
        f"Felipe selected: {item.get('answer')}"
        for item in answers
        if item.get("answer")
    )
    return {
        "title": title,
        "overview": str(context.get("vision_brief") or "").strip() or title,
        "product_intent": "Turn the clarified vision into a reviewable plan artifact before any implementation starts.",
        "scope": scope[:8] or ["Create a bounded v1 plan from the clarification answers."],
        "non_goals": ["Do not create or launch a Dev execution plan in Phase 28."],
        "assumptions": assumptions[:8],
        "user_workflow": [
            "Felipe reviews the generated plan artifact in the right-side planning panel.",
            "Felipe can provide feedback to create a revised artifact version.",
            "Felipe can approve the artifact as ready for the future build transition.",
        ],
        "implementation_slices": [
            {"title": "Define the artifact", "description": "Capture scope, non-goals, assumptions, slices, risks, and acceptance criteria."},
            {"title": "Review and revise", "description": "Support feedback-based revisions without mutating execution plans or launching workers."},
            {"title": "Prepare for build handoff", "description": "Mark approved artifacts as ready for the future Phase 29 build/create-plan transition."},
        ],
        "validation_slices": [
            {"title": "No execution side effects", "description": "Confirm artifact creation does not create Dev execution plans or worker sessions."},
            {"title": "Versioned revision", "description": "Confirm revision creates a new artifact version and supersedes the prior version."},
        ],
        "acceptance_criteria": [
            {
                "statement": "The plan artifact is durable and can be retrieved after refresh.",
                "verification_method": "test",
                "verification_detail": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -k dev_plan_artifact",
                "machine_checkable": True,
            },
            {
                "statement": "The artifact is visible in the right-side planning panel.",
                "verification_method": "manual",
                "verification_detail": "Review the Oryn Workspace planning panel.",
                "machine_checkable": False,
            },
            {
                "statement": "Approval does not create or launch any worker execution.",
                "verification_method": "test",
                "verification_detail": "scripts/run_tests.sh tests/gateway/test_api_server_runs.py -- -k dev_plan_artifact",
                "machine_checkable": True,
            },
        ],
        "risks": ["The fallback artifact is useful for continuity but may be less specific than an LLM-generated artifact."],
        "open_questions": [item.get("question") for item in skipped if item.get("question")],
        "recommended_next_action": "Review the plan artifact, revise if needed, then approve it only when ready for the future build transition.",
    }


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines = [f"# {payload['title']}", "", payload["overview"], ""]
    sections = [
        ("Product Intent", payload.get("product_intent")),
        ("Scope", payload.get("scope")),
        ("Non-Goals", payload.get("non_goals")),
        ("Assumptions", payload.get("assumptions")),
        ("User Workflow", payload.get("user_workflow")),
        ("Implementation Slices", payload.get("implementation_slices")),
        ("Validation Slices", payload.get("validation_slices")),
        ("Acceptance Criteria", payload.get("acceptance_criteria")),
        ("Risks", payload.get("risks")),
        ("Open Questions", payload.get("open_questions")),
        ("Recommended Next Action", payload.get("recommended_next_action")),
    ]
    for title, value in sections:
        if value in (None, "", []):
            continue
        lines.extend([f"## {title}", ""])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("statement"):
                    detail = item.get("verification_detail") or "Review manually."
                    method = item.get("verification_method") or "manual"
                    lines.append(f"- {item.get('statement')} _(verify: {method} - {detail})_")
                elif isinstance(item, dict):
                    lines.append(f"- **{item.get('title', 'Slice')}**: {item.get('description', '')}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines).strip()


def _row_values(payload: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload["plan_artifact_id"],
        payload["clarification_id"],
        payload.get("project_id"),
        payload.get("session_id"),
        payload["status"],
        int(payload.get("version") or 1),
        payload["source"],
        payload["title"],
        payload["markdown"],
        json.dumps(payload.get("payload") or {}, ensure_ascii=False),
        json.dumps(payload.get("revision_history") or [], ensure_ascii=False),
        payload.get("superseded_by"),
        float(payload["created_at"]),
        float(payload["updated_at"]),
        payload.get("approved_at"),
        payload.get("cancelled_at"),
    )


def _row_to_payload(row: sqlite3.Row) -> Dict[str, Any]:
    structured = json.loads(row["payload"] or "{}")
    payload = {
        "object": "hermes.dev_plan_artifact",
        "plan_artifact_id": row["plan_artifact_id"],
        "clarification_id": row["clarification_id"],
        "project_id": row["project_id"],
        "session_id": row["session_id"],
        "status": row["status"],
        "version": int(row["version"] or 1),
        "source": row["source"],
        "title": row["title"],
        "markdown": row["markdown"],
        "payload": structured,
        "revision_history": json.loads(row["revision_history"] or "[]"),
        "superseded_by": row["superseded_by"],
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "approved_at": row["approved_at"],
        "cancelled_at": row["cancelled_at"],
    }
    warning = structured.get("warning")
    if warning:
        payload["warning"] = warning
    return payload


def _build_row_to_payload(row: sqlite3.Row) -> Dict[str, Any]:
    payload = {
        "object": "hermes.dev_plan_artifact_build",
        "build_id": row["build_id"],
        "plan_artifact_id": row["plan_artifact_id"],
        "plan_id": row["plan_id"],
        "status": row["status"],
        "source": row["source"],
        "task_count": int(row["task_count"] or 0),
        "created_at": float(row["created_at"]),
        "payload": json.loads(row["payload"] or "{}"),
    }
    return payload


def _required_text(payload: Dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Plan artifact missing {key}")
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _slice_list(value: Any) -> list[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    slices = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        if title and description:
            slices.append({"title": title, "description": description})
    return slices


def _title_from_vision(vision: str) -> str:
    text = re.sub(r"\s+", " ", vision).strip()
    if not text:
        return "Planning Artifact"
    text = re.sub(r"^(i want|we need|build|add|create)\s+", "", text, flags=re.IGNORECASE)
    return text[:72].strip(" .") or "Planning Artifact"


def _repo_roots_from_project_context(project_context: Dict[str, Any]) -> list[str]:
    roots: list[str] = []
    repositories = project_context.get("repositories")
    if isinstance(repositories, list):
        for repo in repositories:
            if isinstance(repo, dict):
                path = str(repo.get("path") or "").strip()
                if path:
                    roots.append(path)
    return roots


def _grounding_paths(provenance: Any) -> list[str]:
    paths: list[str] = []
    if not isinstance(provenance, list):
        return paths
    for item in provenance:
        if isinstance(item, str):
            path = item.strip()
            if path:
                paths.append(path)
            continue
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("repo_path") or "").strip()
        if path:
            paths.append(path)
    return paths
