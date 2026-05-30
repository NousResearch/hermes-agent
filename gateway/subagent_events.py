"""Persistent subagent event history backed by Hermes state.db."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gateway.dev_control.events import normalize_subagent_event
from gateway.dev_control.worker_output_contract import parse_worker_output_contract, worker_output_contract_score
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS subagent_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    session_id TEXT,
    run_id TEXT,
    subagent_id TEXT NOT NULL,
    parent_id TEXT,
    runtime TEXT,
    ao_session_id TEXT,
    event_type TEXT NOT NULL,
    status TEXT,
    goal TEXT,
    summary TEXT,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_subagent_events_session
    ON subagent_events(session_id, event_id);
CREATE INDEX IF NOT EXISTS idx_subagent_events_run
    ON subagent_events(run_id, event_id);
CREATE INDEX IF NOT EXISTS idx_subagent_events_ao_session
    ON subagent_events(ao_session_id, event_id);

CREATE TABLE IF NOT EXISTS ao_session_prompts (
    ao_session_id TEXT PRIMARY KEY,
    project_id TEXT,
    prompt TEXT NOT NULL,
    goal TEXT,
    issue_id TEXT,
    branch TEXT,
    agent TEXT,
    model TEXT,
    reasoning_effort TEXT,
    launch_profile_id TEXT,
    launch_plan_id TEXT,
    launch_task_id TEXT,
    permissions TEXT,
    acceptance_criteria TEXT,
    runtime_selection TEXT,
    selected_runtime TEXT,
    runtime_selection_reason TEXT,
    runtime_fallback_reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class SubagentEventStore:
    """Small append-only event store for normalized ``subagent.*`` payloads."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="state.db")
        self._lock = threading.Lock()
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)
            self._ensure_prompt_metadata_columns()

    def close(self) -> None:
        self._conn.close()

    def append_event(self, payload: Dict[str, Any], *, session_id: Optional[str] = None) -> Dict[str, Any]:
        payload = self._with_output_contract_fields(payload)
        event = normalize_subagent_event(payload, session_id=session_id)
        created_at = float(event["created_at"])

        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO subagent_events (
                    created_at, session_id, run_id, subagent_id, parent_id,
                    runtime, ao_session_id, event_type, status, goal, summary, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    event.get("session_id"),
                    event.get("run_id"),
                    event.get("subagent_id"),
                    event.get("parent_id"),
                    event.get("runtime"),
                    event.get("ao_session_id"),
                    event.get("event") or "subagent.progress",
                    event.get("status"),
                    event.get("goal"),
                    event.get("summary"),
                    json.dumps(event, ensure_ascii=False),
                ),
            )
            event_id = int(cur.lastrowid)

        event["event_id"] = event_id
        try:
            from gateway.dev_control.laminar_exporter import export_subagent_event

            export_subagent_event(event)
        except Exception:
            pass
        return event

    def list_events(
        self,
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        subagent_id: Optional[str] = None,
        ao_session_id: Optional[str] = None,
        runtime: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 500,
    ) -> list[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if subagent_id:
            clauses.append("subagent_id = ?")
            params.append(subagent_id)
        if ao_session_id:
            clauses.append("ao_session_id = ?")
            params.append(ao_session_id)
        if runtime:
            clauses.append("runtime = ?")
            params.append(runtime)
        if status:
            clauses.append("LOWER(status) = LOWER(?)")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 500), 2000)))
        rows = self._conn.execute(
            f"""
            SELECT event_id, created_at, payload
            FROM subagent_events
            {where}
            ORDER BY event_id ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def upsert_ao_prompt(
        self,
        *,
        ao_session_id: str,
        project_id: Optional[str],
        prompt: str,
        goal: Optional[str],
        issue_id: Optional[str],
        branch: Optional[str],
        agent: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        launch_profile_id: Optional[str] = None,
        launch_plan_id: Optional[str] = None,
        launch_task_id: Optional[str] = None,
        permissions: Optional[str] = None,
        acceptance_criteria: Optional[Any] = None,
        runtime_selection: Optional[Any] = None,
        selected_runtime: Optional[str] = None,
        runtime_selection_reason: Optional[str] = None,
        runtime_fallback_reason: Optional[str] = None,
    ) -> None:
        now = time.time()
        criteria_payload = json.dumps(acceptance_criteria or [], ensure_ascii=False)
        runtime_selection_payload = json.dumps(runtime_selection or {}, ensure_ascii=False)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO ao_session_prompts (
                    ao_session_id, project_id, prompt, goal, issue_id,
                    branch, agent, model, reasoning_effort, launch_profile_id,
                    launch_plan_id, launch_task_id, permissions, acceptance_criteria,
                    runtime_selection, selected_runtime, runtime_selection_reason,
                    runtime_fallback_reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ao_session_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    prompt = excluded.prompt,
                    goal = excluded.goal,
                    issue_id = excluded.issue_id,
                    branch = excluded.branch,
                    agent = excluded.agent,
                    model = excluded.model,
                    reasoning_effort = excluded.reasoning_effort,
                    launch_profile_id = excluded.launch_profile_id,
                    launch_plan_id = excluded.launch_plan_id,
                    launch_task_id = excluded.launch_task_id,
                    permissions = excluded.permissions,
                    acceptance_criteria = excluded.acceptance_criteria,
                    runtime_selection = excluded.runtime_selection,
                    selected_runtime = excluded.selected_runtime,
                    runtime_selection_reason = excluded.runtime_selection_reason,
                    runtime_fallback_reason = excluded.runtime_fallback_reason,
                    updated_at = excluded.updated_at
                """,
                (
                    ao_session_id, project_id, prompt, goal, issue_id, branch,
                    agent, model, reasoning_effort, launch_profile_id,
                    launch_plan_id, launch_task_id, permissions, criteria_payload,
                    runtime_selection_payload, selected_runtime, runtime_selection_reason,
                    runtime_fallback_reason,
                    now, now,
                ),
            )

    def get_ao_prompt(self, ao_session_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT ao_session_id, project_id, prompt, goal, issue_id, branch, agent, model,
                   reasoning_effort, launch_profile_id, launch_plan_id, launch_task_id,
                   permissions, acceptance_criteria, runtime_selection, selected_runtime,
                   runtime_selection_reason, runtime_fallback_reason, created_at, updated_at
            FROM ao_session_prompts
            WHERE ao_session_id = ?
            """,
            (ao_session_id,),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        try:
            result["acceptance_criteria"] = json.loads(result.get("acceptance_criteria") or "[]")
        except Exception:
            result["acceptance_criteria"] = []
        try:
            result["runtime_selection"] = json.loads(result.get("runtime_selection") or "{}")
        except Exception:
            result["runtime_selection"] = {}
        return result

    def _ensure_prompt_metadata_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(ao_session_prompts)").fetchall()
        }
        for name in (
            "reasoning_effort",
            "launch_profile_id",
            "launch_plan_id",
            "launch_task_id",
            "permissions",
            "acceptance_criteria",
            "runtime_selection",
            "selected_runtime",
            "runtime_selection_reason",
            "runtime_fallback_reason",
        ):
            if name not in columns:
                self._conn.execute(f"ALTER TABLE ao_session_prompts ADD COLUMN {name} TEXT")

    def latest_event_for_ao_session(self, ao_session_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT event_id, created_at, payload
            FROM subagent_events
            WHERE ao_session_id = ?
            ORDER BY event_id DESC
            LIMIT 1
            """,
            (ao_session_id,),
        ).fetchone()
        return self._row_to_event(row) if row else None

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Dict[str, Any]:
        payload = json.loads(row["payload"])
        payload["event_id"] = int(row["event_id"])
        payload.setdefault("created_at", float(row["created_at"]))
        return payload

    @staticmethod
    def _with_output_contract_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
        event_type = str(payload.get("event") or "").lower()
        if event_type != "subagent.complete" and str(payload.get("status") or "").lower() not in {"completed", "complete", "done", "success", "succeeded"}:
            return payload
        if payload.get("output_contract_status"):
            return payload
        text_parts = [
            str(payload.get("summary") or ""),
            str(payload.get("message") or ""),
            str(payload.get("preview") or ""),
        ]
        output_tail = payload.get("output_tail")
        if isinstance(output_tail, list):
            text_parts.extend(str(item.get("text") if isinstance(item, dict) else item) for item in output_tail)
        elif output_tail:
            text_parts.append(str(output_tail))
        parsed = parse_worker_output_contract("\n".join(text_parts))
        marker = parsed.get("final_marker")
        parsed["output_contract_score"] = worker_output_contract_score(parsed, required_marker=marker)
        merged = dict(payload)
        for key, value in parsed.items():
            if key == "files_read" and merged.get("files_read"):
                continue
            if key == "files_changed":
                merged.setdefault("files_changed", value)
                merged.setdefault("files_written", value)
                continue
            if key == "verification_evidence" and merged.get("verification_evidence"):
                continue
            merged.setdefault(key, value)
        if parsed.get("structured_summary") and not merged.get("summary"):
            merged["summary"] = parsed["structured_summary"]
        return merged


def events_response(events: Iterable[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    data = list(events)
    return {
        "object": "list",
        "data": data,
        "total": len(data),
        **extra,
    }
