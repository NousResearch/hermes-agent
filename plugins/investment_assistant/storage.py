"""SQLite persistence for investment assistant workflows."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from pydantic import BaseModel


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _json_dump(payload: Any) -> str:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _json_load(raw: str | None) -> Any:
    if not raw:
        return None
    return json.loads(raw)


class InvestmentAssistantStore:
    """Small durable store keyed by tenant and workflow session."""

    def __init__(self, path: Path | None = None):
        self.path = path or get_hermes_home() / "investment_assistant" / "workflow.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflow_sessions (
                    session_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    status TEXT NOT NULL,
                    theme TEXT NOT NULL,
                    current_artifact_id TEXT,
                    last_human_action_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_sessions_tenant_updated
                    ON workflow_sessions(tenant_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS workflow_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES workflow_sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_artifacts_session_type
                    ON workflow_artifacts(session_id, type, version DESC);

                CREATE TABLE IF NOT EXISTS human_actions (
                    action_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    prompt_json TEXT NOT NULL,
                    response_schema_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer_json TEXT,
                    created_at TEXT NOT NULL,
                    answered_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES workflow_sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_human_actions_session_status
                    ON human_actions(session_id, status, created_at DESC);

                CREATE TABLE IF NOT EXISTS workflow_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES workflow_sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS workflow_state_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    artifact_ids_json TEXT NOT NULL,
                    error_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_ms INTEGER,
                    FOREIGN KEY(session_id) REFERENCES workflow_sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_state_runs_session_started
                    ON workflow_state_runs(session_id, started_at ASC);

                CREATE INDEX IF NOT EXISTS idx_workflow_state_runs_session_state
                    ON workflow_state_runs(session_id, state, started_at DESC);
                """
            )

    def create_session(self, tenant_id: str, theme: str, state: str, status: str) -> dict[str, Any]:
        now = utc_now()
        session_id = new_id("iaw")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_sessions (
                    session_id, tenant_id, workflow_type, state, status, theme,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tenant_id,
                    "portfolio_construction",
                    state,
                    status,
                    theme,
                    now,
                    now,
                ),
            )
        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM workflow_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def latest_session(self, tenant_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM workflow_sessions
                WHERE tenant_id = ?
                  AND status IN ('active', 'waiting_for_human', 'completed', 'failed')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (tenant_id,),
            ).fetchone()
        return dict(row) if row else None

    def latest_session_by_tenant_prefix(self, tenant_prefix: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM workflow_sessions
                WHERE tenant_id LIKE ?
                  AND status IN ('active', 'waiting_for_human', 'completed', 'failed')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (f"{tenant_prefix}%",),
            ).fetchone()
        return dict(row) if row else None

    def update_session(self, session_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {
            "state",
            "status",
            "current_artifact_id",
            "last_human_action_id",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [session_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE workflow_sessions SET {assignments} WHERE session_id = ?",
                values,
            )
        updated = self.get_session(session_id)
        if updated is None:
            raise KeyError(f"Unknown workflow session: {session_id}")
        return updated

    def add_artifact(self, session_id: str, artifact_type: str, payload: Any) -> dict[str, Any]:
        artifact_id = new_id("art")
        now = utc_now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next_version
                FROM workflow_artifacts
                WHERE session_id = ? AND type = ?
                """,
                (session_id, artifact_type),
            ).fetchone()
            version = int(row["next_version"])
            conn.execute(
                """
                INSERT INTO workflow_artifacts (
                    artifact_id, session_id, type, version, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, session_id, artifact_type, version, _json_dump(payload), now),
            )
        return {
            "artifact_id": artifact_id,
            "session_id": session_id,
            "type": artifact_type,
            "version": version,
            "payload": payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload,
            "created_at": now,
        }

    def latest_artifact(self, session_id: str, artifact_type: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM workflow_artifacts
                WHERE session_id = ? AND type = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (session_id, artifact_type),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["payload"] = _json_load(data.pop("payload_json"))
        return data

    def list_artifacts(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM workflow_artifacts
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        artifacts: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["payload"] = _json_load(data.pop("payload_json"))
            artifacts.append(data)
        return artifacts

    def start_state_run(
        self,
        session_id: str,
        state: str,
        input_payload: Any | None = None,
    ) -> dict[str, Any]:
        run_id = new_id("run")
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_state_runs (
                    run_id, session_id, state, status, input_json, output_json,
                    artifact_ids_json, error_json, started_at
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    state,
                    _json_dump(input_payload or {}),
                    _json_dump(None),
                    _json_dump([]),
                    _json_dump(None),
                    now,
                ),
            )
        return self.get_state_run(run_id) or {}

    def finish_state_run(
        self,
        run_id: str,
        *,
        status: str,
        output_payload: Any | None = None,
        artifact_ids: list[str] | None = None,
        error_payload: Any | None = None,
    ) -> dict[str, Any]:
        existing = self.get_state_run(run_id)
        if not existing:
            raise KeyError(f"Unknown workflow state run: {run_id}")
        ended_at = utc_now()
        duration_ms = _duration_ms(existing.get("started_at"), ended_at)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE workflow_state_runs
                SET status = ?, output_json = ?, artifact_ids_json = ?,
                    error_json = ?, ended_at = ?, duration_ms = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    _json_dump(output_payload),
                    _json_dump(artifact_ids or []),
                    _json_dump(error_payload),
                    ended_at,
                    duration_ms,
                    run_id,
                ),
            )
        updated = self.get_state_run(run_id)
        if updated is None:
            raise KeyError(f"Unknown workflow state run: {run_id}")
        return updated

    def get_state_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM workflow_state_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return self._decode_state_run(dict(row))

    def list_state_runs(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM workflow_state_runs
                WHERE session_id = ?
                ORDER BY started_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._decode_state_run(dict(row)) for row in rows]

    def create_human_action(
        self,
        session_id: str,
        state: str,
        kind: str,
        prompt: dict[str, Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        action_id = new_id("act")
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO human_actions (
                    action_id, session_id, state, kind, prompt_json,
                    response_schema_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    action_id,
                    session_id,
                    state,
                    kind,
                    _json_dump(prompt),
                    _json_dump(response_schema),
                    now,
                ),
            )
        self.update_session(session_id, last_human_action_id=action_id)
        return self.get_human_action(action_id) or {}

    def get_human_action(self, action_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM human_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if not row:
            return None
        return self._decode_human_action(dict(row))

    def pending_human_action(
        self,
        session_id: str,
        action_id: str | None = None,
    ) -> dict[str, Any] | None:
        query = """
            SELECT * FROM human_actions
            WHERE session_id = ? AND status = 'pending'
        """
        params: list[Any] = [session_id]
        if action_id:
            query += " AND action_id = ?"
            params.append(action_id)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            return None
        return self._decode_human_action(dict(row))

    def answer_human_action(self, action_id: str, answer: Any) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE human_actions
                SET status = 'answered', answer_json = ?, answered_at = ?
                WHERE action_id = ? AND status = 'pending'
                """,
                (_json_dump(answer), now, action_id),
            )
        action = self.get_human_action(action_id)
        if action is None:
            raise KeyError(f"Unknown human action: {action_id}")
        return action

    def add_event(
        self,
        session_id: str,
        action: str,
        from_state: str | None,
        to_state: str | None,
        payload: Any,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_events (
                    event_id, session_id, action, from_state, to_state,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("evt"),
                    session_id,
                    action,
                    from_state,
                    to_state,
                    _json_dump(payload),
                    utc_now(),
                ),
            )

    def _decode_human_action(self, data: dict[str, Any]) -> dict[str, Any]:
        data["prompt"] = _json_load(data.pop("prompt_json"))
        data["response_schema"] = _json_load(data.pop("response_schema_json"))
        data["answer"] = _json_load(data.pop("answer_json"))
        return data

    def _decode_state_run(self, data: dict[str, Any]) -> dict[str, Any]:
        data["input"] = _json_load(data.pop("input_json"))
        data["output"] = _json_load(data.pop("output_json"))
        data["artifact_ids"] = _json_load(data.pop("artifact_ids_json")) or []
        data["error"] = _json_load(data.pop("error_json"))
        return data


def _duration_ms(started_at: str | None, ended_at: str | None) -> int | None:
    if not started_at or not ended_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
        ended = datetime.fromisoformat(ended_at)
    except ValueError:
        return None
    return max(0, int((ended - started).total_seconds() * 1000))
