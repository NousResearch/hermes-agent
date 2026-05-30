from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from hermes_cli.workflow.orchestrator import (
    Subtask,
    TaskAnalysis,
    WorkerOutput,
    WorkflowConfig,
    _redact_payload,
)


class RunStore:
    """SQLite state for resumable workflow runs."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def from_config(cls, config: WorkflowConfig) -> "RunStore":
        configured = config.state_db.strip()
        if configured:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = config.path.parent / path
        else:
            try:
                from hermes_constants import get_hermes_home

                path = get_hermes_home() / "workflow_runs.db"
            except Exception:
                path = Path.home() / ".hermes" / "workflow_runs.db"
        return cls(path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    original_request TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    codex_plan INTEGER NOT NULL,
                    contest INTEGER NOT NULL DEFAULT 0,
                    cross_review INTEGER NOT NULL DEFAULT 0,
                    analysis_json TEXT NOT NULL,
                    final_response TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    pid INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_run_column(conn, "contest", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_run_column(conn, "cross_review", "INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_subtasks (
                    run_id TEXT NOT NULL,
                    subtask_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_type TEXT NOT NULL DEFAULT '',
                    output_json TEXT NOT NULL DEFAULT '',
                    evaluation_json TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, subtask_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    event TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_run_column(
        self, conn: sqlite3.Connection, name: str, definition: str
    ) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(workflow_runs)").fetchall()
        }
        if name not in columns:
            conn.execute(f"ALTER TABLE workflow_runs ADD COLUMN {name} {definition}")

    def create_run(
        self,
        *,
        run_id: str,
        original_request: str,
        analysis: TaskAnalysis,
        mode: str,
        dry_run: bool,
        codex_plan: bool,
        contest: bool = False,
        cross_review: bool = False,
        status: str = "queued",
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_runs (
                    run_id, original_request, status, mode, dry_run, codex_plan,
                    contest, cross_review, analysis_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    original_request,
                    status,
                    mode,
                    int(dry_run),
                    int(codex_plan),
                    int(contest),
                    int(cross_review),
                    json.dumps(analysis.to_dict(), ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        final_response: str | None = None,
        error: str | None = None,
        pid: int | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if final_response is not None:
            fields.append("final_response = ?")
            values.append(final_response)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if pid is not None:
            fields.append("pid = ?")
            values.append(pid)
        fields.append("updated_at = ?")
        values.append(_now())
        values.append(run_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE workflow_runs SET {', '.join(fields)} WHERE run_id = ?",
                values,
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def save_subtask(self, run_id: str, subtask: Subtask, status: str = "pending") -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workflow_subtasks (
                    run_id, subtask_id, payload_json, status, worker_type,
                    output_json, evaluation_json, updated_at
                )
                VALUES (
                    ?, ?, ?, ?,
                    COALESCE((SELECT worker_type FROM workflow_subtasks WHERE run_id=? AND subtask_id=?), ?),
                    COALESCE((SELECT output_json FROM workflow_subtasks WHERE run_id=? AND subtask_id=?), ''),
                    COALESCE((SELECT evaluation_json FROM workflow_subtasks WHERE run_id=? AND subtask_id=?), ''),
                    ?
                )
                """,
                (
                    run_id,
                    subtask.id,
                    json.dumps(subtask.to_dict(), ensure_ascii=False),
                    status,
                    run_id,
                    subtask.id,
                    subtask.worker_type or "",
                    run_id,
                    subtask.id,
                    run_id,
                    subtask.id,
                    now,
                ),
            )

    def update_subtask_status(
        self, run_id: str, subtask_id: str, status: str, worker_type: str | None = None
    ) -> None:
        fields = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status, _now()]
        if worker_type is not None:
            fields.insert(1, "worker_type = ?")
            values.insert(1, worker_type)
        values.extend([run_id, subtask_id])
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE workflow_subtasks
                SET {', '.join(fields)}
                WHERE run_id = ? AND subtask_id = ?
                """,
                values,
            )

    def save_output(self, run_id: str, output: WorkerOutput) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE workflow_subtasks
                SET status = ?, worker_type = ?, output_json = ?,
                    evaluation_json = ?, updated_at = ?
                WHERE run_id = ? AND subtask_id = ?
                """,
                (
                    "completed" if output.evaluation.passed else "failed",
                    output.worker_type,
                    json.dumps(output.to_dict(), ensure_ascii=False),
                    json.dumps(output.evaluation.to_dict(), ensure_ascii=False),
                    _now(),
                    run_id,
                    output.subtask.id,
                ),
            )

    def load_subtasks(self, run_id: str) -> list[Subtask]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM workflow_subtasks
                WHERE run_id = ?
                ORDER BY rowid
                """,
                (run_id,),
            ).fetchall()
        return [Subtask.from_dict(json.loads(row["payload_json"])) for row in rows]

    def load_outputs(self, run_id: str) -> list[WorkerOutput]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT output_json
                FROM workflow_subtasks
                WHERE run_id = ? AND output_json != ''
                ORDER BY rowid
                """,
                (run_id,),
            ).fetchall()
        return [WorkerOutput.from_dict(json.loads(row["output_json"])) for row in rows]

    def append_event(self, run_id: str, event: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_events (run_id, ts, event, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    _now(),
                    event,
                    json.dumps(_redact_payload(payload), ensure_ascii=False, sort_keys=True),
                ),
            )

    def events(self, run_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT ts, event, payload_json FROM workflow_events "
            "WHERE run_id = ? ORDER BY id"
        )
        params: tuple[Any, ...] = (run_id,)
        if limit is not None:
            sql += " DESC LIMIT ?"
            params = (run_id, limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        events = [
            {
                "ts": row["ts"],
                "event": row["event"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]
        if limit is not None:
            events.reverse()
        return events


def analysis_from_record(record: dict[str, Any]) -> TaskAnalysis:
    data = json.loads(record.get("analysis_json") or "{}")
    return TaskAnalysis(
        task_type=str(data.get("task_type") or "planning"),
        complexity=str(data.get("complexity") or "medium"),
        execute_directly=bool(data.get("execute_directly", False)),
        rationale=str(data.get("rationale") or "Loaded from workflow state."),
    )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
