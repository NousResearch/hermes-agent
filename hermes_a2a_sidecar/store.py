from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from google.protobuf import json_format

from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import ListTasksRequest, ListTasksResponse, Role, Task, TaskState, TaskStatus
from a2a.helpers import new_text_artifact, new_text_message

from hermes_cli import kanban_db as kb


SCHEMA = """
CREATE TABLE IF NOT EXISTS a2a_tasks (
    a2a_task_id     TEXT PRIMARY KEY,
    context_id      TEXT NOT NULL,
    peer_id         TEXT NOT NULL,
    skill           TEXT NOT NULL,
    kanban_task_id  TEXT,
    correlation_id  TEXT NOT NULL,
    task_json       TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    last_state      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_a2a_tasks_peer ON a2a_tasks(peer_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_a2a_tasks_kanban ON a2a_tasks(kanban_task_id);

CREATE TABLE IF NOT EXISTS a2a_audit_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      INTEGER NOT NULL,
    correlation_id  TEXT NOT NULL,
    peer_id         TEXT NOT NULL,
    a2a_task_id     TEXT,
    kanban_task_id  TEXT,
    event           TEXT NOT NULL,
    metadata_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_a2a_audit_corr ON a2a_audit_events(correlation_id, created_at);
"""

KANBAN_TO_A2A_STATE = {
    "triage": TaskState.TASK_STATE_SUBMITTED,
    "todo": TaskState.TASK_STATE_SUBMITTED,
    "scheduled": TaskState.TASK_STATE_SUBMITTED,
    "ready": TaskState.TASK_STATE_SUBMITTED,
    "running": TaskState.TASK_STATE_WORKING,
    "blocked": TaskState.TASK_STATE_INPUT_REQUIRED,
    "review": TaskState.TASK_STATE_INPUT_REQUIRED,
    "done": TaskState.TASK_STATE_COMPLETED,
    "archived": TaskState.TASK_STATE_CANCELED,
}


def _state_name(state: int) -> str:
    try:
        return TaskState.Name(state)
    except Exception:
        return str(state)


def _task_to_json(task: Task) -> str:
    return json_format.MessageToJson(task, preserving_proto_field_name=False)


def _task_from_json(payload: str) -> Task:
    return json_format.Parse(payload, Task())


def _message(text: str, *, task_id: str, context_id: str) -> Any:
    return new_text_message(
        text,
        context_id=context_id,
        task_id=task_id,
        role=Role.ROLE_AGENT,
    )


class HermesKanbanTaskStore(TaskStore):
    """A2A TaskStore backed by a sidecar SQLite DB and synced from Kanban."""

    def __init__(self, db_path: Path, *, default_board: str = "") -> None:
        self.db_path = db_path
        self.default_board = default_board
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def audit(
        self,
        *,
        event: str,
        correlation_id: str,
        peer_id: str,
        a2a_task_id: str = "",
        kanban_task_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        safe_metadata = metadata or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO a2a_audit_events
                    (created_at, correlation_id, peer_id, a2a_task_id, kanban_task_id, event, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    correlation_id,
                    peer_id,
                    a2a_task_id,
                    kanban_task_id,
                    event,
                    json.dumps(safe_metadata, sort_keys=True, ensure_ascii=False),
                ),
            )
            conn.commit()

    async def save(self, task: Task, context: ServerCallContext) -> None:
        peer_id = _peer_id(context)
        metadata = json_format.MessageToDict(task.metadata) if task.metadata else {}
        kanban_task_id = str(metadata.get("hermes_kanban_id") or "")
        skill = str(metadata.get("skill") or metadata.get("a2a_skill") or "")
        correlation_id = str(metadata.get("correlation_id") or task.id)
        now = int(time.time())
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at, peer_id, skill, kanban_task_id, correlation_id FROM a2a_tasks WHERE a2a_task_id = ?",
                (task.id,),
            ).fetchone()
            created_at = int(existing["created_at"]) if existing else now
            if existing:
                peer_id = peer_id or existing["peer_id"]
                skill = skill or existing["skill"]
                kanban_task_id = kanban_task_id or existing["kanban_task_id"] or ""
                correlation_id = correlation_id or existing["correlation_id"]
            conn.execute(
                """
                INSERT INTO a2a_tasks
                    (a2a_task_id, context_id, peer_id, skill, kanban_task_id, correlation_id,
                     task_json, created_at, updated_at, last_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(a2a_task_id) DO UPDATE SET
                    context_id = excluded.context_id,
                    peer_id = excluded.peer_id,
                    skill = excluded.skill,
                    kanban_task_id = excluded.kanban_task_id,
                    correlation_id = excluded.correlation_id,
                    task_json = excluded.task_json,
                    updated_at = excluded.updated_at,
                    last_state = excluded.last_state
                """,
                (
                    task.id,
                    task.context_id,
                    peer_id,
                    skill,
                    kanban_task_id,
                    correlation_id,
                    _task_to_json(task),
                    created_at,
                    now,
                    _state_name(task.status.state),
                ),
            )
            conn.commit()

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM a2a_tasks WHERE a2a_task_id = ?", (task_id,)
            ).fetchone()
        if not row:
            return None
        task = _task_from_json(row["task_json"])
        task = self._sync_from_kanban(task, row)
        await self.save(task, context)
        return task

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        peer_id = _peer_id(context)
        sql = "SELECT * FROM a2a_tasks"
        values: list[Any] = []
        if peer_id:
            sql += " WHERE peer_id = ?"
            values.append(peer_id)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        values.append(int(params.page_size or 50))
        with self._connect() as conn:
            rows = conn.execute(sql, values).fetchall()
        tasks = [self._sync_from_kanban(_task_from_json(row["task_json"]), row) for row in rows]
        return ListTasksResponse(tasks=tasks, total_size=len(tasks), page_size=len(tasks))

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM a2a_tasks WHERE a2a_task_id = ?", (task_id,))
            conn.commit()

    def _sync_from_kanban(self, task: Task, row: sqlite3.Row) -> Task:
        kanban_task_id = row["kanban_task_id"]
        if not kanban_task_id:
            return task
        board = self.default_board or ""
        try:
            with kb.connect_closing(board=board or None) as conn:
                ktask = kb.get_task(conn, kanban_task_id)
                if not ktask:
                    return task
                state = KANBAN_TO_A2A_STATE.get(ktask.status, TaskState.TASK_STATE_WORKING)
                summary = kb.latest_summary(conn, kanban_task_id) or ktask.result or ""
        except Exception:
            return task

        if task.status.state != state:
            task.status.CopyFrom(TaskStatus(state=state))
        status_text = f"Hermes Kanban task {kanban_task_id}: {ktask.status}"
        if ktask.status in {"blocked", "review"} and ktask.last_failure_error:
            status_text = f"{status_text} — {ktask.last_failure_error}"
        if ktask.status == "done" and summary:
            status_text = summary
            keep = [
                a for a in task.artifacts if a.name not in {"hermes-kanban-result", "hermes-kanban-status"}
            ]
            del task.artifacts[:]
            task.artifacts.extend(keep)
            task.artifacts.append(
                new_text_artifact(
                    "hermes-kanban-result",
                    summary,
                    media_type="text/plain",
                    description="Final Hermes Kanban worker summary.",
                )
            )
        elif ktask.status != "done":
            keep = [
                a for a in task.artifacts if a.name != "hermes-kanban-status"
            ]
            del task.artifacts[:]
            task.artifacts.extend(keep)
            task.artifacts.append(
                new_text_artifact(
                    "hermes-kanban-status",
                    status_text,
                    media_type="text/plain",
                    description="Current Hermes Kanban task state.",
                )
            )
        task.status.message.CopyFrom(
            _message(status_text, task_id=task.id, context_id=task.context_id)
        )
        task.metadata.update(
            {
                "hermes_kanban_id": kanban_task_id,
                "hermes_kanban_status": ktask.status,
                "correlation_id": row["correlation_id"],
                "peer_id": row["peer_id"],
            }
        )
        return task


def _peer_id(context: ServerCallContext) -> str:
    principal = context.state.get("a2a_peer") if context else None
    if principal:
        return str(getattr(principal, "peer_id", ""))
    return ""
