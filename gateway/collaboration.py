"""Collaboration protocol and persistence primitives for gateway sessions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.config import GatewayConfig


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CollaborationJob:
    job_id: str
    requester_session_key: str
    target_session_key: str
    target_agent: str
    task_text: str
    status: str = "pending"
    result_text: str | None = None
    error_reason: str | None = None
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    lineage: list[str] = field(default_factory=list)


@dataclass
class InternalGatewayEvent:
    kind: str
    session_key: str
    job_id: str
    payload: Dict[str, Any]
    created_at: str = field(default_factory=_utcnow)


class CollaborationStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS collaboration_jobs (
                job_id TEXT PRIMARY KEY,
                requester_session_key TEXT NOT NULL,
                target_session_key TEXT NOT NULL,
                target_agent TEXT NOT NULL,
                task_text TEXT NOT NULL,
                status TEXT NOT NULL,
                result_text TEXT,
                error_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                lineage_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def save_job(self, job: CollaborationJob) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO collaboration_jobs (
                    job_id, requester_session_key, target_session_key, target_agent,
                    task_text, status, result_text, error_reason, created_at,
                    updated_at, lineage_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.requester_session_key,
                    job.target_session_key,
                    job.target_agent,
                    job.task_text,
                    job.status,
                    job.result_text,
                    job.error_reason,
                    job.created_at,
                    job.updated_at,
                    json.dumps(job.lineage),
                ),
            )

    def get_job(self, job_id: str) -> Optional[CollaborationJob]:
        row = self._conn.execute(
            "SELECT * FROM collaboration_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return CollaborationJob(
            job_id=row["job_id"],
            requester_session_key=row["requester_session_key"],
            target_session_key=row["target_session_key"],
            target_agent=row["target_agent"],
            task_text=row["task_text"],
            status=row["status"],
            result_text=row["result_text"],
            error_reason=row["error_reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            lineage=json.loads(row["lineage_json"] or "[]"),
        )

    def has_pending_for_session(self, session_key: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM collaboration_jobs
            WHERE status = 'pending'
              AND (requester_session_key = ? OR target_session_key = ?)
            LIMIT 1
            """,
            (session_key, session_key),
        ).fetchone()
        return row is not None


def create_collaboration_job(
    *,
    store: CollaborationStore,
    requester_session_key: str,
    target_session_key: str,
    target_agent: str,
    task_text: str,
    lineage: list[str] | None = None,
) -> CollaborationJob:
    job = CollaborationJob(
        job_id=f"job-{uuid.uuid4().hex[:12]}",
        requester_session_key=requester_session_key,
        target_session_key=target_session_key,
        target_agent=target_agent,
        task_text=task_text,
        lineage=list(lineage or []),
    )
    store.save_job(job)
    return job


def resolve_target_alias(config: GatewayConfig, target_agent: str, requester_session_key: str) -> Dict[str, Any]:
    collaboration_cfg = config.collaboration or {}
    targets = collaboration_cfg.get("targets", {}) if isinstance(collaboration_cfg, dict) else {}
    target = targets.get(target_agent)
    if not isinstance(target, dict):
        raise KeyError(f"Unknown collaboration target: {target_agent}")
    chat_id = str(target.get("chat_id", "")).strip()
    if not chat_id:
        raise ValueError(f"Target {target_agent} is missing chat_id")
    if requester_session_key.endswith(f"webhook:dm:{chat_id}"):
        raise ValueError("Cannot collaborate with the current session")
    return target
