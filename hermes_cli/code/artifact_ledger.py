#!/usr/bin/env python3
"""
ArtifactLedger — typed engineering artifact store for Hermes Code Mode.

Wraps the existing CodeSessionDB artifact persistence with typed categories
and a clean service API. Does not replace the low-level artifact table;
adds a separate `ledger_artifacts` table for structured engineering docs.

Artifact categories:
  task_intake           raw task description as received
  prd_lite              minimal product requirements doc
  acceptance_criteria   structured AC list
  architecture_note     design/architecture commentary
  adr                   architecture decision record
  implementation_plan   step-by-step implementation plan
  command_log           captured command output
  diff_summary          summary of a git diff
  test_report           test run results
  review_report         code review notes
  deploy_plan           deployment checklist/plan
  deploy_report         deployment outcome report
  memory_update         memory/context update fragment
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

ArtifactCategory = Literal[
    "task_intake",
    "prd_lite",
    "acceptance_criteria",
    "architecture_note",
    "adr",
    "implementation_plan",
    "command_log",
    "diff_summary",
    "test_report",
    "review_report",
    "deploy_plan",
    "deploy_report",
    "memory_update",
    "other",
]

ARTIFACT_CATEGORIES: list[str] = list(ArtifactCategory.__args__)  # type: ignore[attr-defined]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactLedgerDB:
    """Persistence layer for ledger artifacts. Operates on the shared state DB."""

    _WRITE_MAX_RETRIES = 5

    def __init__(self, db_path: Optional[Path] = None):
        import sqlite3
        import time
        import random
        from hermes_cli.config import get_hermes_home

        self._time = time
        self._random = random
        self._db_path = db_path or (get_hermes_home() / "state.db")
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        ddl = """
        CREATE TABLE IF NOT EXISTS ledger_artifacts (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL DEFAULT '',
            format TEXT NOT NULL DEFAULT 'markdown',
            workspace_id TEXT,
            code_session_id TEXT,
            flow_id TEXT,
            command_id TEXT,
            orchestrated_run_id TEXT,
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_code_session_id ON ledger_artifacts(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_workspace_id ON ledger_artifacts(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_category ON ledger_artifacts(category)",
            "CREATE INDEX IF NOT EXISTS idx_ledger_artifacts_created_at ON ledger_artifacts(created_at DESC)",
        ]
        try:
            self._conn.execute(ddl)
            for idx in indices:
                try:
                    self._conn.execute(idx)
                except Exception:
                    pass
            self._conn.commit()
        except Exception as exc:
            logger.error("ArtifactLedgerDB schema init failed: %s", exc)

    def _execute_write(self, fn):
        import sqlite3
        last_err = None
        for _ in range(self._WRITE_MAX_RETRIES):
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    result = fn(self._conn)
                    self._conn.commit()
                    return result
                except BaseException:
                    self._conn.rollback()
                    raise
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" not in str(e).lower():
                    raise
                self._time.sleep(self._random.uniform(0.05, 0.15))
        raise last_err

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def create_artifact(
        self,
        category: str,
        content: str,
        title: Optional[str] = None,
        format: str = "markdown",
        workspace_id: Optional[str] = None,
        code_session_id: Optional[str] = None,
        flow_id: Optional[str] = None,
        command_id: Optional[str] = None,
        orchestrated_run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        artifact_id = str(uuid.uuid4())
        now = _utc_now()
        meta_json = json.dumps(metadata or {})

        def _do(conn):
            conn.execute(
                """INSERT INTO ledger_artifacts
                   (id, category, title, content, format,
                    workspace_id, code_session_id, flow_id, command_id,
                    orchestrated_run_id, metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact_id, category, title, content, format,
                    workspace_id, code_session_id, flow_id, command_id,
                    orchestrated_run_id, meta_json, now, now,
                ),
            )

        self._execute_write(_do)
        cursor = self._conn.execute(
            "SELECT * FROM ledger_artifacts WHERE id = ?", (artifact_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else {}

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM ledger_artifacts WHERE id = ?", (artifact_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def list_artifacts(
        self,
        code_session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        orchestrated_run_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: list = []
        if code_session_id:
            clauses.append("code_session_id = ?")
            params.append(code_session_id)
        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if orchestrated_run_id:
            clauses.append("orchestrated_run_id = ?")
            params.append(orchestrated_run_id)
        if category:
            clauses.append("category = ?")
            params.append(category)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        cursor = self._conn.execute(
            f"SELECT * FROM ledger_artifacts {where} ORDER BY created_at ASC LIMIT ? OFFSET ?",
            params,
        )
        return [self._row_to_dict(r) for r in cursor.fetchall()]

    def update_artifact(self, artifact_id: str, content: str) -> Optional[Dict[str, Any]]:
        now = _utc_now()

        def _do(conn):
            conn.execute(
                "UPDATE ledger_artifacts SET content = ?, updated_at = ? WHERE id = ?",
                (content, now, artifact_id),
            )

        self._execute_write(_do)
        return self.get_artifact(artifact_id)

    def _row_to_dict(self, row) -> Dict[str, Any]:
        result = dict(row)
        raw_meta = result.pop("metadata_json", "{}")
        try:
            result["metadata"] = json.loads(raw_meta) if raw_meta else {}
        except (json.JSONDecodeError, TypeError):
            result["metadata"] = {}
        return result


class ArtifactLedger:
    """Service layer for the artifact ledger."""

    def __init__(self, db_path: Optional[Path] = None, realtime_hub=None):
        self._db_path = db_path
        self._realtime_hub = realtime_hub

    def _db(self) -> ArtifactLedgerDB:
        return ArtifactLedgerDB(db_path=self._db_path)

    async def _emit(self, event_type: str, payload: dict) -> None:
        if self._realtime_hub:
            try:
                await self._realtime_hub.broadcast(event_type, {"payload": payload})
            except Exception:
                pass

    def create_artifact(
        self,
        category: str,
        content: str,
        title: Optional[str] = None,
        format: str = "markdown",
        workspace_id: Optional[str] = None,
        code_session_id: Optional[str] = None,
        flow_id: Optional[str] = None,
        command_id: Optional[str] = None,
        orchestrated_run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create and persist a typed ledger artifact."""
        if category not in ARTIFACT_CATEGORIES:
            raise ValueError(f"Unknown artifact category: {category!r}. Valid: {ARTIFACT_CATEGORIES}")

        db = self._db()
        try:
            artifact = db.create_artifact(
                category=category,
                content=content,
                title=title,
                format=format,
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                flow_id=flow_id,
                command_id=command_id,
                orchestrated_run_id=orchestrated_run_id,
                metadata=metadata,
            )
        finally:
            db.close()

        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        try:
            return db.get_artifact(artifact_id)
        finally:
            db.close()

    def list_artifacts(
        self,
        code_session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        orchestrated_run_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        db = self._db()
        try:
            return db.list_artifacts(
                code_session_id=code_session_id,
                workspace_id=workspace_id,
                orchestrated_run_id=orchestrated_run_id,
                category=category,
                limit=limit,
                offset=offset,
            )
        finally:
            db.close()

    def update_artifact(self, artifact_id: str, content: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        try:
            return db.update_artifact(artifact_id, content)
        finally:
            db.close()

    def categories(self) -> List[str]:
        return list(ARTIFACT_CATEGORIES)
