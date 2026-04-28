#!/usr/bin/env python3
"""
AgentOrchestrator — guided coding process state machine for Hermes Code Mode.

Lifecycle states:
  intake           task received, not yet analyzed
  discovery        exploring codebase/context
  product_framing  clarifying scope and requirements
  architecture     technical design phase
  planning         generating step-by-step plan
  approval         waiting for human review/approval
  implementation   code changes in progress
  validation       tests/build verification
  review           final diff/code review
  ready_for_pr     ready to submit for review
  completed        task done
  cancelled        cancelled by user or system
  failed           terminal failure

Valid transitions are defined in TRANSITIONS below.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

class OrchestratorState:
    INTAKE = "intake"
    DISCOVERY = "discovery"
    PRODUCT_FRAMING = "product_framing"
    ARCHITECTURE = "architecture"
    PLANNING = "planning"
    APPROVAL = "approval"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    REVIEW = "review"
    READY_FOR_PR = "ready_for_pr"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    ALL_STATES: frozenset = frozenset({
        "intake", "discovery", "product_framing", "architecture", "planning",
        "approval", "implementation", "validation", "review", "ready_for_pr",
        "completed", "cancelled", "failed",
    })

    TERMINAL_STATES: frozenset = frozenset({"completed", "cancelled", "failed"})


# Valid transitions: {from_state: {to_state, ...}}
TRANSITIONS: Dict[str, frozenset] = {
    OrchestratorState.INTAKE: frozenset({
        OrchestratorState.DISCOVERY,
        OrchestratorState.PRODUCT_FRAMING,
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    OrchestratorState.DISCOVERY: frozenset({
        OrchestratorState.PRODUCT_FRAMING,
        OrchestratorState.ARCHITECTURE,
        OrchestratorState.PLANNING,
        OrchestratorState.APPROVAL,
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    OrchestratorState.PRODUCT_FRAMING: frozenset({
        OrchestratorState.ARCHITECTURE,
        OrchestratorState.PLANNING,
        OrchestratorState.APPROVAL,
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    OrchestratorState.ARCHITECTURE: frozenset({
        OrchestratorState.PLANNING,
        OrchestratorState.APPROVAL,
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    OrchestratorState.PLANNING: frozenset({
        OrchestratorState.APPROVAL,
        OrchestratorState.IMPLEMENTATION,
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    OrchestratorState.APPROVAL: frozenset({
        OrchestratorState.IMPLEMENTATION,
        OrchestratorState.PLANNING,    # re-plan after review
        OrchestratorState.ARCHITECTURE,
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    OrchestratorState.IMPLEMENTATION: frozenset({
        OrchestratorState.VALIDATION,
        OrchestratorState.APPROVAL,    # mid-implementation approval needed
        OrchestratorState.REVIEW,
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    OrchestratorState.VALIDATION: frozenset({
        OrchestratorState.REVIEW,
        OrchestratorState.IMPLEMENTATION,  # fix failures
        OrchestratorState.APPROVAL,
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    OrchestratorState.REVIEW: frozenset({
        OrchestratorState.READY_FOR_PR,
        OrchestratorState.IMPLEMENTATION,  # address review comments
        OrchestratorState.VALIDATION,
        OrchestratorState.COMPLETED,
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    OrchestratorState.READY_FOR_PR: frozenset({
        OrchestratorState.COMPLETED,
        OrchestratorState.IMPLEMENTATION,  # post-review changes
        OrchestratorState.CANCELLED,
        OrchestratorState.FAILED,
    }),
    # Terminal states — no outgoing transitions
    OrchestratorState.COMPLETED: frozenset(),
    OrchestratorState.CANCELLED: frozenset(),
    OrchestratorState.FAILED: frozenset(),
}


def validate_transition(from_state: str, to_state: str) -> tuple[bool, str]:
    """Return (valid, reason). reason is empty string when valid."""
    if from_state not in OrchestratorState.ALL_STATES:
        return False, f"Unknown from_state: {from_state!r}"
    if to_state not in OrchestratorState.ALL_STATES:
        return False, f"Unknown to_state: {to_state!r}"
    if from_state in OrchestratorState.TERMINAL_STATES:
        return False, f"State {from_state!r} is terminal — no transitions allowed"
    allowed = TRANSITIONS.get(from_state, frozenset())
    if to_state not in allowed:
        return False, (
            f"Invalid transition {from_state!r} → {to_state!r}. "
            f"Allowed: {sorted(allowed)}"
        )
    return True, ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DB layer
# ---------------------------------------------------------------------------

class OrchestratedRunDB:
    _WRITE_MAX_RETRIES = 5

    def __init__(self, db_path: Optional[Path] = None):
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
        for ddl in [
            """CREATE TABLE IF NOT EXISTS orchestrated_runs (
                id TEXT PRIMARY KEY,
                workspace_id TEXT,
                code_session_id TEXT,
                title TEXT,
                task_description TEXT,
                state TEXT NOT NULL DEFAULT 'intake',
                branch TEXT,
                worktree_path TEXT,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS orchestrated_run_events (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                type TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT,
                message TEXT,
                payload_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )""",
        ]:
            try:
                self._conn.execute(ddl)
            except sqlite3.OperationalError:
                pass

        for idx in [
            "CREATE INDEX IF NOT EXISTS idx_orch_runs_workspace ON orchestrated_runs(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_orch_runs_session ON orchestrated_runs(code_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_orch_runs_state ON orchestrated_runs(state)",
            "CREATE INDEX IF NOT EXISTS idx_orch_run_events_run_id ON orchestrated_run_events(run_id)",
        ]:
            try:
                self._conn.execute(idx)
            except sqlite3.OperationalError:
                pass

        self._conn.commit()

    def _execute_write(self, fn):
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

    def _row_to_dict(self, row) -> Dict[str, Any]:
        result = dict(row)
        raw = result.pop("metadata_json", "{}")
        try:
            result["metadata"] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            result["metadata"] = {}
        return result

    def create_run(
        self,
        workspace_id: Optional[str] = None,
        code_session_id: Optional[str] = None,
        title: Optional[str] = None,
        task_description: Optional[str] = None,
        branch: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        now = _utc_now()
        meta_json = json.dumps(metadata or {})

        def _do(conn):
            conn.execute(
                """INSERT INTO orchestrated_runs
                   (id, workspace_id, code_session_id, title, task_description,
                    state, branch, metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, workspace_id, code_session_id, title, task_description,
                 OrchestratorState.INTAKE, branch, meta_json, now, now),
            )

        self._execute_write(_do)
        cursor = self._conn.execute(
            "SELECT * FROM orchestrated_runs WHERE id = ?", (run_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else {}

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM orchestrated_runs WHERE id = ?", (run_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def list_runs(
        self,
        workspace_id: Optional[str] = None,
        code_session_id: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: list = []
        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if code_session_id:
            clauses.append("code_session_id = ?")
            params.append(code_session_id)
        if state:
            clauses.append("state = ?")
            params.append(state)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        cursor = self._conn.execute(
            f"SELECT * FROM orchestrated_runs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        return [self._row_to_dict(r) for r in cursor.fetchall()]

    def update_run(self, run_id: str, **updates) -> Optional[Dict[str, Any]]:
        allowed_fields = {
            "title", "task_description", "state", "branch",
            "worktree_path", "metadata_json", "completed_at",
        }
        fields = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}
        if not fields:
            return self.get_run(run_id)
        fields["updated_at"] = _utc_now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [run_id]

        def _do(conn):
            conn.execute(
                f"UPDATE orchestrated_runs SET {set_clause} WHERE id = ?", values
            )

        self._execute_write(_do)
        return self.get_run(run_id)

    def add_event(
        self,
        run_id: str,
        event_type: str,
        from_state: Optional[str] = None,
        to_state: Optional[str] = None,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event_id = str(uuid.uuid4())
        now = _utc_now()
        payload_json = json.dumps(payload or {})

        def _do(conn):
            conn.execute(
                """INSERT INTO orchestrated_run_events
                   (id, run_id, type, from_state, to_state, message, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, run_id, event_type, from_state, to_state, message, payload_json, now),
            )

        self._execute_write(_do)
        return {
            "id": event_id, "run_id": run_id, "type": event_type,
            "from_state": from_state, "to_state": to_state,
            "message": message, "payload": payload or {}, "created_at": now,
        }

    def list_events(self, run_id: str) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM orchestrated_run_events WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        )
        events = []
        for row in cursor.fetchall():
            e = dict(row)
            try:
                e["payload"] = json.loads(e.pop("payload_json", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                e["payload"] = {}
            events.append(e)
        return events


# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """Guided coding process state machine.

    Manages orchestrated runs, validates state transitions, emits events,
    and links artifacts via ArtifactLedger.
    """

    def __init__(self, db_path: Optional[Path] = None, realtime_hub=None):
        self._db_path = db_path
        self._realtime_hub = realtime_hub

    def _db(self) -> OrchestratedRunDB:
        return OrchestratedRunDB(db_path=self._db_path)

    def _ledger(self):
        from hermes_cli.code.artifact_ledger import ArtifactLedger
        return ArtifactLedger(db_path=self._db_path, realtime_hub=self._realtime_hub)

    async def _emit(self, event_type: str, payload: dict) -> None:
        if self._realtime_hub:
            try:
                await self._realtime_hub.broadcast(event_type, {"payload": payload})
            except Exception:
                pass

    def create_run(
        self,
        workspace_id: Optional[str] = None,
        code_session_id: Optional[str] = None,
        title: Optional[str] = None,
        task_description: Optional[str] = None,
        branch: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_create_intake_artifact: bool = True,
    ) -> Dict[str, Any]:
        """Create a new orchestrated run starting at 'intake' state."""
        db = self._db()
        try:
            run = db.create_run(
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                title=title,
                task_description=task_description,
                branch=branch,
                metadata=metadata,
            )
            db.add_event(
                run["id"],
                "orchestrator.run.created",
                to_state=OrchestratorState.INTAKE,
                message=f"Run created: {title or 'untitled'}",
            )
        finally:
            db.close()

        if auto_create_intake_artifact and task_description:
            try:
                ledger = self._ledger()
                ledger.create_artifact(
                    category="task_intake",
                    content=task_description,
                    title=title or "Task Intake",
                    workspace_id=workspace_id,
                    code_session_id=code_session_id,
                    orchestrated_run_id=run["id"],
                )
            except Exception as exc:
                logger.warning("Failed to create intake artifact: %s", exc)

        return run

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        try:
            return db.get_run(run_id)
        finally:
            db.close()

    def list_runs(
        self,
        workspace_id: Optional[str] = None,
        code_session_id: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        db = self._db()
        try:
            return db.list_runs(
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                state=state,
                limit=limit,
            )
        finally:
            db.close()

    def transition(
        self,
        run_id: str,
        to_state: str,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transition run to *to_state*. Raises ValueError on invalid transition."""
        db = self._db()
        try:
            run = db.get_run(run_id)
            if not run:
                raise ValueError(f"Orchestrated run not found: {run_id}")

            from_state = run["state"]
            valid, reason = validate_transition(from_state, to_state)
            if not valid:
                raise ValueError(reason)

            updates: Dict[str, Any] = {"state": to_state}
            if to_state in OrchestratorState.TERMINAL_STATES:
                updates["completed_at"] = _utc_now()

            run = db.update_run(run_id, **updates)
            db.add_event(
                run_id,
                "orchestrator.run.transitioned",
                from_state=from_state,
                to_state=to_state,
                message=message or f"{from_state} → {to_state}",
                payload=payload,
            )
        finally:
            db.close()

        return run

    def cancel_run(self, run_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        return self.transition(
            run_id,
            OrchestratorState.CANCELLED,
            message=reason or "Cancelled",
        )

    def fail_run(self, run_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        return self.transition(
            run_id,
            OrchestratorState.FAILED,
            message=reason or "Failed",
        )

    def list_events(self, run_id: str) -> List[Dict[str, Any]]:
        db = self._db()
        try:
            return db.list_events(run_id)
        finally:
            db.close()

    def attach_artifact(
        self,
        run_id: str,
        category: str,
        content: str,
        title: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a typed ledger artifact linked to this run."""
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        ledger = self._ledger()
        return ledger.create_artifact(
            category=category,
            content=content,
            title=title,
            workspace_id=run.get("workspace_id"),
            code_session_id=run.get("code_session_id"),
            orchestrated_run_id=run_id,
            **kwargs,
        )

    @staticmethod
    def valid_states() -> List[str]:
        return sorted(OrchestratorState.ALL_STATES)

    @staticmethod
    def valid_transitions(from_state: str) -> List[str]:
        return sorted(TRANSITIONS.get(from_state, frozenset()))
