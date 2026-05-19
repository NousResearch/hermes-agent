"""
RunStore — CRUD for workflow_runs, node_runs, workflow_events tables.

All methods synchronous. Caller owns transaction commit where noted.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ms_to_dt(ms: Optional[int]) -> Optional[str]:
    """Convert epoch-ms to ISO-8601 string for JSON serialisation."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _row_to_run(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["started_at"] = _ms_to_dt(d.get("started_at"))
    d["completed_at"] = _ms_to_dt(d.get("completed_at"))
    d["last_heartbeat"] = _ms_to_dt(d.get("last_heartbeat"))
    if d.get("metadata"):
        try:
            d["metadata"] = json.loads(d["metadata"])
        except Exception:
            d["metadata"] = {}
    return d


def _row_to_node_run(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["started_at"] = _ms_to_dt(d.get("started_at"))
    d["completed_at"] = _ms_to_dt(d.get("completed_at"))
    for col in ("depends_on", "skills", "allowed_tools", "denied_tools", "artifact_refs"):
        if d.get(col):
            try:
                d[col] = json.loads(d[col])
            except Exception:
                d[col] = None
    if d.get("metadata"):
        try:
            d["metadata"] = json.loads(d["metadata"])
        except Exception:
            d["metadata"] = {}
    return d


def _row_to_event(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    if d.get("data"):
        try:
            d["data"] = json.loads(d["data"])
        except Exception:
            d["data"] = {}
    d["created_at"] = _ms_to_dt(d.get("created_at"))
    return d


class RunStore:
    """CRUD for runs, node_runs and events."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------ #
    # Workflow Runs                                                        #
    # ------------------------------------------------------------------ #

    def create_workflow_run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        working_path: str,
        user_message: str,
        trigger: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        now = _now_ms()
        self._conn.execute(
            """
            INSERT INTO workflow_runs
              (id, workflow_id, conversation_id, working_path, user_message,
               status, current_phase, metadata, started_at, last_heartbeat)
            VALUES (?, ?, ?, ?, ?, 'pending', 'plan', ?, ?, ?)
            """,
            (
                run_id,
                workflow_id,
                conversation_id,
                working_path,
                user_message,
                json.dumps({"trigger": trigger} if trigger else {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        return self.get_workflow_run(run_id)  # type: ignore[return-value]

    def get_workflow_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM workflow_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _row_to_run(row) if row else None

    def list_workflow_runs(
        self,
        *,
        workflow_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if workflow_id:
            clauses.append("workflow_id = ?")
            params.append(workflow_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM workflow_runs {where} ORDER BY started_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_run(r) for r in rows]

    def update_workflow_run(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        cols: List[str] = []
        vals: List[Any] = []
        now = _now_ms()
        cols.append("last_heartbeat = ?")
        vals.append(now)
        if status is not None:
            cols.append("status = ?")
            vals.append(status)
            if status in ("completed", "failed", "cancelled"):
                cols.append("completed_at = ?")
                vals.append(now)
        if error is not None:
            cols.append("error = ?")
            vals.append(error)
        if metadata is not None:
            cols.append("metadata = ?")
            vals.append(json.dumps(metadata))
        vals.append(run_id)
        self._conn.execute(
            f"UPDATE workflow_runs SET {', '.join(cols)} WHERE id = ?",
            vals,
        )
        self._conn.commit()

    def mark_crashed_runs(self) -> int:
        """Mark any pending/running runs as crashed (resume policy: no auto-resume)."""
        now = _now_ms()
        result = self._conn.execute(
            """
            UPDATE workflow_runs
               SET status = 'failed', error = 'crashed: plugin restarted', completed_at = ?
             WHERE status IN ('pending', 'running')
            """,
            (now,),
        )
        self._conn.commit()
        return result.rowcount

    def cancel_workflow_run(self, run_id: str) -> None:
        now = _now_ms()
        self._conn.execute(
            """
            UPDATE workflow_runs
               SET status = 'cancelled', completed_at = ?
             WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
            """,
            (now, run_id),
        )
        # Cancel any running/pending node_runs too
        self._conn.execute(
            """
            UPDATE node_runs
               SET status = 'cancelled', completed_at = ?
             WHERE workflow_run_id = ? AND status NOT IN ('completed', 'failed', 'cancelled', 'skipped')
            """,
            (now, run_id),
        )
        self._conn.commit()

    def pause_workflow_run(self, run_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        now = _now_ms()
        self._conn.execute(
            "UPDATE workflow_runs SET status = 'paused', last_heartbeat = ?, metadata = COALESCE(?, metadata) WHERE id = ?",
            (now, json.dumps(metadata) if metadata else None, run_id),
        )
        self._conn.commit()

    def resume_workflow_run(self, run_id: str) -> None:
        now = _now_ms()
        self._conn.execute(
            "UPDATE workflow_runs SET status = 'running', last_heartbeat = ? WHERE id = ? AND status = 'paused'",
            (now, run_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Node Runs                                                           #
    # ------------------------------------------------------------------ #

    def create_node_run(
        self,
        *,
        workflow_run_id: str,
        dag_node_id: str,
        node_type: str,
        node_run_id: Optional[str] = None,
        agent_profile_hint: Optional[str] = None,
        skills: Optional[List[str]] = None,
        model_hint: Optional[str] = None,
        parent_subgraph_node_run_id: Optional[str] = None,
        loop_iteration: Optional[int] = None,
        approval_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nr_id = node_run_id or str(uuid.uuid4())
        now = _now_ms()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO node_runs
              (id, workflow_run_id, dag_node_id, node_type, status,
               agent_profile_hint, skills, model_hint,
               parent_subgraph_node_run_id, loop_iteration,
               approval_message, metadata, started_at)
            VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nr_id,
                workflow_run_id,
                dag_node_id,
                node_type,
                agent_profile_hint,
                json.dumps(skills) if skills else None,
                model_hint,
                parent_subgraph_node_run_id,
                loop_iteration,
                approval_message,
                json.dumps(metadata) if metadata else None,
                now,
            ),
        )
        self._conn.commit()
        return self.get_node_run(nr_id)  # type: ignore[return-value]

    def get_node_run(self, node_run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM node_runs WHERE id = ?", (node_run_id,)
        ).fetchone()
        return _row_to_node_run(row) if row else None

    def find_node_run(
        self,
        workflow_run_id: str,
        dag_node_id: str,
        loop_iteration: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if loop_iteration is not None:
            row = self._conn.execute(
                "SELECT * FROM node_runs WHERE workflow_run_id = ? AND dag_node_id = ? AND loop_iteration = ?",
                (workflow_run_id, dag_node_id, loop_iteration),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM node_runs WHERE workflow_run_id = ? AND dag_node_id = ? AND loop_iteration IS NULL",
                (workflow_run_id, dag_node_id),
            ).fetchone()
        return _row_to_node_run(row) if row else None

    def list_node_runs(self, workflow_run_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM node_runs WHERE workflow_run_id = ? ORDER BY started_at",
            (workflow_run_id,),
        ).fetchall()
        return [_row_to_node_run(r) for r in rows]

    def update_node_run(self, node_run_id: str, patch: Dict[str, Any]) -> None:
        _ALLOWED = {
            "status", "error", "summary", "completed_at", "started_at",
            "kanban_task_id", "assigned_agent", "approval_response",
            "artifact_refs", "metadata", "skip_reason", "retries",
        }
        cols: List[str] = []
        vals: List[Any] = []
        for k, v in patch.items():
            if k not in _ALLOWED:
                raise ValueError(f"update_node_run: unknown column '{k}'")
            if k in ("artifact_refs", "metadata") and v is not None:
                v = json.dumps(v)
            cols.append(f"{k} = ?")
            vals.append(v)
        if not cols:
            return
        vals.append(node_run_id)
        self._conn.execute(
            f"UPDATE node_runs SET {', '.join(cols)} WHERE id = ?",
            vals,
        )
        self._conn.commit()

    def try_claim_approval(
        self,
        node_run_id: str,
        decision: Literal["approve", "reject"],
        comment: Optional[str],
    ) -> bool:
        """Atomic CAS: update node_run status from paused → completed/failed. Returns True if claimed."""
        terminal = "completed" if decision == "approve" else "failed"
        now = _now_ms()
        result = self._conn.execute(
            """
            UPDATE node_runs
               SET status = ?, approval_response = ?, completed_at = ?
             WHERE id = ? AND status = 'paused'
            """,
            (terminal, comment or decision, now, node_run_id),
        )
        self._conn.commit()
        return result.rowcount > 0

    # ------------------------------------------------------------------ #
    # Workflow Events                                                      #
    # ------------------------------------------------------------------ #

    def insert_event(
        self,
        *,
        workflow_run_id: str,
        event_type: str,
        node_run_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> str:
        eid = event_id or str(uuid.uuid4())
        now = _now_ms()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO workflow_events
              (id, workflow_run_id, node_run_id, event_type, data, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                eid,
                workflow_run_id,
                node_run_id,
                event_type,
                json.dumps(data) if data else None,
                now,
            ),
        )
        self._conn.commit()
        return eid

    def list_events(
        self,
        workflow_run_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM workflow_events
             WHERE workflow_run_id = ?
             ORDER BY created_at ASC
             LIMIT ? OFFSET ?
            """,
            (workflow_run_id, limit, offset),
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def list_recent_events(
        self,
        run_id: Optional[str],
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return last N events for a run (or all runs if run_id is None)."""
        if run_id:
            rows = self._conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM workflow_events WHERE workflow_run_id = ?
                    ORDER BY created_at DESC LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (run_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM workflow_events ORDER BY created_at DESC LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (limit,),
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Phase Transitions                                                    #
    # ------------------------------------------------------------------ #

    def record_phase_transition(
        self,
        *,
        run_id: str,
        to_phase: str,
        decided_by: str,
        decision_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a phase transition for a run. Returns {from, to}."""
        row = self._conn.execute(
            "SELECT current_phase FROM workflow_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"WorkflowRun not found: {run_id}")
        from_phase = row["current_phase"]
        if from_phase == to_phase:
            return {"from": from_phase, "to": to_phase}
        tid = str(uuid.uuid4())
        now = _now_ms()
        self._conn.execute(
            "UPDATE workflow_runs SET current_phase = ? WHERE id = ?",
            (to_phase, run_id),
        )
        self._conn.execute(
            """
            INSERT INTO phase_transitions
              (id, workflow_run_id, from_phase, to_phase, decided_by, decision_data, at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tid,
                run_id,
                from_phase,
                to_phase,
                decided_by,
                json.dumps(decision_data) if decision_data else None,
                now,
            ),
        )
        self._conn.commit()
        return {"from": from_phase, "to": to_phase}

    def list_phase_transitions(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, from_phase, to_phase, decided_by, decision_data, at
              FROM phase_transitions
             WHERE workflow_run_id = ?
             ORDER BY at ASC
            """,
            (run_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("decision_data"):
                try:
                    d["decision_data"] = json.loads(d["decision_data"])
                except Exception:
                    d["decision_data"] = None
            result.append(d)
        return result

    # ------------------------------------------------------------------ #
    # Extended lookups                                                     #
    # ------------------------------------------------------------------ #

    def find_run_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM workflow_runs WHERE conversation_id = ? LIMIT 1",
            (conversation_id,),
        ).fetchone()
        return _row_to_run(row) if row else None

    def get_active_run_by_path(self, scope_path: str) -> Optional[Dict[str, Any]]:
        """Return the most recent active run for scope_path (pending/running/paused)."""
        STALE_MS = 5 * 60 * 1000
        stale_threshold = _now_ms() - STALE_MS
        row = self._conn.execute(
            """
            SELECT * FROM workflow_runs
             WHERE working_path = ?
               AND status IN ('pending', 'running', 'paused')
               AND (status != 'pending' OR last_heartbeat >= ?)
             ORDER BY started_at ASC, id ASC
             LIMIT 1
            """,
            (scope_path, stale_threshold),
        ).fetchone()
        return _row_to_run(row) if row else None

    def find_node_run_by_id(self, node_run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM node_runs WHERE id = ? LIMIT 1", (node_run_id,)
        ).fetchone()
        return _row_to_node_run(row) if row else None

    def append_workflow_event(
        self,
        *,
        workflow_run_id: str,
        event_type: str,
        node_run_id: Optional[str] = None,
        step_index: Optional[int] = None,
        step_name: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        created_at: Optional[int] = None,
    ) -> None:
        """Alias for insert_event with extended fields matching TS appendWorkflowEvent."""
        eid = event_id or str(uuid.uuid4())
        now = created_at or _now_ms()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO workflow_events
              (id, workflow_run_id, node_run_id, event_type, step_index, step_name, data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eid,
                workflow_run_id,
                node_run_id,
                event_type,
                step_index,
                step_name,
                json.dumps(data) if data else None,
                now,
            ),
        )
        self._conn.commit()

    def try_claim_approval_for_resume(
        self,
        node_run_id: str,
        decision: Literal["approved", "rejected"],
        approval_response: str,
    ) -> Dict[str, Any]:
        """Atomic CAS matching TS tryClaimApprovalForResume. Returns {claimed, terminalStatus}."""
        terminal_status = "completed" if decision == "approved" else "failed"
        now = _now_ms()
        result = self._conn.execute(
            """
            UPDATE node_runs
               SET status = ?, approval_response = ?, completed_at = ?
             WHERE id = ? AND status = 'paused'
            """,
            (terminal_status, approval_response, now, node_run_id),
        )
        self._conn.commit()
        return {"claimed": result.rowcount > 0, "terminalStatus": terminal_status}
