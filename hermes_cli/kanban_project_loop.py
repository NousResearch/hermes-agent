"""Bounded, idempotent project-goal continuation for Kanban boards."""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import time
from typing import Iterable, Optional

from hermes_cli import kanban_db as kb


def _from_row(row: sqlite3.Row) -> kb.ProjectLoop:
    try:
        criteria = json.loads(row["acceptance_criteria"])
    except (TypeError, json.JSONDecodeError):
        criteria = []
    return kb.ProjectLoop(
        project_key=row["project_key"],
        goal=row["goal"],
        acceptance_criteria=[str(item) for item in criteria if str(item).strip()],
        status=row["status"],
        max_rounds=int(row["max_rounds"]),
        max_tasks=int(row["max_tasks"]),
        rounds_used=int(row["rounds_used"]),
        tasks_created=int(row["tasks_created"]),
        current_verify_task_id=row["current_verify_task_id"],
        current_owner_gate_task_id=row["current_owner_gate_task_id"],
        last_decision=row["last_decision"],
        stop_reason=row["stop_reason"],
    )


def configure_project_loop(
    conn: sqlite3.Connection,
    *,
    project_key: str,
    goal: str,
    acceptance_criteria: Iterable[str],
    verify_task_id: str,
    max_rounds: int,
    max_tasks: int,
) -> kb.ProjectLoop:
    """Persist an explicit project goal loop and bind its current Verify card."""
    key = str(project_key or "").strip()
    clean_goal = str(goal or "").strip()
    criteria = [str(item).strip() for item in acceptance_criteria if str(item).strip()]
    if not key or not clean_goal or not criteria:
        raise ValueError("project_key, goal, and acceptance_criteria are required")
    if int(max_rounds) < 1 or int(max_tasks) < 1:
        raise ValueError("max_rounds and max_tasks must be positive")
    if kb.get_task(conn, verify_task_id) is None:
        raise ValueError(f"unknown verify task: {verify_task_id}")
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO project_loops "
            "(project_key, goal, acceptance_criteria, status, max_rounds, max_tasks, "
            "current_verify_task_id, created_at, updated_at) "
            "VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_key) DO UPDATE SET goal=excluded.goal, "
            "acceptance_criteria=excluded.acceptance_criteria, "
            "max_rounds=excluded.max_rounds, max_tasks=excluded.max_tasks, "
            "current_verify_task_id=excluded.current_verify_task_id, "
            "current_owner_gate_task_id=NULL, "
            "updated_at=excluded.updated_at",
            (
                key,
                clean_goal,
                json.dumps(criteria, ensure_ascii=False),
                int(max_rounds),
                int(max_tasks),
                verify_task_id,
                now,
                now,
            ),
        )
    state = get_project_loop(conn, key)
    if state is None:  # pragma: no cover
        raise RuntimeError("project loop was not persisted")
    return state


def get_project_loop(
    conn: sqlite3.Connection, project_key: str
) -> Optional[kb.ProjectLoop]:
    row = conn.execute(
        "SELECT * FROM project_loops WHERE project_key = ?", (project_key,)
    ).fetchone()
    return _from_row(row) if row else None


def list_project_loop_round_tasks(
    conn: sqlite3.Connection,
    project_key: str,
    *,
    round_no: Optional[int] = None,
) -> list[kb.ProjectLoopTask]:
    query = "SELECT * FROM project_loop_tasks WHERE project_key = ?"
    params: list[object] = [project_key]
    if round_no is not None:
        query += " AND round_no = ?"
        params.append(int(round_no))
    query += " ORDER BY round_no, role, step_key"
    return [kb.ProjectLoopTask(**dict(row)) for row in conn.execute(query, params)]


def _normalize_steps(next_steps: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    seen: set[str] = set()
    for index, raw in enumerate(next_steps):
        if not isinstance(raw, dict):
            raise ValueError("each project-loop next step must be an object")
        title = str(raw.get("title") or "").strip()
        key = str(raw.get("key") or f"step-{index + 1}").strip().lower()
        key = re.sub(r"[^a-z0-9_-]+", "-", key).strip("-_")
        if not title or not key:
            raise ValueError("each project-loop next step needs a title and stable key")
        if key in seen:
            raise ValueError(f"duplicate project-loop step key: {key}")
        seen.add(key)
        normalized.append(
            {
                "key": key,
                "title": title,
                "body": str(raw.get("body") or "").strip() or None,
                "assignee": kb._canonical_assignee(raw.get("assignee")),
            }
        )
    return normalized


def _cached_result(row: sqlite3.Row) -> kb.ProjectLoopReconcileResult:
    payload = json.loads(row["result"])
    return kb.ProjectLoopReconcileResult(
        decision=payload["decision"],
        created_task_ids=list(payload.get("created_task_ids") or []),
        owner_gate_task_id=payload.get("owner_gate_task_id"),
        reason=payload.get("reason"),
    )


def _store_receipt(
    conn: sqlite3.Connection,
    *,
    project_key: str,
    verify_task_id: str,
    result: kb.ProjectLoopReconcileResult,
) -> None:
    conn.execute(
        "INSERT INTO project_loop_reconciliations "
        "(project_key, verify_task_id, decision, result, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            project_key,
            verify_task_id,
            result.decision,
            json.dumps(result.__dict__, ensure_ascii=False),
            int(time.time()),
        ),
    )


def _stop(
    conn: sqlite3.Connection,
    project_key: str,
    reason: str,
    now: int,
) -> kb.ProjectLoopReconcileResult:
    conn.execute(
        "UPDATE project_loops SET status='stopped', last_decision='stop', "
        "current_owner_gate_task_id=NULL, stop_reason=?, updated_at=? "
        "WHERE project_key=?",
        (reason, now, project_key),
    )
    return kb.ProjectLoopReconcileResult(decision="stop", reason=reason)


def reconcile_project_loop(
    conn: sqlite3.Connection,
    *,
    project_key: str,
    verify_task_id: str,
    decision: str,
    next_steps: Iterable[dict] = (),
    owner_question: Optional[str] = None,
    _receipt_task_id: Optional[str] = None,
    _owner_gate_task_id: Optional[str] = None,
    _in_transaction: bool = False,
) -> kb.ProjectLoopReconcileResult:
    """Apply one Verify decision and atomically materialize a bounded next round."""
    decision = str(decision or "").strip()
    if decision not in kb.VALID_PROJECT_LOOP_DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(kb.VALID_PROJECT_LOOP_DECISIONS)}"
        )
    receipt_task_id = _receipt_task_id or verify_task_id
    txn = contextlib.nullcontext(conn) if _in_transaction else kb.write_txn(conn)
    with txn:
        cached = conn.execute(
            "SELECT result FROM project_loop_reconciliations "
            "WHERE project_key = ? AND verify_task_id = ?",
            (project_key, receipt_task_id),
        ).fetchone()
        if cached:
            return _cached_result(cached)
        row = conn.execute(
            "SELECT * FROM project_loops WHERE project_key = ?", (project_key,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown project loop: {project_key}")
        state = _from_row(row)
        if state.current_verify_task_id != verify_task_id:
            raise ValueError(
                f"{verify_task_id} is not the current Verify card for {project_key}"
            )
        if _owner_gate_task_id is not None and (
            state.status != "owner_gate"
            or state.current_owner_gate_task_id != _owner_gate_task_id
        ):
            raise ValueError(
                f"{_owner_gate_task_id} is not the open owner gate for {project_key}"
            )
        now = int(time.time())

        if decision == "goal_complete":
            result = kb.ProjectLoopReconcileResult(decision=decision)
            conn.execute(
                "UPDATE project_loops SET status='complete', last_decision=?, "
                "current_owner_gate_task_id=NULL, stop_reason=NULL, updated_at=? "
                "WHERE project_key=?",
                (decision, now, project_key),
            )
        elif decision == "stop":
            reason = (
                "Owner requested stop"
                if _owner_gate_task_id is not None
                else "Verify requested stop"
            )
            result = _stop(conn, project_key, reason, now)
        elif decision == "owner_judgment_required":
            question = str(owner_question or "Owner judgment is required.").strip()
            gate_body = (
                f"{question}\n\n"
                "Complete this gate with outcome goal_complete, continue_bounded, "
                "or stop. continue_bounded must provide "
                "metadata.project_loop.next_steps."
            )
            gate_id = kb.create_task(
                conn,
                title=f"Owner decision: {project_key}",
                body=gate_body,
                initial_status="blocked",
                created_by=verify_task_id,
                idempotency_key=(
                    f"project-loop:{project_key}:owner-gate:{state.rounds_used}"
                ),
                _in_transaction=True,
            )
            conn.execute(
                "UPDATE tasks SET block_kind='needs_input', result=? WHERE id=?",
                (question, gate_id),
            )
            result = kb.ProjectLoopReconcileResult(
                decision=decision,
                created_task_ids=[gate_id],
                owner_gate_task_id=gate_id,
            )
            conn.execute(
                "UPDATE project_loops SET status='owner_gate', last_decision=?, "
                "current_owner_gate_task_id=?, stop_reason=NULL, updated_at=? "
                "WHERE project_key=?",
                (decision, gate_id, now, project_key),
            )
        else:
            invalid_steps_reason: Optional[str] = None
            try:
                steps = _normalize_steps(next_steps)
            except ValueError as exc:
                invalid_steps_reason = f"invalid continue_bounded next steps: {exc}"
                steps = []
            if not steps:
                result = _stop(
                    conn,
                    project_key,
                    invalid_steps_reason
                    or "continue_bounded requires at least one next step",
                    now,
                )
            else:
                round_no = state.rounds_used + 1
                needed_tasks = len(steps) + 1
                if round_no > state.max_rounds:
                    result = _stop(
                        conn,
                        project_key,
                        f"round budget exhausted ({state.max_rounds})",
                        now,
                    )
                elif state.tasks_created + needed_tasks > state.max_tasks:
                    result = _stop(
                        conn,
                        project_key,
                        f"task budget exhausted ({state.max_tasks})",
                        now,
                    )
                else:
                    verify_task = kb.get_task(conn, verify_task_id)
                    if verify_task is None:  # pragma: no cover
                        raise ValueError(f"unknown verify task: {verify_task_id}")
                    work_ids: list[str] = []
                    for step in steps:
                        task_id = kb.create_task(
                            conn,
                            title=step["title"],
                            body=step["body"],
                            assignee=step["assignee"] or verify_task.assignee,
                            created_by=verify_task_id,
                            workspace_kind=verify_task.workspace_kind,
                            workspace_path=verify_task.workspace_path,
                            tenant=verify_task.tenant,
                            priority=verify_task.priority,
                            idempotency_key=(
                                f"project-loop:{project_key}:{round_no}:{step['key']}"
                            ),
                            _in_transaction=True,
                        )
                        conn.execute(
                            "INSERT INTO project_loop_tasks "
                            "(project_key, round_no, step_key, role, task_id) "
                            "VALUES (?, ?, ?, 'work', ?)",
                            (project_key, round_no, step["key"], task_id),
                        )
                        work_ids.append(task_id)
                    criteria = "\n".join(
                        f"- {item}" for item in state.acceptance_criteria
                    )
                    next_verify = kb.create_task(
                        conn,
                        title=f"Verify project goal: {project_key} (round {round_no})",
                        body=(
                            f"Goal: {state.goal}\n\nAcceptance criteria:\n{criteria}\n\n"
                            "Complete with outcome goal_complete, continue_bounded, "
                            "owner_judgment_required, or stop. continue_bounded must "
                            "provide metadata.project_loop.next_steps."
                        ),
                        assignee=verify_task.assignee,
                        created_by=verify_task_id,
                        workspace_kind=verify_task.workspace_kind,
                        workspace_path=verify_task.workspace_path,
                        tenant=verify_task.tenant,
                        priority=verify_task.priority,
                        parents=work_ids,
                        idempotency_key=(
                            f"project-loop:{project_key}:{round_no}:verify"
                        ),
                        _in_transaction=True,
                    )
                    conn.execute(
                        "INSERT INTO project_loop_tasks "
                        "(project_key, round_no, step_key, role, task_id) "
                        "VALUES (?, ?, 'verify', 'verify', ?)",
                        (project_key, round_no, next_verify),
                    )
                    created = [*work_ids, next_verify]
                    result = kb.ProjectLoopReconcileResult(
                        decision=decision, created_task_ids=created
                    )
                    conn.execute(
                        "UPDATE project_loops SET status='awaiting_launch', "
                        "rounds_used=?, tasks_created=tasks_created+?, "
                        "current_verify_task_id=?, current_owner_gate_task_id=NULL, "
                        "last_decision=?, stop_reason=NULL, "
                        "updated_at=? WHERE project_key=?",
                        (
                            round_no,
                            needed_tasks,
                            next_verify,
                            decision,
                            now,
                            project_key,
                        ),
                    )
        _store_receipt(
            conn,
            project_key=project_key,
            verify_task_id=receipt_task_id,
            result=result,
        )
        return result


def reconcile_owner_gate(
    conn: sqlite3.Connection,
    *,
    gate_task_id: str,
    decision: str,
    next_steps: Iterable[dict] = (),
    _in_transaction: bool = False,
) -> kb.ProjectLoopReconcileResult:
    """Consume an owner gate exactly once through the bounded loop controls."""
    decision = str(decision or "").strip()
    if decision not in {"goal_complete", "continue_bounded", "stop"}:
        raise ValueError(
            "owner gate decision must be goal_complete, continue_bounded, or stop"
        )
    cached = conn.execute(
        "SELECT result FROM project_loop_reconciliations "
        "WHERE verify_task_id = ?",
        (gate_task_id,),
    ).fetchone()
    if cached:
        return _cached_result(cached)
    row = conn.execute(
        "SELECT project_key, current_verify_task_id FROM project_loops "
        "WHERE current_owner_gate_task_id = ? AND status = 'owner_gate'",
        (gate_task_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"{gate_task_id} is not an open project-loop owner gate")
    return reconcile_project_loop(
        conn,
        project_key=row["project_key"],
        verify_task_id=row["current_verify_task_id"],
        decision=decision,
        next_steps=next_steps,
        _receipt_task_id=gate_task_id,
        _owner_gate_task_id=gate_task_id,
        _in_transaction=_in_transaction,
    )


def reconcile_from_completion(
    conn: sqlite3.Connection,
    task_id: str,
    outcome: Optional[str],
    metadata: Optional[dict],
    *,
    _in_transaction: bool = False,
) -> None:
    """Trigger only for the current Verify card of an opted-in project loop."""
    gate = conn.execute(
        "SELECT 1 FROM project_loops "
        "WHERE current_owner_gate_task_id = ? AND status = 'owner_gate'",
        (task_id,),
    ).fetchone()
    if gate is not None:
        payload = metadata.get("project_loop") if isinstance(metadata, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        reconcile_owner_gate(
            conn,
            gate_task_id=task_id,
            decision=outcome or "",
            next_steps=payload.get("next_steps") or (),
            _in_transaction=_in_transaction,
        )
        return
    if outcome not in kb.VALID_PROJECT_LOOP_DECISIONS:
        return
    row = conn.execute(
        "SELECT project_key FROM project_loops WHERE current_verify_task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return
    payload = metadata.get("project_loop") if isinstance(metadata, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    reconcile_project_loop(
        conn,
        project_key=row["project_key"],
        verify_task_id=task_id,
        decision=outcome,
        next_steps=payload.get("next_steps") or (),
        owner_question=payload.get("owner_question"),
        _in_transaction=_in_transaction,
    )


def mark_launched_rounds(conn: sqlite3.Connection) -> int:
    """Mark awaiting rounds active only after dispatcher spawn succeeds."""
    rows = conn.execute(
        "SELECT project_key, rounds_used FROM project_loops "
        "WHERE status = 'awaiting_launch'"
    ).fetchall()
    activated = 0
    for row in rows:
        observed = conn.execute(
            "SELECT 1 FROM project_loop_tasks plt "
            "JOIN task_runs r ON r.task_id = plt.task_id "
            "JOIN tasks t ON t.id = plt.task_id AND t.current_run_id = r.id "
            "WHERE plt.project_key = ? AND plt.round_no = ? AND plt.role = 'work' "
            "AND t.status = 'running' "
            "AND r.status = 'running' AND r.ended_at IS NULL "
            "AND r.outcome IS NULL AND r.worker_pid IS NOT NULL "
            "LIMIT 1",
            (row["project_key"], row["rounds_used"]),
        ).fetchone()
        if observed:
            with kb.write_txn(conn):
                updated = conn.execute(
                    "UPDATE project_loops SET status='active', updated_at=? "
                    "WHERE project_key=? AND status='awaiting_launch'",
                    (int(time.time()), row["project_key"]),
                )
                activated += updated.rowcount
    return activated
