"""SQLite dispatcher for cheap local workflow nodes."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from hermes_cli import kanban_db as kb
from hermes_cli import workflows_db as wfdb
from hermes_cli.workflows_engine import EngineResult, run_in_memory_until_waiting
from hermes_cli.workflows_prompts import render_agent_prompt
from hermes_cli.workflows_spec import WorkflowSpec


class _AgentTaskMaterializationError(RuntimeError):
    def __init__(self, node_id: str, message: str):
        super().__init__(message)
        self.node_id = node_id


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _claim_next(
    conn: sqlite3.Connection,
    *,
    now: int,
    lease_seconds: int,
) -> tuple[str, str] | None:
    row = conn.execute(
        """
        SELECT execution_id
          FROM workflow_executions
         WHERE status = 'queued'
           AND (claim_lock IS NULL OR claim_expires <= ?)
         ORDER BY created_at, execution_id
         LIMIT 1
        """,
        (now,),
    ).fetchone()
    if row is None:
        return None

    token = secrets.token_hex(16)
    with wfdb.write_txn(conn):
        updated = conn.execute(
            """
            UPDATE workflow_executions
               SET claim_lock = ?, claim_expires = ?, updated_at = ?
             WHERE execution_id = ?
               AND status = 'queued'
               AND (claim_lock IS NULL OR claim_expires <= ?)
            """,
            (token, now + lease_seconds, now, row["execution_id"], now),
        ).rowcount
    if updated != 1:
        return None
    return row["execution_id"], token


def _append_event(
    conn: sqlite3.Connection,
    execution_id: str,
    kind: str,
    payload: dict[str, Any] | None,
    now: int,
) -> None:
    conn.execute(
        """
        INSERT INTO workflow_events (
            execution_id, node_run_id, kind, payload_json, created_at
        ) VALUES (?, NULL, ?, ?, ?)
        """,
        (execution_id, kind, _json_dumps(payload or {}), now),
    )


def _schedule_input(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    spec = wfdb.get_definition(conn, row["workflow_id"], row["version"])
    for trigger in spec.triggers:
        if trigger.type != "schedule":
            continue
        expr = trigger.cron or trigger.schedule or getattr(trigger, "expr", None)
        if row["trigger_id"] is not None:
            if trigger.id == row["trigger_id"]:
                return dict(trigger.input)
        elif expr == row["schedule"]:
            return dict(trigger.input)
    return {}


def _fire_due_schedules(conn: sqlite3.Connection, *, now: int) -> None:
    with wfdb.write_txn(conn):
        rows = conn.execute(
            """
            SELECT * FROM workflow_schedules
             WHERE enabled = 1
               AND next_run_at IS NOT NULL
               AND next_run_at <= ?
             ORDER BY next_run_at, id
            """,
            (now,),
        ).fetchall()
        for row in rows:
            wfdb.start_execution(
                conn,
                row["workflow_id"],
                input_data=_schedule_input(conn, row),
                trigger_type="schedule",
                trigger_id=row["trigger_id"],
                version=row["version"],
                now=now,
            )
            conn.execute(
                """
                UPDATE workflow_schedules
                   SET next_run_at = ?, updated_at = ?
                 WHERE id = ?
                """,
                (wfdb._next_cron_run(row["schedule"], now), now, row["id"]),
            )


def _resume_due_waits(conn: sqlite3.Connection, *, now: int) -> None:
    with wfdb.write_txn(conn):
        rows = conn.execute(
            """
            SELECT nr.id, nr.execution_id
              FROM workflow_node_runs nr
              JOIN workflow_executions ex ON ex.execution_id = nr.execution_id
             WHERE nr.status = 'waiting'
               AND nr.wait_until IS NOT NULL
               AND nr.wait_until <= ?
               AND ex.status = 'waiting'
             ORDER BY nr.wait_until, nr.id
            """,
            (now,),
        ).fetchall()
        for row in rows:
            updated = conn.execute(
                """
                UPDATE workflow_node_runs
                   SET status = 'succeeded', completed_at = ?
                 WHERE id = ? AND status = 'waiting'
                """,
                (now, row["id"]),
            ).rowcount
            if updated:
                conn.execute(
                    """
                    UPDATE workflow_executions
                       SET status = 'queued', claim_lock = NULL,
                           claim_expires = NULL, updated_at = ?
                     WHERE execution_id = ? AND status = 'waiting'
                    """,
                    (now, row["execution_id"]),
                )


def _resume_due_retries(conn: sqlite3.Connection, *, now: int) -> None:
    with wfdb.write_txn(conn):
        rows = conn.execute(
            """
            SELECT DISTINCT nr.execution_id
              FROM workflow_node_runs nr
              JOIN workflow_executions ex ON ex.execution_id = nr.execution_id
             WHERE nr.status = 'queued'
               AND (nr.wait_until IS NULL OR nr.wait_until <= ?)
               AND ex.status = 'waiting'
             ORDER BY nr.wait_until, nr.id
            """,
            (now,),
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                UPDATE workflow_executions
                   SET status = 'queued', claim_lock = NULL,
                       claim_expires = NULL, updated_at = ?
                 WHERE execution_id = ? AND status = 'waiting'
                """,
                (now, row["execution_id"]),
            )


def _render_agent_prompt(node: Any, context: dict[str, Any]) -> str:
    return render_agent_prompt(node.prompt, context)


def _create_or_get_agent_task(
    *,
    execution_id: str,
    spec: WorkflowSpec,
    node_id: str,
    node: Any,
    context: dict[str, Any],
) -> tuple[str, str]:
    board = kb.get_current_board()
    with kb.connect_closing(board=board) as kconn:
        task_id = kb.create_task(
            kconn,
            title=(node.title or f"{spec.name}: {node_id}").strip(),
            body=_render_agent_prompt(node, context),
            assignee=node.profile,
            created_by=f"workflow:{execution_id}",
            workspace_kind=node.workspace_kind or "scratch",
            workspace_path=node.workspace_path,
            skills=node.skills or None,
            max_retries=node.max_retries,
            goal_mode=bool(node.goal_mode),
            goal_max_turns=node.goal_max_turns,
            workflow_template_id=spec.id,
            current_step_key=node_id,
            model_override=node.model_override,
            idempotency_key=f"workflow:{execution_id}:{node_id}",
        )
        return task_id, board


def _parse_agent_result(raw: str | None) -> Any:
    if raw is None or raw == "":
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {"result": raw}


def _validate_result_contract(output: Any, contract: dict[str, Any]) -> list[str]:
    errors = []
    if not contract:
        return errors
    if not isinstance(output, dict):
        return ["agent result must be a JSON object to satisfy result_contract"]
    for key, expected in contract.items():
        if key not in output:
            errors.append(f"missing required result key: {key}")
            continue
        value = output[key]
        if expected == "string" and not isinstance(value, str):
            errors.append(f"result key {key} must be string")
        elif expected == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            errors.append(f"result key {key} must be number")
        elif expected == "boolean" and not isinstance(value, bool):
            errors.append(f"result key {key} must be boolean")
        elif expected == "array" and not isinstance(value, list):
            errors.append(f"result key {key} must be array")
        elif expected == "object" and not isinstance(value, dict):
            errors.append(f"result key {key} must be object")
        elif isinstance(expected, str) and "|" in expected:
            allowed = {part.strip() for part in expected.split("|") if part.strip()}
            actual = "true" if value is True else "false" if value is False else str(value)
            if actual not in allowed:
                errors.append(f"result key {key} must be one of {sorted(allowed)}")
    return errors


def _kanban_block_reason(conn: sqlite3.Connection, task_id: str) -> str:
    row = conn.execute(
        """
        SELECT payload FROM task_events
         WHERE task_id = ? AND kind IN ('blocked', 'dependency_wait', 'block_loop_detected', 'gave_up')
         ORDER BY id DESC LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if row is not None:
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        for key in ("reason", "error", "message"):
            reason = payload.get(key)
            if isinstance(reason, str) and reason:
                return reason
    row = conn.execute(
        "SELECT last_failure_error FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is not None and row["last_failure_error"]:
        return row["last_failure_error"]
    row = conn.execute(
        """
        SELECT error, summary FROM task_runs
         WHERE task_id = ?
           AND ((error IS NOT NULL AND error != '') OR (summary IS NOT NULL AND summary != ''))
         ORDER BY COALESCE(ended_at, started_at) DESC, id DESC LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if row is not None:
        return row["error"] or row["summary"]
    return "kanban task blocked"


def _block_agent_node(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    error: dict[str, Any],
    now: int,
) -> None:
    try:
        context = json.loads(row["context_json"] or "{}")
    except (TypeError, ValueError):
        context = {"node": {}}
    context["error"] = error
    sibling_refs: list[dict[str, Any]] = []
    with wfdb.write_txn(conn):
        updated = conn.execute(
            """
            UPDATE workflow_node_runs
               SET status = 'blocked', error = ?, completed_at = ?
             WHERE id = ? AND status = 'waiting'
            """,
            (_json_dumps(error), now, row["id"]),
        ).rowcount
        if updated:
            conn.execute(
                """
                UPDATE workflow_executions
                   SET status = 'blocked', context_json = ?, claim_lock = NULL,
                       claim_expires = NULL, updated_at = ?
                 WHERE execution_id = ? AND status = 'waiting'
                """,
                (_json_dumps(context), now, row["execution_id"]),
            )
            sibling_refs = _linked_waiting_kanban_task_refs(conn, row["execution_id"])
            if sibling_refs:
                sibling_ids = [ref["node_run_id"] for ref in sibling_refs]
                placeholders = ", ".join("?" for _ in sibling_ids)
                conn.execute(
                    f"""
                    UPDATE workflow_node_runs
                       SET status = 'blocked', error = ?, completed_at = ?, wait_until = NULL
                     WHERE id IN ({placeholders})
                    """,
                    (_json_dumps(error), now, *sibling_ids),
                )
            _append_event(conn, row["execution_id"], "execution_blocked", error, now)
    if sibling_refs:
        wfdb.block_linked_kanban_tasks(
            [(ref["task_id"], ref["kanban_board"]) for ref in sibling_refs],
            execution_id=row["execution_id"],
            source="agent_task_block",
            reason=f"workflow execution {row['execution_id']} blocked by node {row['node_id']}",
        )


def _resume_completed_agent_tasks(conn: sqlite3.Connection, *, now: int) -> None:
    rows = conn.execute(
        """
        SELECT nr.id, nr.execution_id, nr.node_id, nr.kanban_task_id, nr.kanban_board,
               ex.context_json, ex.workflow_id, ex.version
          FROM workflow_node_runs nr
          JOIN workflow_executions ex ON ex.execution_id = nr.execution_id
         WHERE nr.status = 'waiting'
           AND nr.kanban_task_id IS NOT NULL
           AND ex.status = 'waiting'
         ORDER BY nr.id
        """
    ).fetchall()
    if not rows:
        return

    for row in rows:
        with kb.connect_closing(board=row["kanban_board"]) as kconn:
            task = kb.get_task(kconn, row["kanban_task_id"])
            if task is None:
                continue
            if task.status == "done":
                output = _parse_agent_result(task.result or kb.latest_summary(kconn, task.id))
                spec = wfdb.get_definition(conn, row["workflow_id"], row["version"])
                node = spec.nodes.get(row["node_id"])
                contract = node.result_contract if node is not None else {}
                errors = _validate_result_contract(output, contract)
                if errors:
                    _block_agent_node(
                        conn,
                        row,
                        error={
                            "node_id": row["node_id"],
                            "kanban_task_id": task.id,
                            "reason": "; ".join(errors),
                        },
                        now=now,
                    )
                    continue
                with wfdb.write_txn(conn):
                    updated = conn.execute(
                        """
                        UPDATE workflow_node_runs
                           SET status = 'succeeded', output_json = ?, completed_at = ?
                         WHERE id = ? AND status = 'waiting'
                        """,
                        (_json_dumps(output), now, row["id"]),
                    ).rowcount
                    if updated:
                        conn.execute(
                            """
                            UPDATE workflow_executions
                               SET status = 'queued', claim_lock = NULL,
                                   claim_expires = NULL, updated_at = ?
                             WHERE execution_id = ? AND status = 'waiting'
                            """,
                            (now, row["execution_id"]),
                        )
            elif task.status == "blocked":
                _block_agent_node(
                    conn,
                    row,
                    error={
                        "node_id": row["node_id"],
                        "kanban_task_id": task.id,
                        "reason": _kanban_block_reason(kconn, task.id),
                    },
                    now=now,
                )


def _completed_wait_nodes(conn: sqlite3.Connection, execution_id: str) -> set[str]:
    return {
        row["node_id"]
        for row in conn.execute(
            """
            SELECT node_id FROM workflow_node_runs
             WHERE execution_id = ?
               AND status = 'succeeded'
               AND wait_until IS NOT NULL
            """,
            (execution_id,),
        )
    }


def _completed_node_outputs(conn: sqlite3.Connection, execution_id: str) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    rows = conn.execute(
        """
        SELECT node_id, output_json FROM workflow_node_runs
         WHERE execution_id = ?
           AND status = 'succeeded'
           AND output_json IS NOT NULL
        """,
        (execution_id,),
    ).fetchall()
    for row in rows:
        try:
            outputs[row["node_id"]] = json.loads(row["output_json"])
        except (TypeError, ValueError):
            continue
    return outputs


def _persist_waiting_nodes(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    result: EngineResult,
    spec: WorkflowSpec | None,
    now: int,
) -> None:
    if result.status != "waiting":
        return
    for node_id in result.waiting_nodes:
        node = spec.nodes.get(node_id) if spec is not None else None
        if node is not None and node.type == "wait":
            wait_until = now + node.seconds
        else:
            wait_until = None
        kanban_task_id = None
        kanban_board = None
        if spec is not None and node is not None and node.type == "agent_task":
            try:
                kanban_task_id, kanban_board = _create_or_get_agent_task(
                    execution_id=execution_id,
                    spec=spec,
                    node_id=node_id,
                    node=node,
                    context=result.context,
                )
            except Exception as exc:
                raise _AgentTaskMaterializationError(node_id, str(exc)) from exc
        exists = conn.execute(
            """
            SELECT id, kanban_task_id, kanban_board FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = ? AND status = 'waiting'
            """,
            (execution_id, node_id),
        ).fetchone()
        if exists is None:
            conn.execute(
                """
                INSERT INTO workflow_node_runs (
                    execution_id, node_id, status, started_at, wait_until, kanban_task_id, kanban_board
                ) VALUES (?, ?, 'waiting', ?, ?, ?, ?)
                """,
                (execution_id, node_id, now, wait_until, kanban_task_id, kanban_board),
            )
        elif kanban_task_id and (not exists["kanban_task_id"] or not exists["kanban_board"]):
            conn.execute(
                "UPDATE workflow_node_runs SET kanban_task_id = ?, kanban_board = ? WHERE id = ?",
                (kanban_task_id, kanban_board, exists["id"]),
            )


def _failed_node_id(result: EngineResult, spec: WorkflowSpec | None) -> str | None:
    if result.status != "failed" or spec is None or not result.error:
        return None
    node_id = result.error.get("node")
    if isinstance(node_id, str) and node_id in spec.nodes:
        return node_id
    return None


def _context_with_error(context: dict[str, Any], error: dict[str, Any]) -> dict[str, Any]:
    updated = dict(context)
    updated.setdefault("node", {})
    updated["error"] = error
    return updated


def _linked_waiting_kanban_task_refs(conn: sqlite3.Connection, execution_id: str) -> list[dict[str, Any]]:
    return [
        {
            "node_run_id": row["id"],
            "task_id": row["kanban_task_id"],
            "kanban_board": row["kanban_board"],
        }
        for row in conn.execute(
            """
            SELECT id, kanban_task_id, kanban_board
              FROM workflow_node_runs
             WHERE execution_id = ?
               AND status = 'waiting'
               AND kanban_task_id IS NOT NULL
            """,
            (execution_id,),
        ).fetchall()
    ]


def _materialization_failure_result(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    result: EngineResult,
    exc: _AgentTaskMaterializationError,
    now: int,
) -> EngineResult:
    error = {
        "message": str(exc),
        "node": exc.node_id,
        "phase": "agent_task_materialization",
    }
    _persist_failed_attempt(
        conn,
        execution_id=execution_id,
        node_id=exc.node_id,
        error=error,
        now=now,
    )
    linked_task_refs = _linked_waiting_kanban_task_refs(conn, execution_id)
    wfdb.block_linked_kanban_tasks(
        [(ref["task_id"], ref["kanban_board"]) for ref in linked_task_refs],
        execution_id=execution_id,
        source="agent_task_materialization",
        reason=f"workflow execution {execution_id} failed to create agent task {exc.node_id}: {exc}",
    )
    if linked_task_refs:
        linked_node_run_ids = [ref["node_run_id"] for ref in linked_task_refs]
        placeholders = ", ".join("?" for _ in linked_node_run_ids)
        conn.execute(
            f"""
            UPDATE workflow_node_runs
               SET status = 'blocked', error = ?, completed_at = ?, wait_until = NULL
             WHERE id IN ({placeholders})
            """,
            (_json_dumps(error), now, *linked_node_run_ids),
        )
    return EngineResult(
        status="failed",
        context=_context_with_error(result.context, error),
        waiting_nodes=[],
        error=error,
    )


def _persist_failed_attempt(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    node_id: str,
    error: dict[str, Any],
    now: int,
) -> None:
    queued = conn.execute(
        """
        SELECT id FROM workflow_node_runs
         WHERE execution_id = ? AND node_id = ? AND status = 'queued'
         ORDER BY id DESC LIMIT 1
        """,
        (execution_id, node_id),
    ).fetchone()
    if queued is not None:
        conn.execute(
            """
            UPDATE workflow_node_runs
               SET status = 'failed', error = ?, started_at = COALESCE(started_at, ?),
                   completed_at = ?, wait_until = NULL
             WHERE id = ?
            """,
            (_json_dumps(error), now, now, queued["id"]),
        )
        return
    conn.execute(
        """
        INSERT INTO workflow_node_runs (
            execution_id, node_id, status, error, started_at, completed_at
        ) VALUES (?, ?, 'failed', ?, ?, ?)
        """,
        (execution_id, node_id, _json_dumps(error), now, now),
    )


def _persist_successful_queued_attempts(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    result: EngineResult,
    now: int,
) -> None:
    for node_id, node_context in result.context.get("node", {}).items():
        output_json = None
        if isinstance(node_context, dict) and "output" in node_context:
            output_json = _json_dumps(node_context["output"])
        conn.execute(
            """
            UPDATE workflow_node_runs
               SET status = 'succeeded', output_json = ?, completed_at = ?, wait_until = NULL
             WHERE id = (
                SELECT id FROM workflow_node_runs
                 WHERE execution_id = ? AND node_id = ? AND status = 'queued'
                 ORDER BY id DESC LIMIT 1
             )
            """,
            (output_json, now, execution_id, node_id),
        )


def _failed_attempts(conn: sqlite3.Connection, execution_id: str, node_id: str) -> int:
    return int(conn.execute(
        """
        SELECT count(*) FROM workflow_node_runs
         WHERE execution_id = ? AND node_id = ? AND status = 'failed'
        """,
        (execution_id, node_id),
    ).fetchone()[0])


def _catch_resume_kwargs(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    context: dict[str, Any],
    spec: WorkflowSpec,
) -> dict[str, Any]:
    error = context.get("error")
    if not isinstance(error, dict):
        return {}
    node_id = error.get("node")
    if not isinstance(node_id, str) or node_id not in spec.nodes or not spec.nodes[node_id].catch:
        return {}
    queued_retry = conn.execute(
        """
        SELECT 1 FROM workflow_node_runs
         WHERE execution_id = ? AND node_id = ? AND status = 'queued'
         LIMIT 1
        """,
        (execution_id, node_id),
    ).fetchone()
    if queued_retry is not None:
        return {}
    return {"catch_failed_nodes": {node_id}, "error_context": error}


def _retry_due_at(node: Any, *, failed_attempts: int, now: int) -> int:
    retry = node.retry
    base = retry.backoff_seconds if retry.backoff_seconds is not None else retry.delay_seconds
    return int(now + base * (retry.multiplier ** max(0, failed_attempts - 1)))


def _emit_progress_events(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    result: EngineResult,
    spec: WorkflowSpec | None,
    now: int,
    existing_events: list[sqlite3.Row],
) -> None:
    emitted_nodes: set[str] = set()
    for event in existing_events:
        if event["kind"] != "node_succeeded":
            continue
        try:
            payload = json.loads(event["payload_json"])
        except (TypeError, ValueError):
            continue
        node_id = payload.get("node_id")
        if isinstance(node_id, str):
            emitted_nodes.add(node_id)

    _persist_waiting_nodes(
        conn,
        execution_id=execution_id,
        result=result,
        spec=spec,
        now=now,
    )
    if not existing_events:
        _append_event(conn, execution_id, "execution_started", {}, now)
    for node_id, node_context in result.context.get("node", {}).items():
        if node_id in emitted_nodes:
            continue
        output = node_context.get("output") if isinstance(node_context, dict) else None
        _append_event(
            conn,
            execution_id,
            "node_succeeded",
            {"node_id": node_id, "output": output},
            now,
        )
        emitted_nodes.add(node_id)


def _finish(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    token: str,
    result: EngineResult,
    spec: WorkflowSpec | None,
    now: int,
) -> bool:
    with wfdb.write_txn(conn):
        row = conn.execute(
            "SELECT claim_lock, input_json FROM workflow_executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if row is None or row["claim_lock"] != token:
            return False

        existing_events = conn.execute(
            "SELECT kind, payload_json FROM workflow_events WHERE execution_id = ?",
            (execution_id,),
        ).fetchall()

        node_id = _failed_node_id(result, spec)
        if node_id is not None and spec is not None:
            error = result.error or {}
            result.context = _context_with_error(result.context, error)
            _persist_failed_attempt(
                conn,
                execution_id=execution_id,
                node_id=node_id,
                error=error,
                now=now,
            )
            failed_attempts = _failed_attempts(conn, execution_id, node_id)
            node = spec.nodes[node_id]
            if node.retry is not None and failed_attempts < node.retry.max_attempts:
                due_at = _retry_due_at(node, failed_attempts=failed_attempts, now=now)
                status = "waiting" if due_at > now else "queued"
                conn.execute(
                    """
                    INSERT INTO workflow_node_runs (
                        execution_id, node_id, status, started_at, wait_until
                    ) VALUES (?, ?, 'queued', ?, ?)
                    """,
                    (execution_id, node_id, now, due_at),
                )
                _emit_progress_events(
                    conn,
                    execution_id=execution_id,
                    result=result,
                    spec=spec,
                    now=now,
                    existing_events=existing_events,
                )
                if status == "waiting":
                    _append_event(conn, execution_id, "execution_waiting", {"waiting_nodes": []}, now)
                conn.execute(
                    """
                    UPDATE workflow_executions
                       SET status = ?, context_json = ?, claim_lock = NULL,
                           claim_expires = NULL, updated_at = ?
                     WHERE execution_id = ? AND claim_lock = ?
                    """,
                    (status, _json_dumps(result.context), now, execution_id, token),
                )
                return True
            if node.catch:
                completed_wait_nodes = _completed_wait_nodes(conn, execution_id)
                completed_outputs = _completed_node_outputs(conn, execution_id)
                kwargs: dict[str, Any] = {
                    "catch_failed_nodes": {node_id},
                    "error_context": error,
                }
                if completed_wait_nodes:
                    kwargs["completed_wait_nodes"] = completed_wait_nodes
                if completed_outputs:
                    kwargs["completed_node_outputs"] = completed_outputs
                catch_context = _context_with_error(result.context, error)
                try:
                    result = run_in_memory_until_waiting(
                        spec,
                        json.loads(row["input_json"]),
                        **kwargs,
                    )
                except Exception as exc:
                    result = EngineResult(
                        status="failed",
                        context=catch_context,
                        waiting_nodes=[],
                        error={
                            "message": str(exc),
                            "catch_node": node.catch,
                            "caught_node": node_id,
                        },
                    )
                else:
                    result.context = _context_with_error(result.context, error)

        final_event = {
            "succeeded": "execution_succeeded",
            "waiting": "execution_waiting",
            "failed": "execution_failed",
        }[result.status]
        final_payload: dict[str, Any] = {}
        if result.status == "waiting":
            final_payload = {"waiting_nodes": result.waiting_nodes}
        elif result.status == "failed":
            final_payload = {"error": result.error or {}}

        _persist_successful_queued_attempts(
            conn,
            execution_id=execution_id,
            result=result,
            now=now,
        )
        try:
            _emit_progress_events(
                conn,
                execution_id=execution_id,
                result=result,
                spec=spec,
                now=now,
                existing_events=existing_events,
            )
        except _AgentTaskMaterializationError as exc:
            result = _materialization_failure_result(
                conn,
                execution_id=execution_id,
                result=result,
                exc=exc,
                now=now,
            )
            final_event = "execution_failed"
            final_payload = {"error": result.error or {}}
        _append_event(conn, execution_id, final_event, final_payload, now)
        conn.execute(
            """
            UPDATE workflow_executions
               SET status = ?, context_json = ?, claim_lock = NULL,
                   claim_expires = NULL, updated_at = ?
             WHERE execution_id = ? AND claim_lock = ?
            """,
            (result.status, _json_dumps(result.context), now, execution_id, token),
        )
    return True


def tick(
    *,
    db_path: Path | None = None,
    limit: int = 10,
    now: int | None = None,
    lease_seconds: int = 60,
) -> int:
    """Advance up to limit queued cheap workflow executions. Return number processed."""
    if limit <= 0:
        return 0

    tick_now = int(time.time()) if now is None else now
    processed = 0
    wfdb.init_db(db_path)
    with wfdb.connect(db_path) as conn:
        _fire_due_schedules(conn, now=tick_now)
        _resume_due_waits(conn, now=tick_now)
        _resume_due_retries(conn, now=tick_now)
        _resume_completed_agent_tasks(conn, now=tick_now)
        while processed < limit:
            claimed = _claim_next(conn, now=tick_now, lease_seconds=lease_seconds)
            if claimed is None:
                break
            execution_id, token = claimed
            execution = None
            spec = None
            try:
                execution = wfdb.get_execution(conn, execution_id)
                spec = wfdb.get_definition(conn, execution.workflow_id, execution.version)
                completed_wait_nodes = _completed_wait_nodes(conn, execution_id)
                completed_outputs = _completed_node_outputs(conn, execution_id)
                kwargs: dict[str, Any] = _catch_resume_kwargs(
                    conn,
                    execution_id=execution_id,
                    context=execution.context,
                    spec=spec,
                )
                if completed_wait_nodes:
                    kwargs["completed_wait_nodes"] = completed_wait_nodes
                if completed_outputs:
                    kwargs["completed_node_outputs"] = completed_outputs
                result = run_in_memory_until_waiting(spec, execution.input, **kwargs)
            except Exception as exc:
                context = execution.context if execution is not None else {"node": {}}
                result = EngineResult(
                    status="failed",
                    context=context,
                    waiting_nodes=[],
                    error={"message": str(exc)},
                )
            if _finish(
                conn,
                execution_id=execution_id,
                token=token,
                result=result,
                spec=spec,
                now=tick_now,
            ):
                processed += 1
    return processed
