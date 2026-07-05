"""SQLite dispatcher for cheap local workflow nodes."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from hermes_cli import workflows_db as wfdb
from hermes_cli.workflows_engine import EngineResult, run_in_memory_until_waiting


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


def _finish(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    token: str,
    result: EngineResult,
    now: int,
) -> bool:
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

    with wfdb.write_txn(conn):
        row = conn.execute(
            "SELECT claim_lock FROM workflow_executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if row is None or row["claim_lock"] != token:
            return False

        _append_event(conn, execution_id, "execution_started", {}, now)
        for node_id, node_context in result.context.get("node", {}).items():
            output = node_context.get("output") if isinstance(node_context, dict) else None
            _append_event(
                conn,
                execution_id,
                "node_succeeded",
                {"node_id": node_id, "output": output},
                now,
            )
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
    with wfdb.connect(db_path) as conn:
        while processed < limit:
            claimed = _claim_next(conn, now=tick_now, lease_seconds=lease_seconds)
            if claimed is None:
                break
            execution_id, token = claimed
            execution = None
            try:
                execution = wfdb.get_execution(conn, execution_id)
                spec = wfdb.get_definition(conn, execution.workflow_id, execution.version)
                result = run_in_memory_until_waiting(spec, execution.input)
            except Exception as exc:
                context = execution.context if execution is not None else {"node": {}}
                result = EngineResult(
                    status="failed",
                    context=context,
                    waiting_nodes=[],
                    error={"message": str(exc)},
                )
            if _finish(conn, execution_id=execution_id, token=token, result=result, now=tick_now):
                processed += 1
    return processed
