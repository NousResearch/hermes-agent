"""A worker's sanitized provider verdict, fenced to its active native run.

The process exit code is lossy (and may disappear on dispatcher restart). An
append-only run event carries the verdict to the ordinary dead-worker reaper.
"""

from __future__ import annotations

import os
from contextlib import closing

TRANSIENT_REASONS = frozenset({
    "rate_limit",
    "upstream_rate_limit",
    "overloaded",
    "server_error",
    "timeout",
})
STOP_REASONS = frozenset({
    "billing",
    "auth",
    "auth_permanent",
    "provider_policy_blocked",
    "content_policy_blocked",
    "model_not_found",
    "ssl_cert_verification",
})


def report_provider_failure(result: dict) -> None:
    from agent.delegation_context import owned_kanban_task
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    task_id = owned_kanban_task()
    reason = result.get("failure_reason")
    if not task_id or reason not in TRANSIENT_REASONS | STOP_REASONS:
        return
    raw_run = os.environ.get("HERMES_KANBAN_RUN_ID", "")
    claim = os.environ.get("HERMES_KANBAN_CLAIM_LOCK", "")
    if not raw_run.isdecimal() or not claim:
        return
    run_id = int(raw_run)
    # Never copy exception text, headers, provider URLs, credentials or messages.
    evidence = {
        "schema_version": 1,
        "reason": reason,
        "transient": reason in TRANSIENT_REASONS
        or (reason == "billing" and result.get("billing_unverified") is True),
        "pid": os.getpid(),
    }
    code = result.get("failure_status_code")
    if type(code) is int and 400 <= code <= 599:
        evidence["http_status"] = code
    # Legacy launches retain their launch-window guard. Supervised workers
    # must already have the distinct child identity persisted before exec.
    with closing(kbc.connect()) as conn, kb.write_txn(conn):
        row = conn.execute(
            """SELECT t.worker_pid,r.execution_scope FROM tasks t JOIN task_runs r ON r.id=t.current_run_id
            WHERE t.id=? AND t.status='running' AND t.current_run_id=?
            AND t.claim_lock=?
            AND r.task_id=t.id AND r.ended_at IS NULL""",
            (task_id, run_id, claim),
        ).fetchone()
        if not row:
            return
        if row['execution_scope'] is not None:
            from hermes_cli.kanban_execution_scope import CONTRACT
            from hermes_cli.kanban_db_dispatch import _process_fingerprint
            scope = kb._json_dict(row['execution_scope'])
            fingerprint = _process_fingerprint(os.getpid())
            if (scope.get('contract') != CONTRACT or scope.get('state') != 'active'
                    or scope.get('worker_pid') != os.getpid() or not fingerprint
                    or scope.get('worker_fingerprint') != fingerprint):
                return
            evidence.update(scope_id=scope['id'], worker_fingerprint=fingerprint)
        elif row['worker_pid'] not in (None, os.getpid()):
            return
        if not kb._latest_event(conn, task_id, "provider_failure", run_id):
            kb._append_event(conn, task_id, "provider_failure", evidence, run_id=run_id)


def provider_verdict(conn, task_id: str, pid: int):
    """Read only the report belonging to this still-active run and process."""
    from hermes_cli import kanban_db as kb

    row = conn.execute(
        """SELECT e.payload,r.execution_scope FROM tasks t JOIN task_runs r ON r.id=t.current_run_id
        JOIN task_events e
        ON e.task_id=t.id AND e.run_id=t.current_run_id
        WHERE t.id=? AND t.worker_pid=? AND e.kind='provider_failure'
        ORDER BY e.id LIMIT 1""",
        (task_id, pid),
    ).fetchone()
    if row is None:
        return None
    evidence = kb._json_dict(row["payload"])
    expected_pid = pid
    if row['execution_scope'] is not None:
        from hermes_cli.kanban_execution_scope import CONTRACT
        scope = kb._json_dict(row['execution_scope'])
        if (scope.get('contract') != CONTRACT or scope.get('supervisor_pid') != pid
                or not scope.get('worker_fingerprint')
                or evidence.get('scope_id') != scope.get('id')
                or evidence.get('worker_fingerprint') != scope['worker_fingerprint']):
            return None
        expected_pid = scope.get('worker_pid')
    if (
        evidence.get("schema_version") != 1
        or evidence.get("pid") != expected_pid
        or evidence.get("reason") not in TRANSIENT_REASONS | STOP_REASONS
    ):
        return None
    return evidence


def closed_retry_runs(conn, task_id: str, limit: int):
    """Closed attempts after the last explicit unblock, ordered newest first.

    Event IDs establish the boundary even when a block, unblock and new claim
    all happen in one second. No run/event is rewritten to open a new window.
    """
    cutoff = conn.execute(
        "SELECT max(id) AS id FROM task_events WHERE task_id=? AND kind='unblocked'",
        (task_id,),
    ).fetchone()["id"]
    return conn.execute(
        """SELECT r.* FROM task_runs r WHERE r.task_id=? AND r.ended_at IS NOT NULL
           AND (? IS NULL OR (
               SELECT min(e.id) FROM task_events e WHERE e.task_id=r.task_id AND e.run_id=r.id)>?)
           ORDER BY r.id DESC LIMIT ?""",
        (task_id, cutoff, cutoff, limit),
    ).fetchall()


def transient_budget_exhausted(conn, task_id: str) -> bool:
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import DEFAULT_FAILURE_LIMIT

    row = conn.execute(
        "SELECT max_retries FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    limit = (
        row["max_retries"] if row["max_retries"] is not None else DEFAULT_FAILURE_LIMIT
    )
    # Count this failing attempt plus the immediately preceding provider failures.
    streak = 1
    for run in closed_retry_runs(conn, task_id, max(0, limit)):
        evidence = kb._json_dict(run["metadata"]).get("provider_failure", {})
        if evidence.get("transient") is not True:
            break
        streak += 1
    return streak >= limit


def protocol_breaker_tripped(conn, task_id: str) -> bool:
    """The latest breaker decision survives until an explicit unblock.

    A forced protocol breaker counts one unified failure, not the protocol
    streak. Re-evaluating only the unified count would immediately undo it.
    """
    from hermes_cli import kanban_db as kb

    row = conn.execute(
        """SELECT kind,payload FROM task_events WHERE task_id=?
        AND kind IN ('gave_up','unblocked') ORDER BY id DESC LIMIT 1""",
        (task_id,),
    ).fetchone()
    return bool(
        row
        and row["kind"] == "gave_up"
        and kb._json_dict(row["payload"]).get("protocol_violation_limit") is not None
    )
