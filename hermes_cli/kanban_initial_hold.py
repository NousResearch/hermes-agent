"""Initial-hold identity for conditional unblock under the native write lock."""
import json


def matches(conn, task_id: str, expected_key: str) -> bool:
    """Caller owns write_txn; a prior show must never authorize a later hold."""
    if not isinstance(expected_key, str) or not expected_key.strip():
        return False
    row = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
    if (row is None or row['idempotency_key'] != expected_key or row['status'] != 'blocked'
            or any(row[key] for key in ('current_run_id', 'worker_pid', 'claim_lock',
                'claim_expires', 'worker_started_at', 'block_kind', 'block_recurrences', 'consecutive_failures',
                'last_failure_error', 'last_heartbeat_at'))):
        return False
    if conn.execute('SELECT 1 FROM task_runs WHERE task_id=? LIMIT 1', (task_id,)).fetchone():
        return False
    events = conn.execute("""SELECT kind,payload,run_id FROM task_events WHERE task_id=?
        AND kind IN ('blocked','unblocked','scheduled')""", (task_id,)).fetchall()
    if len(events) != 1 or events[0]['kind'] != 'blocked' or events[0]['run_id'] is not None:
        return False
    try:
        payload = json.loads(events[0]['payload'] or 'null')
    except (ValueError, TypeError):
        return False
    return (isinstance(payload, dict) and payload.get('reason') == 'initial_status'
            and payload.get('status') == 'blocked')
