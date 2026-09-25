"""Read-only refusal gate; never orchestrates or grants Output retirement."""
import json
from hermes_state_runtime import RuntimeStoreError


def require_room_retired(conn, room_id, *, unavailable=False, sealed_inventory=False):
    reason = 'runtime_coordination_required' if unavailable else 'output_cleanup_pending'
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    def refuse():
        raise RuntimeStoreError(reason)
    from gateway.hosted_room_task_scan import pending
    if not sealed_inventory and pending(conn, room_id):
        refuse()
    if 'session_admissions' in tables:
        # Retained canonical admissions are stronger evidence than driver inactivity.
        for row in conn.execute("SELECT request_id FROM session_admissions WHERE status IN ('started','unknown','queued') AND request_id LIKE 'hosted:%'"):
            try:
                identity, generation = json.loads(row[0][7:])
            except (ValueError, TypeError):
                refuse()
            if identity.get('room_id') == room_id:
                refuse()
    if 'hosted_room_driver_tasks' in tables:
        for row in conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE room_id=?', (room_id,)):
            if row['status'] in {'queued', 'running', 'stopping', 'indeterminate', 'deferred'}:
                refuse()
            result = json.loads(row['result_json']) if row['result_json'] else {}
            if result.get('artifacts'):
                if 'hosted_room_artifact_completions' not in tables or conn.execute(
                        'SELECT 1 FROM hosted_room_artifact_completions WHERE room_id=? AND task_id=? AND execution_generation=?',
                        (room_id, row['task_id'], row['execution_generation'])).fetchone() is None:
                    refuse()
    if 'hosted_room_artifact_retries' in tables and conn.execute(
            'SELECT 1 FROM hosted_room_artifact_retries WHERE room_id=? LIMIT 1', (room_id,)).fetchone():
        refuse()
    if 'hosted_room_output_artifacts' in tables and conn.execute(
            "SELECT 1 FROM hosted_room_output_artifacts WHERE json_extract(scope_json,'$.room_id')=? "
            'AND (acknowledged_at IS NULL OR cleanup_required_at IS NOT NULL OR blob_reclaimed_at IS NULL) LIMIT 1',
            (room_id,)).fetchone():
        refuse()
    if 'state_meta' in tables and conn.execute(
            "SELECT 1 FROM state_meta WHERE key LIKE 'gateway.hosted.output_cleanup.v1:%' "
            "AND json_extract(value,'$.room_id')=? AND json_extract(value,'$.state') IS NOT 'completed' LIMIT 1",
            (room_id,)).fetchone():
        refuse()
