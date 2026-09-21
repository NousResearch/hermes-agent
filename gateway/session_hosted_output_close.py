"""Cleanup completion seal: retained facts, never post-revocation artifact rights."""
import json
from gateway.session_group_retirement import require_room_retired
from gateway.session_hosted_output_retry import digest
from hermes_state_runtime import RuntimeStoreError


def inventory(conn, room_id):
    # A seal may survive legitimate deletion of completed driver/receipt rows,
    # but never a new or changed attempt, publication, or pending obligation.
    result = {}
    for table, key in (('hosted_room_driver_tasks', 'task_id'),
                       ('hosted_room_artifact_completions', 'task_id')):
        result[table] = {r[key]: digest(dict(r)) for r in conn.execute(
            f'SELECT * FROM {table} WHERE room_id=?', (room_id,))}
    result['cleanup'] = {r['key']: digest(json.loads(r['value'])) for r in conn.execute(
        "SELECT key,value FROM state_meta WHERE key LIKE 'gateway.hosted.output_cleanup.v1:%' "
        "AND json_extract(value,'$.room_id')=?", (room_id,))}
    return result


def capture(service, conn, room_id):
    service._output_owner(conn)
    require_room_retired(conn, room_id)
    return inventory(conn, room_id)


def require(service, conn, room_id, saved):
    service._output_owner(conn)
    current = inventory(conn, room_id)
    if any(any(saved[table].get(k) != value for k, value in rows.items())
           for table, rows in current.items()):
        raise RuntimeStoreError('output_cleanup_pending')
    # Missing completed rows can be normal pruning. The sealed full inventory,
    # not a bounded scan's absence, supplies the original negative proof.
    require_room_retired(conn, room_id, sealed_inventory=True)
