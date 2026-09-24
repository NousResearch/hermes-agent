"""Non-destructive stopped-Output facts, compacted at completion/retirement.

These digests recognise completion only. They contain no blob names, input bytes
or authority to unlink, execute or adopt another owner's generation.
"""
import hashlib
import json


_IDENTITY_FIELDS = (
    'version', 'room_id', 'task_id', 'member_id', 'execution_generation',
    'cancel_generation', 'cancel_id', 'identity', 'payload', 'owner', 'roster',
    'epoch', 'instance', 'stop_event',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def completion_identity(binding):
    return digest({**{key: binding[key] for key in _IDENTITY_FIELDS}, 'scope': binding.get('scope')})


def compact_completed(record):
    if record['state'] != 'completed' or record['version'] not in {1, 2}:
        raise ValueError('invalid Output completion')
    if record['version'] == 2:
        return record
    binding = record['binding']
    # Preserve the exact canonical tuple/digest, never its payload or input paths.
    admission = binding.get('admission')
    return dict(version=2, **{key: record[key] for key in (
        'room_id', 'task_id', 'member_id', 'execution_generation', 'state',
        'reason_code', 'attempts', 'next_attempt_at', 'removed')},
        completion_identity=completion_identity(binding), binding_digest=digest(binding),
        admission=({key: value for key, value in admission.items() if key != 'payload_json'}
                   if admission is not None else None),
        disposition=record.get('completion', {}).get('operation', 'local_cleanup'),
        disposition_digest=digest(record.get('completion', {'operation': 'local_cleanup'})))


def compact_task_completions(conn, room_id, task_ids):
    rows = conn.execute(f"""SELECT key,value FROM state_meta
        WHERE key LIKE 'gateway.hosted.output_cleanup.v1:%'
        AND json_extract(value,'$.state')='completed'
        AND json_extract(value,'$.room_id')=?
        AND json_extract(value,'$.task_id') IN ({','.join('?' for _ in task_ids)})""",
        (room_id, *task_ids)).fetchall()
    for row in rows:
        conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                     (json.dumps(compact_completed(json.loads(row['value'])), sort_keys=True), row['key']))


def compact_retired_completions(conn, session_id):
    rows = conn.execute("""SELECT key,value FROM state_meta
        WHERE key LIKE 'gateway.hosted.output_cleanup.v1:%'
        AND json_extract(value,'$.state')='completed'
        AND json_extract(value,'$.binding.admission.target_session_id')=?""", (session_id,)).fetchall()
    for row in rows:
        conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                     (json.dumps(compact_completed(json.loads(row['value'])), sort_keys=True), row['key']))
