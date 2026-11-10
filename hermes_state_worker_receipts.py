"""Read-only reconciliation of an uncertain worker mutation through its owner."""
import json


def read_worker_receipt(db, *, epoch, execution_id, session_id, generation, sequence,
                        operation, payload):
    from hermes_state_runtime import RuntimeStoreError, _epoch, _json
    from hermes_state_keys import admission_fingerprint
    from hermes_state_worker_compression import worker_receipt_assignment
    from hermes_state_runtime_workers import retired_receipt
    if type(sequence) is not int or sequence < 1 or not isinstance(operation, str):
        raise RuntimeStoreError('invalid_params')
    encoded = _json(payload)
    if len(encoded.encode('utf-8', errors='surrogatepass')) > 4 * 1024 * 1024:
        raise RuntimeStoreError('invalid_params')
    digest = admission_fingerprint(canonical_target=session_id,
        payload={'operation': operation, 'payload': json.loads(encoded)})
    with db._read_ctx() as conn:
        # One snapshot: an epoch replacement or rotation cannot split the fence checks.
        conn.execute('BEGIN')
        try:
            _epoch(conn, epoch)
            row = worker_receipt_assignment(conn, execution_id, session_id, generation, sequence, digest)
            if row['owner_epoch'] != epoch:
                raise RuntimeStoreError('stale_epoch')
            old = conn.execute('SELECT * FROM worker_receipts WHERE execution_id=? AND sequence=?',
                               (execution_id, sequence)).fetchone()
            if old is None:
                raise RuntimeStoreError('not_found')
            if old['payload_digest'] != digest:
                raise RuntimeStoreError('admission_conflict')
            if retired_receipt(old['result_json']):
                raise RuntimeStoreError('stale_generation')
            return json.loads(old['result_json'])
        finally:
            conn.rollback()
