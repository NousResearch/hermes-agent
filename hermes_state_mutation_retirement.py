"""Durable deletion fences outside prunable transcript rows."""
import json
from hermes_state_runtime import RuntimeStoreError, admission_fingerprint, _json, _text

RETIRED_PREFIX = 'gateway.retired_session.v1.'


def retired_session(db, session_id):
    with db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                            (RETIRED_PREFIX + session_id,)).fetchone() is not None


def has_mutation_receipt(db, principal_id, session_id, request_id):
    for value in (principal_id, session_id, request_id):
        _text(value)
    key = 'gateway.mutation.v1.' + admission_fingerprint(
        canonical_target=session_id, payload={'principal': principal_id, 'request': request_id})
    with db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key=?', (key,)).fetchone() is not None


def retire_terminal_receipts(conn, session_ids):
    from hermes_state_terminal import ADMISSION_PREFIX, WORKER_PREFIX, identity_key
    for sid in session_ids:
        admissions = conn.execute('SELECT * FROM session_admissions WHERE target_session_id=?', (sid,)).fetchall()
        workers = conn.execute('SELECT * FROM worker_executions WHERE session_id=?', (sid,)).fetchall()
        states = {r['status'] for r in [*admissions, *workers]} - {'terminal'}
        if states:
            raise RuntimeStoreError('unknown_execution' if 'unknown' in states else 'session_busy')
        for raw in admissions:
            row = dict(raw)
            # Keep the digest for exact retries, not another copy of user input/history.
            row['payload_json'] = '{}'
            row['lineage_json'] = '[]'
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                         (ADMISSION_PREFIX + row['admission_id'], _json(row)))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                         (identity_key(row['principal_id'], sid, row['request_id']), json.dumps(row['admission_id'])))
        for raw in workers:
            row = dict(raw)
            receipts = [dict(r) for r in conn.execute(
                'SELECT sequence,payload_digest,result_json FROM worker_receipts WHERE execution_id=? ORDER BY sequence',
                (row['execution_id'],))]
            # A terminal worker can only replay its closing receipt. Earlier results
            # (history/context reads) are user data that must not outlive the delete;
            # their digests stay so a late duplicate is still recognised as a conflict.
            for receipt in receipts[:-1]:
                receipt['result_json'] = None
            row['receipts'] = receipts
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                         (WORKER_PREFIX + row['execution_id'], _json(row)))
            conn.execute('DELETE FROM worker_receipts WHERE execution_id=?', (row['execution_id'],))
        conn.execute('DELETE FROM worker_executions WHERE session_id=?', (sid,))
        conn.execute('DELETE FROM session_admissions WHERE target_session_id=?', (sid,))


def delete_in_transaction(db, conn, session_id, payload):
    from hermes_state_mutation_guards import require_idle, delete_targets
    targets = delete_targets(conn, session_id)
    require_idle(db, conn, targets)
    retire_terminal_receipts(conn, targets)
    retire_routes(conn, targets)
    for sid in targets:
        db._bump_conversation_generation(conn, sid, 'session_reset')
        conn.execute('UPDATE sessions SET parent_session_id=NULL, runtime_revision=runtime_revision+1 WHERE parent_session_id=?', (sid,))
        conn.execute('DELETE FROM messages WHERE session_id=?', (sid,))
    conn.executemany('DELETE FROM sessions WHERE id=?', [(sid,) for sid in targets])
    db._delete_unreferenced_system_prompts(conn)
    return set(), {'deleted_ids': targets}


def retire_routes(conn, session_ids):
    for sid in session_ids:
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (RETIRED_PREFIX + sid, '{}'))
    targets = set(session_ids)
    for row in conn.execute('SELECT scope,session_key,entry_json FROM gateway_routing').fetchall():
        if json.loads(row['entry_json']).get('session_id') in targets:
            conn.execute('DELETE FROM gateway_routing WHERE scope=? AND session_key=?',
                         (row['scope'], row['session_key']))
