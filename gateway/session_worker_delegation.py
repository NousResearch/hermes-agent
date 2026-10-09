"""Closed delegation-ledger operations in the existing worker receipt transaction.

The caller must supply the verified registration's execution/PID/birth, never
operation payload identity. No handler opens a database or commits independently.
"""
from functools import partial
import json
import math
import time

from hermes_state_runtime import RuntimeStoreError

_TASK_KEYS = {'goal', 'goals', 'context', 'toolsets', 'role', 'model', 'is_batch', 'task_indexes'}
_ROUTE = {'session_key': 'origin_session', 'parent_session_id': 'parent_session_id',
          'origin_ui_session_id': 'origin_ui_session_id', 'origin_session_id': 'origin_session_id'}
_EVENT_KEYS = _TASK_KEYS | set(_ROUTE) | {
    'type', 'delegation_id', 'status', 'completed_at', 'dispatched_at', 'summary', 'error',
    'api_calls', 'duration_seconds', 'total_duration_seconds', 'results', 'live_transcripts',
    'group', 'exit_reason', 'stalled_after_quiet_seconds', 'stall_threshold_seconds',
    'stall_phase', 'stall_grace_seconds'}


def _shape(payload, keys):
    if not isinstance(payload, dict) or set(payload) != set(keys):
        raise RuntimeStoreError('invalid_params')


def _text(value):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise RuntimeStoreError('invalid_params')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise RuntimeStoreError('invalid_params')


def _owned(conn, session_id, execution_id, delegation_id):
    _text(delegation_id)
    row = conn.execute('SELECT * FROM async_delegations WHERE delegation_id=?', (delegation_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    if row['parent_session_id'] != session_id or row['owner_execution_id'] != execution_id:
        raise RuntimeStoreError('permission_denied')
    return row


def _dispatch(db, conn, session_id, payload, *, execution_id, worker_pid, worker_birth):
    _shape(payload, {'delegation_id', 'task', 'dispatched_at', *_ROUTE})
    _text(payload['delegation_id'])
    _number(payload['dispatched_at'])
    if not isinstance(payload['task'], dict) or set(payload['task']) - _TASK_KEYS:
        raise RuntimeStoreError('invalid_params')
    session = conn.execute('SELECT source,session_key FROM sessions WHERE id=?', (session_id,)).fetchone()
    # Only compute registrations are currently enabled. Native/child origins need
    # an owner-reserved source envelope, not worker-selected transport identities.
    if session is None or session['source'] not in ('cli', 'tui', 'gui'):
        raise RuntimeStoreError('unsupported_producer')
    if (payload['parent_session_id'] != session_id
            or payload['session_key'] != (session['session_key'] or '')
            or payload['origin_ui_session_id'] not in ('', session_id)
            or payload['origin_session_id'] not in ('', session_id)):
        raise RuntimeStoreError('permission_denied')
    old = conn.execute('SELECT 1 FROM async_delegations WHERE delegation_id=?', (payload['delegation_id'],)).fetchone()
    if old is not None:
        _owned(conn, session_id, execution_id, payload['delegation_id'])
        # Exact network retries are resolved by worker_receipts before this handler.
        raise RuntimeStoreError('admission_conflict')
    import psutil
    from gateway.status import get_process_start_time
    try:
        if psutil.Process(worker_pid).create_time() != worker_birth:
            raise RuntimeStoreError('permission_denied')
        owner_started_at = get_process_start_time(worker_pid)
    except psutil.Error as exc:
        raise RuntimeStoreError('worker_not_live') from exc
    now = time.time()
    conn.execute('''INSERT INTO async_delegations
        (delegation_id,origin_session,origin_ui_session_id,parent_session_id,state,dispatched_at,
         updated_at,owner_pid,owner_started_at,task_json,origin_session_id,owner_execution_id)
        VALUES(?,?,?,?,'running',?,?,?,?,?,?,?)''',
        (payload['delegation_id'], payload['session_key'], payload['origin_ui_session_id'], session_id,
         payload['dispatched_at'], now, worker_pid, owner_started_at, json.dumps(payload['task']),
         payload['origin_session_id'], execution_id))
    return {'value': None}


def _complete(db, conn, session_id, payload, *, execution_id):
    _shape(payload, {'event', 'result'})
    event = payload['event']
    if (not isinstance(event, dict) or set(event) - _EVENT_KEYS
            or not isinstance(payload['result'], dict) or event.get('type') != 'async_delegation'
            or event.get('status') not in ('completed', 'success', 'error', 'interrupted', 'stalled', 'unknown')):
        raise RuntimeStoreError('invalid_params')
    row = _owned(conn, session_id, execution_id, event.get('delegation_id'))
    if any(event.get(key, '') != (row[column] or '') for key, column in _ROUTE.items()):
        raise RuntimeStoreError('permission_denied')
    _number(event.get('completed_at'))
    if row['state'] != 'running':
        raise RuntimeStoreError('admission_conflict')
    conn.execute('''UPDATE async_delegations SET state=?,completed_at=?,updated_at=?,event_json=?,result_json=?
        WHERE delegation_id=?''', (event['status'], event['completed_at'], time.time(),
        json.dumps(event), json.dumps(payload['result']), event['delegation_id']))
    return {'value': None}


def _child(db, conn, session_id, payload, *, execution_id):
    _shape(payload, {'delegation_id', 'entry'})
    row = _owned(conn, session_id, execution_id, payload['delegation_id'])
    entry = payload['entry']
    if not isinstance(entry, dict) or type(entry.get('task_index')) is not int or entry['task_index'] < 0:
        raise RuntimeStoreError('invalid_params')
    if row['state'] != 'running':
        return {'value': None}
    previous = json.loads(row['result_json'] or '{}')
    results = [item for item in previous.get('results', []) if item.get('task_index') != entry['task_index']]
    results.append(entry)
    conn.execute('UPDATE async_delegations SET result_json=?,updated_at=? WHERE delegation_id=?',
        (json.dumps({'results': results, 'partial': True}), time.time(), payload['delegation_id']))
    return {'value': None}


def _read(db, conn, session_id, payload, *, execution_id):
    _shape(payload, {'delegation_id'})
    row = _owned(conn, session_id, execution_id, payload['delegation_id'])
    return {'value': {key: row[key] for key in ('delegation_id', 'origin_session', 'state', 'dispatched_at',
        'completed_at', 'delivery_state', 'delivery_attempts', 'origin_session_id')} |
        {'result': json.loads(row['result_json']) if row['result_json'] else None}}


def _claim(db, conn, session_id, payload, *, execution_id):
    _shape(payload, {'delegation_id', 'claim_id'})
    row = _owned(conn, session_id, execution_id, payload['delegation_id'])
    _text(payload['claim_id'])
    now = time.time()
    if row['state'] in ('running', 'finalizing'):
        return {'value': False}
    changed = conn.execute('''UPDATE async_delegations SET delivery_claim=?,delivery_claimed_at=?,
        delivery_attempts=delivery_attempts+1,updated_at=? WHERE delegation_id=? AND delivery_state='pending'
        AND (delivery_claim IS NULL OR delivery_claimed_at < ?)''',
        (payload['claim_id'], now, now, payload['delegation_id'], now - 300)).rowcount
    return {'value': changed == 1}


_DELIVERY_UPDATES = {
    'ack': "delivery_state='delivered',delivered_at=:now",
    'drop': "delivery_state='dropped'",
    'defer': 'delivery_attempts=MAX(0,delivery_attempts-1)',
    'release': "delivery_state=CASE WHEN delivery_attempts>=:cap THEN 'dropped' ELSE delivery_state END",
}


def _delivery(db, conn, session_id, payload, *, execution_id, action):
    from tools.async_delegation import _MAX_DELIVERY_ATTEMPTS
    _shape(payload, {'delegation_id', 'claim_id'})
    _owned(conn, session_id, execution_id, payload['delegation_id'])
    _text(payload['claim_id'])
    # Only constant, closed statement fragments enter this query.
    changed = conn.execute('UPDATE async_delegations SET ' + _DELIVERY_UPDATES[action] +
        ",updated_at=:now,delivery_claim=NULL,delivery_claimed_at=NULL WHERE delegation_id=:delegation_id "
        "AND delivery_state='pending' AND delivery_claim=:claim_id",
        payload | {'now': time.time(), 'cap': _MAX_DELIVERY_ATTEMPTS}).rowcount
    return {'value': changed == 1}


def _unscheduled(db, conn, session_id, payload, *, execution_id):
    _shape(payload, {'delegation_id'})
    row = _owned(conn, session_id, execution_id, payload['delegation_id'])
    if row['state'] != 'running' or row['result_json'] is not None:
        raise RuntimeStoreError('admission_conflict')
    conn.execute('DELETE FROM async_delegations WHERE delegation_id=?', (payload['delegation_id'],))
    return {'value': None}


def worker_delegation_handlers(execution_id, worker_pid, worker_birth):
    """Register these handlers only after gateway worker identity verification."""
    _text(execution_id)
    if type(worker_pid) is not int or worker_pid <= 0:
        raise RuntimeStoreError('invalid_params')
    _number(worker_birth)
    handlers = {name: partial(fn, execution_id=execution_id) for name, fn in {
        'complete': _complete, 'child': _child, 'read': _read, 'claim': _claim,
        'unscheduled': _unscheduled}.items()}
    handlers['dispatch'] = partial(_dispatch, execution_id=execution_id, worker_pid=worker_pid, worker_birth=worker_birth)
    handlers.update({name: partial(_delivery, execution_id=execution_id, action=name) for name in _DELIVERY_UPDATES})
    return {'delegation.' + name: handler for name, handler in handlers.items()}


def persist_worker_delegation(db, *, epoch, execution_id, session_id, generation,
                              sequence, operation, payload, worker_pid, worker_birth):
    """Closed ledger sibling of mutate_worker_execution, sharing its receipt stream.

    Gateway calls this ONLY after its ordinary worker claim verification. The
    verified process identity is deliberately separate from the operation payload.
    """
    from gateway.session_admission import admission_fingerprint
    from hermes_state_runtime import _epoch, _worker_assignment, _json
    handlers = worker_delegation_handlers(execution_id, worker_pid, worker_birth)
    if type(sequence) is not int or sequence < 1 or not isinstance(operation, str) or operation not in handlers:
        raise RuntimeStoreError('invalid_params')
    encoded = _json(payload)
    if len(encoded.encode('utf-8', errors='surrogatepass')) > 4 * 1024 * 1024:
        raise RuntimeStoreError('invalid_params')
    digest = admission_fingerprint(canonical_target=session_id,
        payload={'operation': operation, 'payload': json.loads(encoded)})

    def write(conn):
        _epoch(conn, epoch)
        row = _worker_assignment(conn, execution_id, session_id, generation)
        if row['owner_epoch'] != epoch:
            raise RuntimeStoreError('stale_epoch')
        old = conn.execute('SELECT * FROM worker_receipts WHERE execution_id=? AND sequence=?',
                           (execution_id, sequence)).fetchone()
        if old is not None:
            if old['payload_digest'] != digest:
                raise RuntimeStoreError('admission_conflict')
            return json.loads(old['result_json'])
        if row['status'] not in ('registered', 'running'):
            raise RuntimeStoreError('stale_generation')
        if sequence != row['last_sequence'] + 1:
            raise RuntimeStoreError('invalid_params')
        result = handlers[operation](db, conn, session_id, json.loads(encoded))
        conn.execute('INSERT INTO worker_receipts(execution_id,sequence,payload_digest,result_json) VALUES(?,?,?,?)',
                     (execution_id, sequence, digest, json.dumps(result, ensure_ascii=True, allow_nan=False)))
        conn.execute("UPDATE worker_executions SET last_sequence=?,status='running' WHERE execution_id=?",
                     (sequence, execution_id))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (session_id,))
        return result
    return db._execute_write(write, patience_s=db._TRANSCRIPT_WRITE_PATIENCE_S)


async def worker_delegation_request(connection, ref, params):
    """worker.persist delegation-family hook; all authentication stays mandatory."""
    import asyncio
    from gateway.session_worker import _SCOPE, _claim, _verify
    if set(params) != _SCOPE | {'epoch', 'sequence', 'operation', 'payload'}:
        raise RuntimeStoreError('invalid_params')
    claim = _claim(connection, ref, params)
    authority = connection.authority
    authority._require_admission_open()
    if 'worker:adopt' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    _verify(connection, ref, params, claim)

    def persist_and_publish():
        result = persist_worker_delegation(authority.db,
            **{key: params[key] for key in ('epoch', 'execution_id', 'session_id', 'generation',
                                           'sequence', 'operation', 'payload')},
            worker_pid=params['pid'], worker_birth=params['birth'])
        if params['operation'] == 'delegation.complete':
            with authority.db._read_ctx() as conn:
                row = _owned(conn, ref.session_id, params['execution_id'], params['payload']['event']['delegation_id'])
                event = json.loads(row['event_json']) if row['delivery_state'] == 'pending' else None
            if event is not None:
                from tools.process_registry import process_registry
                # Queueing after commit can repeat after a lost ACK. Existing
                # token-guarded claims arbitrate delivery; never ACK at publication.
                process_registry.completion_queue.put(event | {'restored': True})
        return result
    return await asyncio.to_thread(persist_and_publish)
