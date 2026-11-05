"""The gateway owner's worker transactions and reads on the store: the claim check, the admission worker reservation
and its loss, the delegated child reservation, the delegation-ledger operations of the receipt stream, and the
execution reads.

Process identity is verified by the caller before any of these runs; the verified values arrive as parameters
(``pid``, ``birth``, ``owner_started_at``) and a failed probe arrives as ``probe_refused``.
"""
from functools import partial
import hmac
import json
import math
import secrets
import time

from persistence.authority.coordination import (
    DELEGATION_COMPLETION_STATUSES, DELEGATION_EVENT_KEYS,
    DELEGATION_MAX_DELIVERY_ATTEMPTS, DELEGATION_ORIGIN_SOURCES, DELEGATION_ROUTE_KEYS, DELEGATION_TASK_KEYS,
)
from persistence.authority.vocabulary import admission_fingerprint
from persistence.core.errors import AuthorityRefusal
from persistence.workers.vocabulary import CHILD_SEPARATOR
from session_rules.legacy_local import same_owner

from ..authority._runtime import _admission, _epoch, _secret_digest, _session
from ._receipt import apply_worker_receipt
from ._runtime_workers import compact_terminal_receipts, descendant_range


def verify_claim(holder, params, claim):
    """The claim check of a request already addressed to ``params['session_id']`` (the caller resolves a publication
    replay's target with ``worker_retry_target`` first)."""
    target = params['session_id']
    with holder._read_ctx() as conn:
        row = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?',
                           (params['execution_id'],)).fetchone()
        if row is None:
            raise AuthorityRefusal('not_found')
        if row['session_id'] != target:
            raise AuthorityRefusal('permission_denied')
        if type(params['generation']) is not int or row['generation'] != params['generation']:
            raise AuthorityRefusal('stale_generation')
        if not hmac.compare_digest(row['adoption_digest'], _secret_digest(claim)):
            raise AuthorityRefusal('permission_denied')
    return row


def reserve_admission_worker(holder, *, epoch, admission_id, principal_id, profile_id, pid, birth, secret, rule,
                             worker_subject=None):
    """Return a private bootstrap scope; never accept caller-selected kind/target.

    Only the owner process may reserve its live direct exec child. The admission
    supplies physical session and generation; its durable principal must match
    the producer under the owner rule *rule*. Do not publish the returned secret on any client event stream.
    The adoption claim binds *worker_subject*, the identity the worker will authenticate as, when given.
    """
    def write(conn):
        _epoch(conn, epoch)
        admission = _admission(conn, admission_id)
        if admission['status'] != 'started' or admission['owner_epoch'] != epoch:
            raise AuthorityRefusal('producer_not_started')
        if not same_owner(admission['principal_id'], principal_id, rule):
            raise AuthorityRefusal('permission_denied')
        owner, generation = admission['target_session_id'], admission['generation']
        if _session(conn, owner)['runtime_generation'] != generation:
            raise AuthorityRefusal('stale_generation')
        from ..authority._local import local_physical_target
        sid = local_physical_target(conn, owner)
        execution_id = 'admission-worker:' + admission_id
        if conn.execute('SELECT 1 FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone():
            raise AuthorityRefusal('admission_conflict')
        if conn.execute("SELECT 1 FROM worker_executions WHERE session_id=? AND status!='terminal'", (sid,)).fetchone():
            raise AuthorityRefusal('stale_generation')
        # The private worker fence follows its physical assignment; admission/FIFO
        # generation and identity remain on the logical owner.
        conn.execute('UPDATE sessions SET runtime_generation=? WHERE id=?', (generation, sid))
        if sid != owner:
            conn.execute("UPDATE session_admissions SET lineage_json=json_insert(lineage_json,'$[#]',?) "
                         "WHERE admission_id=?", (sid, admission_id))
        scope = dict(profile_id=profile_id, session_id=sid, execution_id=execution_id,
                     generation=generation, pid=pid, birth=birth, secret=secret)
        claim = admission_fingerprint(canonical_target=sid, payload=scope | {'principal': worker_subject or principal_id})
        conn.execute("INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
                     "VALUES(?,?,'compute',?,?,'registered',?)",
                     (execution_id, sid, epoch, generation, _secret_digest(claim)))
        return scope | {'epoch': epoch}

    return holder._execute_write(write)


def lose_admission_worker(holder, row, scope):
    """Revoke a private assignment and pause its FIFO; no inference replay.

    This also fences a result already in the pipe at loss/shutdown. Retention
    requires a still-started admission even when the generation has not changed.
    """
    def write(conn):
        _epoch(conn, scope['epoch'])
        current = _admission(conn, row['admission_id'])
        if (current['status'] != 'started' or current['owner_epoch'] != scope['epoch']
                or current['generation'] != scope['generation']):
            raise AuthorityRefusal('stale_generation')
        worker = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?',
                              (scope['execution_id'],)).fetchone()
        if worker is None or worker['generation'] != scope['generation'] or worker['owner_epoch'] != scope['epoch']:
            raise AuthorityRefusal('stale_generation')
        conn.execute("UPDATE worker_executions SET status='terminal' WHERE execution_id=?", (scope['execution_id'],))
        compact_terminal_receipts(conn, scope['execution_id'])
        conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (row['admission_id'],))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (current['target_session_id'],))
    holder._execute_write(write)


# The ``child.reserve`` handler: a new subagent session and running child execution under this worker, within the
# depth and live-children limits (a refusal is returned, not raised).
def reserve_child_execution(holder, conn, session_id, payload, *, epoch, execution_id, identity, principal, limits, refused):
    from session_rules.session_ids import new_session_id
    secret = payload.get('secret') if isinstance(payload, dict) else None
    if set(payload) != {'secret'} or not isinstance(secret, str) or not 16 <= len(secret) <= 256:
        raise AuthorityRefusal('invalid_params')
    max_children, max_depth = limits
    prefix = execution_id + CHILD_SEPARATOR
    live = conn.execute("SELECT COUNT(*) FROM worker_executions WHERE execution_id>=? AND execution_id<? "
                        "AND instr(substr(execution_id,?),'/')=0 AND status!='terminal'",
                        (*descendant_range(execution_id), len(prefix) + 1)).fetchone()[0]
    if refused is None and prefix.count(CHILD_SEPARATOR) > max_depth:
        refused = 'delegation_depth_exceeded'
    if refused is None and live >= max_children:
        refused = 'delegation_capacity'
    if refused is not None:
        return {'refused': refused}
    parent = conn.execute('SELECT profile_name FROM sessions WHERE id=?', (session_id,)).fetchone()
    child_sid, child_eid = new_session_id(), prefix + secrets.token_hex(8)
    from ..conversation._store.worker_lifecycle import insert_session_row_in_transaction
    insert_session_row_in_transaction(holder, conn, child_sid, 'subagent', model_config={'_delegate_from': session_id},
                                      parent_session_id=session_id, profile_name=parent['profile_name'])
    scope = dict(identity, session_id=child_sid, execution_id=child_eid, generation=0, secret=secret)
    claim = admission_fingerprint(canonical_target=child_sid, payload=scope | {'principal': principal})
    conn.execute("INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
                 "VALUES(?,?,'child',?,0,'running',?)", (child_eid, child_sid, epoch, _secret_digest(claim)))
    return {'child': {'session_id': child_sid, 'execution_id': child_eid, 'generation': 0}}


# ``child.reserve`` through the worker's receipt stream.
def reserve_child(holder, *, epoch, execution_id, session_id, generation, sequence, payload, identity, principal,
                  limits, refused):
    handler = partial(reserve_child_execution, epoch=epoch, execution_id=execution_id, identity=identity,
                      principal=principal, limits=limits, refused=refused)
    return apply_worker_receipt(holder, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                generation=generation, sequence=sequence, operation='child.reserve',
                                payload=payload, handlers={'child.reserve': handler})


def execution_session(holder, execution_id):
    """The session this worker execution last wrote to, as the owner's ledger records it."""
    if not execution_id:
        return None
    with holder._read_ctx() as conn:
        row = conn.execute('SELECT session_id FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
    return row[0] if row else None


# Whether any worker execution is still live.
def has_live_worker(holder):
    return bool(holder._read_all("SELECT 1 FROM worker_executions WHERE status != 'terminal' LIMIT 1"))


# The route keys a dispatch carries and a completion event must echo, mapped to their ledger columns.
_ROUTE = DELEGATION_ROUTE_KEYS


# The payload must be a dict with exactly these keys.
def _shape(payload, keys):
    if not isinstance(payload, dict) or set(payload) != set(keys):
        raise AuthorityRefusal('invalid_params')


# A non-empty string of at most 512 characters.
def _text(value):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise AuthorityRefusal('invalid_params')


# A finite, non-negative number.
def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise AuthorityRefusal('invalid_params')


# The delegation row, refused unless this worker execution on this session owns it.
def _owned(conn, session_id, execution_id, delegation_id):
    _text(delegation_id)
    row = conn.execute('SELECT * FROM async_delegations WHERE delegation_id=?', (delegation_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal('not_found')
    if row['parent_session_id'] != session_id or row['owner_execution_id'] != execution_id:
        raise AuthorityRefusal('permission_denied')
    return row


# ``delegation.dispatch``: record a running delegation of this worker, its route checked against the session.
def _dispatch(holder, conn, session_id, payload, *, execution_id, worker_pid, owner_started_at, probe_refused):
    _shape(payload, {'delegation_id', 'task', 'dispatched_at', *_ROUTE})
    _text(payload['delegation_id'])
    _number(payload['dispatched_at'])
    if not isinstance(payload['task'], dict) or set(payload['task']) - DELEGATION_TASK_KEYS:
        raise AuthorityRefusal('invalid_params')
    session = conn.execute('SELECT source,session_key FROM sessions WHERE id=?', (session_id,)).fetchone()
    # Only compute registrations are currently enabled. Native/child origins need
    # an owner-reserved source envelope, not worker-selected transport identities.
    if session is None or session['source'] not in DELEGATION_ORIGIN_SOURCES:
        raise AuthorityRefusal('unsupported_producer')
    if (payload['parent_session_id'] != session_id
            or payload['session_key'] != (session['session_key'] or '')
            or payload['origin_ui_session_id'] not in ('', session_id)
            or payload['origin_session_id'] not in ('', session_id)):
        raise AuthorityRefusal('permission_denied')
    old = conn.execute('SELECT 1 FROM async_delegations WHERE delegation_id=?', (payload['delegation_id'],)).fetchone()
    if old is not None:
        _owned(conn, session_id, execution_id, payload['delegation_id'])
        # Exact network retries are resolved by worker_receipts before this handler.
        raise AuthorityRefusal('admission_conflict')
    # The caller probed the verified worker process before the call.
    if probe_refused is not None:
        raise AuthorityRefusal(probe_refused)
    now = time.time()
    conn.execute('''INSERT INTO async_delegations
        (delegation_id,origin_session,origin_ui_session_id,parent_session_id,state,dispatched_at,
         updated_at,owner_pid,owner_started_at,task_json,origin_session_id,owner_execution_id)
        VALUES(?,?,?,?,'running',?,?,?,?,?,?,?)''',
        (payload['delegation_id'], payload['session_key'], payload['origin_ui_session_id'], session_id,
         payload['dispatched_at'], now, worker_pid, owner_started_at, json.dumps(payload['task']),
         payload['origin_session_id'], execution_id))
    return {'value': None}


# ``delegation.complete``: the finished state, event and result of a running delegation.
def _complete(holder, conn, session_id, payload, *, execution_id):
    _shape(payload, {'event', 'result'})
    event = payload['event']
    if (not isinstance(event, dict) or set(event) - DELEGATION_EVENT_KEYS
            or not isinstance(payload['result'], dict) or event.get('type') != 'async_delegation'
            or event.get('status') not in DELEGATION_COMPLETION_STATUSES):
        raise AuthorityRefusal('invalid_params')
    row = _owned(conn, session_id, execution_id, event.get('delegation_id'))
    if any(event.get(key, '') != (row[column] or '') for key, column in _ROUTE.items()):
        raise AuthorityRefusal('permission_denied')
    _number(event.get('completed_at'))
    if row['state'] != 'running':
        raise AuthorityRefusal('admission_conflict')
    conn.execute('''UPDATE async_delegations SET state=?,completed_at=?,updated_at=?,event_json=?,result_json=?
        WHERE delegation_id=?''', (event['status'], event['completed_at'], time.time(),
        json.dumps(event), json.dumps(payload['result']), event['delegation_id']))
    return {'value': None}


# ``delegation.child``: merge one finished child into a running delegation's partial result.
def _child(holder, conn, session_id, payload, *, execution_id):
    _shape(payload, {'delegation_id', 'entry'})
    row = _owned(conn, session_id, execution_id, payload['delegation_id'])
    entry = payload['entry']
    if not isinstance(entry, dict) or type(entry.get('task_index')) is not int or entry['task_index'] < 0:
        raise AuthorityRefusal('invalid_params')
    if row['state'] != 'running':
        return {'value': None}
    previous = json.loads(row['result_json'] or '{}')
    results = [item for item in previous.get('results', []) if item.get('task_index') != entry['task_index']]
    results.append(entry)
    conn.execute('UPDATE async_delegations SET result_json=?,updated_at=? WHERE delegation_id=?',
        (json.dumps({'results': results, 'partial': True}), time.time(), payload['delegation_id']))
    return {'value': None}


# ``delegation.read``: the delegation's public fields and its result.
def _read(holder, conn, session_id, payload, *, execution_id):
    _shape(payload, {'delegation_id'})
    row = _owned(conn, session_id, execution_id, payload['delegation_id'])
    return {'value': {key: row[key] for key in ('delegation_id', 'origin_session', 'state', 'dispatched_at',
        'completed_at', 'delivery_state', 'delivery_attempts', 'origin_session_id')} |
        {'result': json.loads(row['result_json']) if row['result_json'] else None}}


# ``delegation.claim``: claim a finished delegation's pending delivery; each claim spends one attempt.
def _claim(holder, conn, session_id, payload, *, execution_id):
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


# The ``delegation.ack``, ``drop``, ``defer`` and ``release`` updates of a claimed delivery.
_DELIVERY_UPDATES = {
    'ack': "delivery_state='delivered',delivered_at=:now",
    'drop': "delivery_state='dropped'",
    'defer': 'delivery_attempts=MAX(0,delivery_attempts-1)',
    'release': "delivery_state=CASE WHEN delivery_attempts>=:cap THEN 'dropped' ELSE delivery_state END",
}


# One delivery update that also clears the claim, only while this claim still holds the pending delivery.
def _delivery(holder, conn, session_id, payload, *, execution_id, action):
    _shape(payload, {'delegation_id', 'claim_id'})
    _owned(conn, session_id, execution_id, payload['delegation_id'])
    _text(payload['claim_id'])
    # Only constant, closed statement fragments enter this query.
    changed = conn.execute('UPDATE async_delegations SET ' + _DELIVERY_UPDATES[action] +
        ",updated_at=:now,delivery_claim=NULL,delivery_claimed_at=NULL WHERE delegation_id=:delegation_id "
        "AND delivery_state='pending' AND delivery_claim=:claim_id",
        payload | {'now': time.time(), 'cap': DELEGATION_MAX_DELIVERY_ATTEMPTS}).rowcount
    return {'value': changed == 1}


# ``delegation.unscheduled``: delete a running delegation whose submit was never scheduled.
def _unscheduled(holder, conn, session_id, payload, *, execution_id):
    _shape(payload, {'delegation_id'})
    row = _owned(conn, session_id, execution_id, payload['delegation_id'])
    if row['state'] != 'running' or row['result_json'] is not None:
        raise AuthorityRefusal('admission_conflict')
    conn.execute('DELETE FROM async_delegations WHERE delegation_id=?', (payload['delegation_id'],))
    return {'value': None}


# The ``delegation.*`` handler table, bound to one verified worker process.
def worker_delegation_handlers(execution_id, worker_pid, worker_birth, owner_started_at, probe_refused):
    """Register these handlers only after gateway worker identity verification."""
    _text(execution_id)
    if type(worker_pid) is not int or worker_pid <= 0:
        raise AuthorityRefusal('invalid_params')
    _number(worker_birth)
    handlers = {name: partial(fn, execution_id=execution_id) for name, fn in {
        'complete': _complete, 'child': _child, 'read': _read, 'claim': _claim,
        'unscheduled': _unscheduled}.items()}
    handlers['dispatch'] = partial(_dispatch, execution_id=execution_id, worker_pid=worker_pid,
                                   owner_started_at=owner_started_at, probe_refused=probe_refused)
    handlers.update({name: partial(_delivery, execution_id=execution_id, action=name) for name in _DELIVERY_UPDATES})
    return {'delegation.' + name: handler for name, handler in handlers.items()}


def persist_worker_delegation(holder, *, epoch, execution_id, session_id, generation,
                              sequence, operation, payload, worker_pid, worker_birth, owner_started_at=None,
                              probe_refused=None):
    """Closed ledger sibling of mutate_worker_execution, sharing its receipt stream.

    Gateway calls this ONLY after its ordinary worker claim verification. The
    verified process identity is deliberately separate from the operation payload.
    """
    handlers = worker_delegation_handlers(execution_id, worker_pid, worker_birth, owner_started_at, probe_refused)
    return apply_worker_receipt(holder, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                generation=generation, sequence=sequence, operation=operation,
                                payload=payload, handlers=handlers)


# The event of the delegation's pending delivery, else None.
def pending_delegation_event(holder, *, session_id, execution_id, delegation_id):
    with holder._read_ctx() as conn:
        row = _owned(conn, session_id, execution_id, delegation_id)
        pending = row['delivery_state'] == 'pending' and row['event_json'] is not None
        event = json.loads(row['event_json']) if pending else None
    return event
