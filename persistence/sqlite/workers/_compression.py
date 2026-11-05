"""Closed compression operations on the worker receipt's connection, and the receipt-aware assignment reads the owner
makes before a worker request.

No handler opens a connection or commits. The read handlers run the conversation family's read bodies over
:class:`_ReceiptView`, which answers their reads on the receipt's connection; ``compression.lock.release`` has no
producer and no handler.
"""
from typing import Any, Optional
import json
from contextlib import nullcontext
from functools import partial
import math
import time

from persistence.authority.vocabulary import admission_fingerprint
from persistence.core.errors import AuthorityRefusal, CooldownRestoreRefused
from persistence.workers.vocabulary import HANDOFF_MESSAGE_FIELDS, SIDECAR_KEYS, row_annotations
from session_rules.automation import GoalState, HeartbeatState, LoopState

from ..authority._runtime import _text, _worker_assignment
from ..authority._runtime_guards import _ENDED_ROW_SQL, _ended_by_compression
from ._context import worker_context

_LOCK_ROW_SQL = "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?"


def _fields(payload, required, optional=()):
    if not isinstance(payload, dict) or set(payload) - set(required) - set(optional) or set(required) - set(payload):
        raise AuthorityRefusal('invalid_params')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise AuthorityRefusal('invalid_params')
    return value


def worker_compression_lock(holder, conn, sid, payload, *, action):
    if action == 'holder':
        _fields(payload, ())
        row = conn.execute('SELECT holder FROM compression_locks WHERE session_id=? AND expires_at>?',
                           (sid, time.time())).fetchone()
        return {'value': row[0] if row else None}
    _fields(payload, ('holder',), ('ttl_seconds',))
    lease_holder = _text(payload['holder'])
    now = time.time()
    ttl = _number(payload.get('ttl_seconds', 300.0))
    if not 0.1 <= ttl <= 3600:
        raise AuthorityRefusal('invalid_params')
    if action == 'acquire':
        from ..authority.coordination import _claim_lease_row
        from ..conversation._store.core import _compression_lock_holder_process_is_dead
        value = _claim_lease_row(conn, 'compression_locks', 'session_id', sid, lease_holder, now, now + ttl,
                                lambda h, e: e < now or _compression_lock_holder_process_is_dead(h))[0]
    else:
        value = conn.execute('UPDATE compression_locks SET expires_at=? WHERE session_id=? AND holder=?',
                             (now + ttl, sid, lease_holder)).rowcount > 0
    return {'value': value}


def worker_cooldown_record(holder, conn, sid, payload):
    _fields(payload, ('cooldown_until', 'error'))
    deadline = _number(payload['cooldown_until'])
    error = payload['error']
    if error is not None and not isinstance(error, str):
        raise AuthorityRefusal('invalid_params')
    conn.execute('UPDATE sessions SET compression_failure_cooldown_until = CASE '
                 'WHEN compression_failure_cooldown_until IS NOT NULL AND compression_failure_cooldown_until > ? '
                 'THEN compression_failure_cooldown_until ELSE ? END, compression_failure_error=? WHERE id=?',
                 (deadline, deadline, error, sid))
    return {'value': None}


def worker_cooldown_restore(holder, conn, sid, payload):
    _fields(payload, ('snapshot',))
    snapshot = payload['snapshot']
    _fields(snapshot, ('session_exists', 'cooldown_until', 'error'))
    if type(snapshot['session_exists']) is not bool:
        raise AuthorityRefusal('invalid_params')
    if not snapshot['session_exists']:
        raise CooldownRestoreRefused(sid, 'cannot restore absent compression cooldown row: session now exists')
    deadline, error = snapshot['cooldown_until'], snapshot['error']
    if deadline is not None:
        _number(deadline)
    if error is not None and not isinstance(error, str):
        raise AuthorityRefusal('invalid_params')
    conn.execute('UPDATE sessions SET compression_failure_cooldown_until=?, compression_failure_error=? WHERE id=?',
                 (deadline, error, sid))
    actual = conn.execute('SELECT compression_failure_cooldown_until,compression_failure_error FROM sessions WHERE id=?',
                          (sid,)).fetchone()
    if actual is None or tuple(actual) != (deadline, error):
        raise CooldownRestoreRefused(sid, 'compression cooldown rollback verification failed')
    return {'value': None}


def worker_cooldown_clear(holder, conn, sid, payload):
    _fields(payload, ())
    return worker_cooldown_restore(holder, conn, sid, {'snapshot': {
        'session_exists': True, 'cooldown_until': None, 'error': None}})


def worker_compression_counter(holder, conn, sid, payload, *, column):
    _fields(payload, ('value',))
    value = payload['value']
    if value is not None:
        _number(value)
    if column != 'compression_recovery_deadline' and (type(value) is not int or value < 0):
        raise AuthorityRefusal('invalid_params')
    conn.execute(f'UPDATE sessions SET {column}=? WHERE id=?', (value, sid))
    return {'value': None}


class _ReceiptView:
    """Reuse production read algorithms on the receipt connection, never a second handle.

    The conversation family's read bodies take a store holder; this view stands in for it and answers their reads
    on the receipt's connection. Only read primitives are provided; it cannot start or commit a write, and it never
    drains the accounting queue.
    """

    def __init__(self, conn):
        self.conn = conn

    def _read_ctx(self):
        return nullcontext(self.conn)

    def _read_one(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()

    def _read_all(self, sql, params=()):
        return self.conn.execute(sql, params).fetchall()

    def flush_token_counts(self):
        return True

    def flush_for_read(self):
        return True


def _view_session(holder, conn, sid):
    return worker_context(holder, conn, sid, {})['session']


def _authorize(view, assigned, target):
    from ..conversation._store.compression import get_compression_chain
    from ..conversation._store.sessions import _session_lineage_root_to_tip
    _text(target)
    # Attribution ancestry is readable; unrelated siblings and foreign trees are not.
    allowed = _session_lineage_root_to_tip(view, assigned)
    allowed += get_compression_chain(view, assigned)
    if target not in allowed:
        raise AuthorityRefusal('permission_denied')


def worker_lineage_context(holder, conn, sid, payload):
    _fields(payload, ('target',))
    view = _ReceiptView(conn)
    target = payload['target']
    _authorize(view, sid, target)
    return {'session': _view_session(holder, conn, target)}


def worker_lineage(holder, conn, sid, payload):
    _fields(payload, ('target',))
    from ..conversation._store.compression import (
        get_compression_chain, get_compression_lineage, get_compression_tip, get_tool_rotation_root,
    )
    from ..conversation._store.messages import get_conversation_root, resolve_resume_session_id
    from ..conversation._store.sessions import _session_lineage_root_to_tip, declared_scope_identity
    view = _ReceiptView(conn)
    target = payload['target']
    _authorize(view, sid, target)
    return {'readable_ids': _session_lineage_root_to_tip(view, sid) + get_compression_chain(view, sid),
            'lineage': get_compression_lineage(view, target),
            'root': get_conversation_root(view, target),
            'tip': get_compression_tip(view, target),
            'resume': resolve_resume_session_id(view, target),
            'identity': declared_scope_identity(view, target),
            'tool_root': get_tool_rotation_root(view, target)}


def worker_boundary(holder, conn, sid, payload):
    _fields(payload, ('session_key', 'source'))
    from ..conversation._store.messages import latest_conversation_boundary
    view = _ReceiptView(conn)
    row = _view_session(holder, conn, sid)
    if (row['session_key'], row['source']) != (payload['session_key'], payload['source']):
        raise AuthorityRefusal('permission_denied')
    return {'value': latest_conversation_boundary(view, payload['session_key'], payload['source'])}


def worker_history(holder, conn, sid, payload):
    _fields(payload, ('target', 'include_ancestors', 'include_inactive', 'repair_alternation',
                      'include_row_ids', 'include_compacted'))
    if any(type(v) is not bool for k, v in payload.items() if k != 'target'):
        raise AuthorityRefusal('invalid_params')
    from ..conversation._store.core import _CONVERSATION_ROW_COLUMNS
    from ..conversation._store.messages import (
        _active_clause, _dedupe_display_generations, _resume_lineage_ids, _rows_to_conversation, _with_removed_identities,
    )
    view = _ReceiptView(conn)
    target = payload['target']
    _authorize(view, sid, target)
    ids = _resume_lineage_ids(view, target) if payload['include_ancestors'] else [target]
    active = _active_clause(payload['include_inactive'], payload['include_compacted'])
    display = payload['include_compacted'] and not payload['include_inactive']
    rows = conn.execute(f'SELECT {"session_id, compacted, display_order, " if display else ""}{_CONVERSATION_ROW_COLUMNS} FROM messages '
                        f'WHERE session_id IN ({",".join("?" for _ in ids)}){active} ORDER BY id', ids).fetchall()
    if display:  # the owner's display reads' rule: a message whose newest copy was removed stays hidden
        rows = _with_removed_identities(view, rows, ids)
    if payload['include_compacted']:
        rows = _dedupe_display_generations(holder, rows, drop_removed=display)
    return {'messages': _rows_to_conversation(holder, rows, session_id=target,
        include_ancestors=payload['include_ancestors'], repair_alternation=payload['repair_alternation'],
        include_row_ids=payload['include_row_ids'])}


def archive_on_connection(holder, conn, session_id: str, compacted_messages: list[dict[str, Any]], model_config_patch: Optional[dict[str, Any]]=None, watermark: Optional[int]=None, lock_holder: Optional[str]=None, tail_count: int=0, carried_messages: Optional[list[dict[str, Any]]]=None, covered_ids: Optional[list[int]]=None, unresolved_held: Optional[list[dict[str, Any]]]=None, held_for_cas: Optional[list[dict[str, Any]]]=None):
    # The owner's in-place compaction body, on this receipt's connection: one implementation, so
    # main's coverage/merge/timestamp/display-order fixes reach worker and mutation verbs too.
    from ..conversation._store.messages import _archive_and_compact_on_conn
    return _archive_and_compact_on_conn(holder, conn, session_id, compacted_messages,
        model_config_patch=model_config_patch, watermark=watermark, lock_holder=lock_holder,
        tail_count=tail_count, carried_messages=carried_messages, covered_ids=covered_ids,
        unresolved_held=unresolved_held, held_for_cas=held_for_cas)


def publish_on_connection(holder, conn, *, parent_session_id: str, child_session_id: str, source: str, messages: list[dict[str, Any]], model: str | None=None, model_config: dict[str, Any] | None=None, system_prompt: str | None=None, cwd: str | None=None, profile_name: str | None=None, compression_lock_holder: str | None=None, require_compression_lease: bool=True, require_lease_refresh: bool=False, lease_ttl_seconds: float=300.0, watermark: Optional[int]=None, watermark_ceiling: Optional[int]=None):
    # The owner's rotation body (lease, closure, child row, timestamps, tail clone, local target,
    # auto-archive lift) on this receipt's connection, so the two paths cannot drift.
    from ..conversation._store.compression import _publish_compression_child_on_conn
    _publish_compression_child_on_conn(holder, conn, parent_session_id=parent_session_id,
        child_session_id=child_session_id, source=source, messages=messages, model=model,
        model_config=model_config, system_prompt=system_prompt, cwd=cwd, profile_name=profile_name,
        compression_lock_holder=compression_lock_holder, require_compression_lease=require_compression_lease,
        require_lease_refresh=require_lease_refresh, lease_ttl_seconds=lease_ttl_seconds,
        watermark=watermark, watermark_ceiling=watermark_ceiling)


def worker_watermark(holder, conn, sid, payload):
    _fields(payload, ())
    return {'value': conn.execute('SELECT COALESCE(MAX(id),0) FROM messages WHERE session_id=? AND active=1',
                                  (sid,)).fetchone()[0]}


def _handoff_messages(conn, sid, messages):
    if not isinstance(messages, list) or len(messages) > 1000:
        raise AuthorityRefusal('invalid_params')
    for msg in messages:
        if not isinstance(msg, dict) or set(msg) - HANDOFF_MESSAGE_FIELDS or msg.get('role') not in ('user', 'assistant', 'system', 'tool'):
            raise AuthorityRefusal('invalid_params')
        if '_row_id' in msg and not conn.execute('SELECT 1 FROM messages WHERE session_id=? AND id=?',
                                                 (sid, msg['_row_id'])).fetchone():
            raise AuthorityRefusal('permission_denied')


def _watermarks(payload):
    for field in ('watermark', 'watermark_ceiling'):
        value = payload.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise AuthorityRefusal('invalid_params')


def worker_archive(holder, conn, sid, payload):
    _fields(payload, ('messages', 'model_config_patch', 'watermark', 'lock_holder', 'tail_count'),
            optional=('carried_messages', 'covered_ids', 'unresolved_held', 'held_for_cas'))
    _handoff_messages(conn, sid, payload['messages'])
    carried = payload.get('carried_messages') or []
    if not isinstance(carried, list) or any(not isinstance(m, dict) for m in carried):
        raise AuthorityRefusal('invalid_params')
    covered = payload.get('covered_ids')
    if covered is not None and (not isinstance(covered, list)
                                or any(type(i) is not int or i <= 0 for i in covered)):
        raise AuthorityRefusal('invalid_params')
    unresolved = payload.get('unresolved_held')
    if unresolved is not None and (not isinstance(unresolved, list)
                                   or any(not isinstance(m, dict) for m in unresolved)):
        raise AuthorityRefusal('invalid_params')
    # The lease-less CAS fence (#124102): dicts projected to the closed handoff fields plus the
    # persistence marker the owner's compare reads; validated like the other held-row payloads.
    cas = payload.get('held_for_cas')
    if cas is not None and (not isinstance(cas, list)
                            or any(not isinstance(m, dict) for m in cas)):
        raise AuthorityRefusal('invalid_params')
    _watermarks(payload)
    # Micro-compaction and proactive prune archive in place without a lease, like the owner path.
    if payload['lock_holder'] is not None:
        _text(payload['lock_holder'])
    patch = payload['model_config_patch']
    if patch is not None and (not isinstance(patch, dict) or set(patch) - SIDECAR_KEYS):
        raise AuthorityRefusal('invalid_params')
    if type(payload['tail_count']) is not int or payload['tail_count'] < 0:
        raise AuthorityRefusal('invalid_params')
    value = archive_on_connection(holder, conn, sid, payload['messages'], model_config_patch=patch,
        watermark=payload['watermark'], lock_holder=payload['lock_holder'], tail_count=payload['tail_count'],
        carried_messages=carried, covered_ids=covered, unresolved_held=unresolved, held_for_cas=cas)
    return {'value': value, 'annotations': row_annotations(payload['messages'])}


def _migrate_worker_automation(holder, conn, parent, child):
    for family, state_type in (('goal', GoalState), ('heartbeat', HeartbeatState), ('loop', LoopState)):
        old_key, new_key = f'{family}:{parent}', f'{family}:{child}'
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (old_key,)).fetchone()
        if row is None:
            continue
        state = state_type.from_json(row[0])
        if state.status == 'cleared':
            continue
        prior = conn.execute('SELECT value FROM state_meta WHERE key=?', (new_key,)).fetchone()
        if prior is not None and (family != 'heartbeat' or state_type.from_json(prior[0]).status != 'cleared'):
            continue
        holder.set_meta(new_key, state.to_json(), cursor=conn)
        state.status = 'cleared'
        holder.set_meta(old_key, state.to_json(), cursor=conn)


def worker_publish(holder, conn, sid, payload):
    _fields(payload, ('child_session_id', 'source', 'messages', 'model', 'model_config', 'system_prompt',
                      'cwd', 'profile_name', 'compression_lock_holder', 'require_compression_lease',
                      'require_lease_refresh', 'lease_ttl_seconds', 'watermark', 'watermark_ceiling'))
    child = _text(payload['child_session_id'])
    _text(payload['compression_lock_holder'])
    _watermarks(payload)
    if payload['require_compression_lease'] is not True or type(payload['require_lease_refresh']) is not bool:
        raise AuthorityRefusal('invalid_params')
    if not 0.1 <= _number(payload['lease_ttl_seconds']) <= 3600:
        raise AuthorityRefusal('invalid_params')
    _handoff_messages(conn, sid, payload['messages'])
    parent = _view_session(holder, conn, sid)
    for key in ('source', 'cwd', 'profile_name'):
        if payload[key] is not None and payload[key] != parent[key]:
            raise AuthorityRefusal('permission_denied')
    for key in ('model', 'system_prompt'):
        if payload[key] is not None and not isinstance(payload[key], str):
            raise AuthorityRefusal('invalid_params')
    cfg = payload['model_config']
    if cfg is not None and not isinstance(cfg, dict):
        raise AuthorityRefusal('invalid_params')
    cfg = dict(cfg or {})
    from ..conversation._store.sessions import _parse_model_config
    prior = _parse_model_config(parent['model_config'])
    for key in ('_branched_from', '_delegate_from', '_reset_from'):
        if key in cfg and cfg[key] != prior.get(key):
            raise AuthorityRefusal('permission_denied')
        if key in prior:
            cfg[key] = prior[key]
    if conn.execute('SELECT 1 FROM sessions WHERE id=?', (child,)).fetchone():
        raise AuthorityRefusal('permission_denied')
    worker = conn.execute("SELECT * FROM worker_executions WHERE session_id=? AND status IN ('registered','running')", (sid,)).fetchone()
    if worker is None:
        raise AuthorityRefusal('stale_generation')
    publish_on_connection(holder, conn, parent_session_id=sid, **dict(payload, model_config=cfg))
    _migrate_worker_automation(holder, conn, sid, child)
    conn.execute('UPDATE sessions SET runtime_generation=? WHERE id=?', (worker['generation'], child))
    conn.execute('UPDATE worker_executions SET session_id=? WHERE execution_id=?', (child, worker['execution_id']))
    # The FIFO/retry key is immutable; only physical lineage follows rotation.
    conn.execute("UPDATE session_admissions SET lineage_json=json_insert(lineage_json,'$[#]',?) "
                 "WHERE json_extract(lineage_json,'$[#-1]')=? AND generation=? AND owner_epoch=? "
                 "AND status IN ('started','unknown')",
                 (child, sid, worker['generation'], worker['owner_epoch']))
    return {'value': None, 'worker_assignment': {'parent': sid, 'session_id': child},
            'annotations': row_annotations(payload['messages'])}


def worker_receipt_assignment(conn, execution_id, session_id, generation, sequence, digest):
    """Only an identical already-committed publication can retry its old assignment."""
    receipt = conn.execute('SELECT payload_digest,result_json FROM worker_receipts WHERE execution_id=? AND sequence=?',
                           (execution_id, sequence)).fetchone()
    if receipt and receipt['payload_digest'] == digest:
        assignment = json.loads(receipt['result_json']).get('worker_assignment')
        if assignment and assignment['parent'] == session_id:
            current = conn.execute('SELECT session_id FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
            return _worker_assignment(conn, execution_id, current['session_id'], generation)
    return _worker_assignment(conn, execution_id, session_id, generation)


def worker_claim_origin(holder, params):
    """Keep the producer claim bound to its original identity after physical rotation."""
    with holder._read_ctx() as conn:
        row = conn.execute("SELECT result_json FROM worker_receipts WHERE execution_id=? "
                           "AND json_extract(result_json,'$.worker_assignment.parent') IS NOT NULL ORDER BY sequence LIMIT 1",
                           (params['execution_id'],)).fetchone()
    return json.loads(row[0])['worker_assignment']['parent'] if row else params['session_id']


def worker_retry_target(holder, params):
    """Transport verification accepts old targets only for exact publication replay."""
    if params.get('operation') != 'compression.publish':
        return params['session_id']
    digest = admission_fingerprint(canonical_target=params['session_id'],
        payload={'operation': params['operation'], 'payload': params['payload']})
    with holder._read_ctx() as conn:
        row = worker_receipt_assignment(conn, params['execution_id'], params['session_id'],
                                        params['generation'], params['sequence'], digest)
    return row['session_id']


def reopen_on_connection(holder, conn, session_id):
    if not _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()):
        return False
    from ..conversation._store.sessions import _NON_CONTINUATION_CHILD_FILTER_SQL
    child = conn.execute('\n                SELECT 1\n                FROM sessions\n                WHERE parent_session_id = ?\n                ' + _NON_CONTINUATION_CHILD_FILTER_SQL.format(alias='') + '\n                LIMIT 1\n                ', (session_id,) * 4).fetchone()
    if child is not None:
        return False
    now = time.time()
    lock_row = conn.execute(_LOCK_ROW_SQL, (session_id,)).fetchone()
    if lock_row is not None:
        expires_at = lock_row['expires_at']
        if expires_at is None or float(expires_at) >= now:
            return False
        deleted = conn.execute('DELETE FROM compression_locks WHERE session_id = ? AND holder = ? AND expires_at = ?', (session_id, lock_row['holder'], expires_at))
        if deleted.rowcount != 1:
            return False
    updated = conn.execute("UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ? AND ended_at IS NOT NULL AND end_reason = 'compression'", (session_id,))
    return updated.rowcount == 1


def worker_reopen(holder, conn, sid, payload):
    _fields(payload, ())
    return {'value': reopen_on_connection(holder, conn, sid)}


def worker_compression_cleanup(holder, conn, sid, payload):
    from ..conversation._store.compression import _session_turn_lease_key_on_conn
    _fields(payload, ('target', 'holder'))
    target = _text(payload['target'])
    lease_holder = _text(payload['holder'])
    _authorize(_ReceiptView(conn), sid, target)
    if _session_turn_lease_key_on_conn(holder, conn, sid) != _session_turn_lease_key_on_conn(holder, conn, target):
        raise AuthorityRefusal('permission_denied')
    conn.execute('DELETE FROM compression_locks WHERE session_id=? AND holder=?', (target, lease_holder))
    return {'value': None}


def worker_compression_append(holder, conn, sid, payload):
    _fields(payload, ('messages', 'compression_lock_holder', 'turn_lease_holder', 'turn_lease_ttl_seconds'))
    _handoff_messages(conn, sid, payload['messages'])
    for key in ('compression_lock_holder', 'turn_lease_holder'):
        if payload[key] is not None:
            _text(payload[key])
    if not 0.1 <= _number(payload['turn_lease_ttl_seconds']) <= 3600:
        raise AuthorityRefusal('invalid_params')
    from ..conversation._store.messages import _append_messages_in_transaction
    count = _append_messages_in_transaction(holder, conn, sid, **payload)
    return {'count': count, 'annotations': row_annotations(payload['messages'])}


def worker_turn_cleanup(holder, conn, sid, payload):
    from ..conversation._store.compression import _session_turn_lease_key_on_conn
    _fields(payload, ('target', 'holder'))
    target = _text(payload['target'])
    lease_holder = _text(payload['holder'])
    _authorize(_ReceiptView(conn), sid, target)
    key = _session_turn_lease_key_on_conn(holder, conn, sid)
    if key != _session_turn_lease_key_on_conn(holder, conn, target):
        raise AuthorityRefusal('permission_denied')
    conn.execute('DELETE FROM session_turn_leases WHERE conversation_id=? AND holder=?', (key, lease_holder))
    return {'released': True}


WORKER_COMPRESSION_HANDLERS = {
    'turn.cleanup': worker_turn_cleanup,
    'compression.append': worker_compression_append,
    'compression.reopen': worker_reopen,
    'compression.cleanup': worker_compression_cleanup,
    'compression.watermark': worker_watermark,
    'compression.archive': worker_archive,
    'compression.publish': worker_publish,
    'compression.context': worker_lineage_context,
    'compression.lineage': worker_lineage,
    'compression.boundary': worker_boundary,
    'compression.history': worker_history,
    **{'compression.lock.' + action: partial(worker_compression_lock, action=action)
       for action in ('acquire', 'renew', 'holder')},
    'compression.cooldown.record': worker_cooldown_record,
    'compression.cooldown.restore': worker_cooldown_restore,
    'compression.cooldown.clear': worker_cooldown_clear,
    **{'compression.' + name: partial(worker_compression_counter, column=column)
       for name, column in (('fallback', 'compression_fallback_streak'),
                            ('ineffective', 'compression_ineffective_count'),
                            ('recovery', 'compression_recovery_deadline'))},
}
