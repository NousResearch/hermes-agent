"""Closed compression operations on the worker receipt's connection.

No handler opens a connection or invokes a self-committing SessionDB method.
"""
from typing import Any, Dict, List, Optional
import json
from functools import partial
import math
import time

from hermes_state_runtime import RuntimeStoreError, _text
from hermes_state_compression import _claim_lease_row


def _fields(payload, required, optional=()):
    if not isinstance(payload, dict) or set(payload) - set(required) - set(optional) or set(required) - set(payload):
        raise RuntimeStoreError('invalid_params')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RuntimeStoreError('invalid_params')
    return value


def worker_compression_lock(db, conn, sid, payload, *, action):
    if action == 'holder':
        _fields(payload, ())
        row = conn.execute('SELECT holder FROM compression_locks WHERE session_id=? AND expires_at>?',
                           (sid, time.time())).fetchone()
        return {'value': row[0] if row else None}
    _fields(payload, ('holder',), () if action == 'release' else ('ttl_seconds',))
    holder = _text(payload['holder'])
    now = time.time()
    ttl = _number(payload.get('ttl_seconds', 300.0))
    if not 0.1 <= ttl <= 3600:
        raise RuntimeStoreError('invalid_params')
    if action == 'acquire':
        from hermes_state import _compression_lock_holder_process_is_dead
        value = _claim_lease_row(conn, 'compression_locks', 'session_id', sid, holder, now, now + ttl,
                                lambda h, e: e < now or _compression_lock_holder_process_is_dead(h))[0]
    elif action == 'renew':
        value = conn.execute('UPDATE compression_locks SET expires_at=? WHERE session_id=? AND holder=?',
                             (now + ttl, sid, holder)).rowcount > 0
    else:
        conn.execute('DELETE FROM compression_locks WHERE session_id=? AND holder=?', (sid, holder))
        value = None
    return {'value': value}


def worker_cooldown_record(db, conn, sid, payload):
    _fields(payload, ('cooldown_until', 'error'))
    deadline = _number(payload['cooldown_until'])
    error = payload['error']
    if error is not None and not isinstance(error, str):
        raise RuntimeStoreError('invalid_params')
    conn.execute('UPDATE sessions SET compression_failure_cooldown_until = CASE '
                 'WHEN compression_failure_cooldown_until IS NOT NULL AND compression_failure_cooldown_until > ? '
                 'THEN compression_failure_cooldown_until ELSE ? END, compression_failure_error=? WHERE id=?',
                 (deadline, deadline, error, sid))
    return {'value': None}


def worker_cooldown_restore(db, conn, sid, payload):
    _fields(payload, ('snapshot',))
    snapshot = payload['snapshot']
    _fields(snapshot, ('session_exists', 'cooldown_until', 'error'))
    if type(snapshot['session_exists']) is not bool:
        raise RuntimeStoreError('invalid_params')
    if not snapshot['session_exists']:
        raise RuntimeError('cannot restore absent compression cooldown row: session now exists')
    deadline, error = snapshot['cooldown_until'], snapshot['error']
    if deadline is not None:
        _number(deadline)
    if error is not None and not isinstance(error, str):
        raise RuntimeStoreError('invalid_params')
    conn.execute('UPDATE sessions SET compression_failure_cooldown_until=?, compression_failure_error=? WHERE id=?',
                 (deadline, error, sid))
    actual = conn.execute('SELECT compression_failure_cooldown_until,compression_failure_error FROM sessions WHERE id=?',
                          (sid,)).fetchone()
    if actual is None or tuple(actual) != (deadline, error):
        raise RuntimeError('compression cooldown rollback verification failed')
    return {'value': None}


def worker_cooldown_clear(db, conn, sid, payload):
    _fields(payload, ())
    return worker_cooldown_restore(db, conn, sid, {'snapshot': {
        'session_exists': True, 'cooldown_until': None, 'error': None}})


def worker_compression_counter(db, conn, sid, payload, *, column):
    _fields(payload, ('value',))
    value = payload['value']
    if value is not None:
        _number(value)
    if column != 'compression_recovery_deadline' and (type(value) is not int or value < 0):
        raise RuntimeStoreError('invalid_params')
    conn.execute(f'UPDATE sessions SET {column}=? WHERE id=?', (value, sid))
    return {'value': None}


class CompressionSnapshot:
    """Reuse production read algorithms on the receipt connection, never a second handle.

    Only read primitives are provided; this object cannot start/commit a write.
    Wire handlers below select concrete projections, not caller-named methods.
    """
    from hermes_state_compression import SessionCompressionMixin as _C
    from hermes_state_messages import SessionMessagesMixin as _M
    from hermes_state_sessions import SessionSessionsMixin as _S

    get_compression_chain = _C.get_compression_chain
    get_compression_tip = _C.get_compression_tip
    get_compression_lineage = _C.get_compression_lineage
    _is_compression_child_row = _C._is_compression_child_row
    _session_lineage_root_to_tip = _S._session_lineage_root_to_tip
    _is_explicit_branch_session = _S._is_explicit_branch_session
    declared_scope_identity = _S.declared_scope_identity
    _resume_lineage_ids = _M._resume_lineage_ids
    get_conversation_root = _M.get_conversation_root
    resolve_resume_session_id = _M.resolve_resume_session_id
    latest_conversation_boundary = _M.latest_conversation_boundary
    _is_explicit_fork_child_row = _M._is_explicit_fork_child_row

    def __init__(self, db, conn):
        self.db, self.conn = db, conn

    def _read_ctx(self):
        from contextlib import nullcontext
        return nullcontext(self.conn)

    def _read_one(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()

    def _read_all(self, sql, params=()):
        return self.conn.execute(sql, params).fetchall()

    def get_session(self, sid):
        from hermes_state_worker_context import worker_context
        return worker_context(self.db, self.conn, sid, {})['session']

    def authorize(self, assigned, target):
        _text(target)
        # Attribution ancestry is readable; unrelated siblings and foreign trees are not.
        allowed = self._session_lineage_root_to_tip(assigned)
        allowed += self.get_compression_chain(assigned)
        if target not in allowed:
            raise RuntimeStoreError('permission_denied')


def worker_lineage_context(db, conn, sid, payload):
    _fields(payload, ('target',))
    view = CompressionSnapshot(db, conn)
    target = payload['target']
    view.authorize(sid, target)
    return {'session': view.get_session(target)}


def worker_lineage(db, conn, sid, payload):
    _fields(payload, ('target',))
    view = CompressionSnapshot(db, conn)
    target = payload['target']
    view.authorize(sid, target)
    return {'readable_ids': view._session_lineage_root_to_tip(sid) + view.get_compression_chain(sid),
            'lineage': view.get_compression_lineage(target),
            'root': view.get_conversation_root(target),
            'tip': view.get_compression_tip(target),
            'resume': view.resolve_resume_session_id(target),
            'identity': view.declared_scope_identity(target)}


def worker_boundary(db, conn, sid, payload):
    _fields(payload, ('session_key', 'source'))
    view = CompressionSnapshot(db, conn)
    row = view.get_session(sid)
    if (row['session_key'], row['source']) != (payload['session_key'], payload['source']):
        raise RuntimeStoreError('permission_denied')
    return {'value': view.latest_conversation_boundary(payload['session_key'], payload['source'])}


def worker_history(db, conn, sid, payload):
    _fields(payload, ('target', 'include_ancestors', 'include_inactive', 'repair_alternation',
                      'include_row_ids', 'include_compacted'))
    if any(type(v) is not bool for k, v in payload.items() if k != 'target'):
        raise RuntimeStoreError('invalid_params')
    view = CompressionSnapshot(db, conn)
    target = payload['target']
    view.authorize(sid, target)
    ids = view._resume_lineage_ids(target) if payload['include_ancestors'] else [target]
    active = db._active_clause(payload['include_inactive'], payload['include_compacted'])
    rows = conn.execute(f'SELECT {db._CONVERSATION_ROW_COLUMNS} FROM messages '
                        f'WHERE session_id IN ({",".join("?" for _ in ids)}){active} ORDER BY id', ids).fetchall()
    if payload['include_compacted']:
        rows = db._dedupe_display_generations(rows)
    return {'messages': db._rows_to_conversation(rows, session_id=target,
        include_ancestors=payload['include_ancestors'], repair_alternation=payload['repair_alternation'],
        include_row_ids=payload['include_row_ids'])}



from hermes_state_common import _COMPRESSION_LOCK_ROW_SQL, is_automatic_end_reason, _placeholders
from hermes_state_messages import _ARCHIVE_ACTIVE_SQL, _SET_COUNTERS_SQL
from hermes_state_errors import CompressionSessionBusyError, SessionCompressionInProgressError
_LOCK_ROW_SQL = _COMPRESSION_LOCK_ROW_SQL


def archive_on_connection(db, conn, session_id: str, compacted_messages: List[Dict[str, Any]], model_config_patch: Optional[Dict[str, Any]]=None, watermark: Optional[int]=None, lock_holder: Optional[str]=None, tail_count: int=0):
    if lock_holder is not None:
        lock_row = conn.execute(_COMPRESSION_LOCK_ROW_SQL, (session_id,)).fetchone()
        if lock_row is None or lock_row['holder'] != lock_holder or float(lock_row['expires_at']) <= time.time():
            raise SessionCompressionInProgressError(f'Compression lease for {session_id!r} lost before commit; refusing to publish a stale compaction')
    patch = model_config_patch is not None
    patched_model_config = db._merge_model_config_json(conn, session_id, model_config_patch, on_missing='raise') if patch else None
    tail_ids, tail_tool_calls = ([], 0) if watermark is None else db._tail_rows_after_watermark(conn, 'SELECT id, tool_calls FROM messages WHERE session_id = ? AND active = 1 AND id > ? ORDER BY id', (session_id, int(watermark)))
    rewind_ids: list[int] = []
    if tail_count > 0:
        bound = watermark is not None
        rewind_ids = [int(row['id']) for row in conn.execute(f"SELECT id FROM messages WHERE session_id = ? AND active = 1{(' AND id <= ?' if bound else '')} ORDER BY id DESC LIMIT ?", (session_id, *((int(watermark),) if bound else ()), int(tail_count))).fetchall()]
    rewind_ids += tail_ids
    if rewind_ids:
        placeholders = _placeholders(rewind_ids)
        conn.execute(f'UPDATE messages SET active = 0, compacted = 0 WHERE session_id = ? AND id IN ({placeholders})', [session_id, *rewind_ids])
        conn.execute(f'{_ARCHIVE_ACTIVE_SQL} AND id NOT IN ({placeholders})', [session_id, *rewind_ids])
    else:
        conn.execute(_ARCHIVE_ACTIVE_SQL, (session_id,))
    inserted, tool_calls_total = db._insert_message_rows(conn, session_id, compacted_messages)
    if tail_ids:
        db._clone_message_rows(conn, tail_ids)
        inserted += len(tail_ids)
        tool_calls_total += tail_tool_calls
    conn.execute(f"{_SET_COUNTERS_SQL}{(', model_config = ?' if patch else '')} WHERE id = ?", (inserted, tool_calls_total, *((patched_model_config,) if patch else ()), session_id))
    return inserted

def publish_on_connection(db, conn, *, parent_session_id: str, child_session_id: str, source: str, messages: List[Dict[str, Any]], model: str=None, model_config: Dict[str, Any]=None, system_prompt: str=None, cwd: str=None, profile_name: str=None, compression_lock_holder: str=None, require_compression_lease: bool=True, require_lease_refresh: bool=False, lease_ttl_seconds: float=300.0, watermark: Optional[int]=None, watermark_ceiling: Optional[int]=None):
    if require_lease_refresh and compression_lock_holder:
        conn.execute('UPDATE compression_locks SET expires_at = ? WHERE session_id = ? AND holder = ?', (time.time() + lease_ttl_seconds, parent_session_id, compression_lock_holder))
    lock_row = conn.execute(_LOCK_ROW_SQL, (parent_session_id,)).fetchone()
    if require_compression_lease and (lock_row is None or not compression_lock_holder or lock_row['holder'] != compression_lock_holder or (float(lock_row['expires_at']) <= time.time())):
        raise CompressionSessionBusyError(f'Compression lease lost before publication: {parent_session_id}')
    parent = conn.execute('SELECT ended_at, end_reason, cwd, git_branch, git_repo_root,\n                          user_id, session_key, chat_id, chat_type,\n                          thread_id, display_name, origin_json, profile_name\n                   FROM sessions WHERE id = ?', (parent_session_id,)).fetchone()
    if parent is None:
        raise RuntimeError(f'Compression parent not found: {parent_session_id}')
    if parent['ended_at'] is not None:
        if not is_automatic_end_reason(parent['end_reason']):
            raise RuntimeError(f'Compression parent already ended: {parent_session_id}')
        conn.execute('UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?', (parent_session_id,))
    if not messages:
        raise RuntimeError('Compression child handoff must not be empty')
    db._publish_child_session_row(conn, parent, parent_session_id=parent_session_id, child_session_id=child_session_id, source=source, model=model, model_config=model_config, system_prompt=system_prompt, cwd=cwd, profile_name=profile_name)
    total_messages, total_tool_calls = db._insert_message_rows(conn, child_session_id, messages)
    if watermark is not None:
        bounded = watermark_ceiling is not None
        tail_ids, tail_tool_calls = db._tail_rows_after_watermark(conn, f"SELECT id, tool_calls FROM messages WHERE session_id = ? AND active = 1 AND id > ?{(' AND id <= ?' if bounded else '')} ORDER BY id", [parent_session_id, int(watermark), *([int(watermark_ceiling)] if bounded else [])])
        if tail_ids:
            db._clone_message_rows(conn, tail_ids, session_id=child_session_id)
            total_messages += len(tail_ids)
            total_tool_calls += tail_tool_calls
    conn.execute('UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?', (total_messages, total_tool_calls, child_session_id))
    updated = conn.execute("UPDATE sessions SET ended_at = ?, end_reason = 'compression' WHERE id = ? AND ended_at IS NULL", (time.time(), parent_session_id))
    if updated.rowcount != 1:
        raise RuntimeError(f'Compression parent changed during publication: {parent_session_id}')
    from hermes_state_local_lineage import advance_local_target
    advance_local_target(conn, parent_session_id, child_session_id)


def worker_watermark(db, conn, sid, payload):
    _fields(payload, ())
    return {'value': conn.execute('SELECT COALESCE(MAX(id),0) FROM messages WHERE session_id=? AND active=1',
                                  (sid,)).fetchone()[0]}


def _handoff_messages(conn, sid, messages):
    from hermes_state_runtime import _MESSAGE_FIELDS
    if not isinstance(messages, list) or len(messages) > 1000:
        raise RuntimeStoreError('invalid_params')
    for msg in messages:
        if not isinstance(msg, dict) or set(msg) - _MESSAGE_FIELDS - {'_compressed_summary_has_user_turn'} or msg.get('role') not in ('user', 'assistant', 'system', 'tool'):
            raise RuntimeStoreError('invalid_params')
        if '_row_id' in msg and not conn.execute('SELECT 1 FROM messages WHERE session_id=? AND id=?',
                                                 (sid, msg['_row_id'])).fetchone():
            raise RuntimeStoreError('permission_denied')


def _watermarks(payload):
    for field in ('watermark', 'watermark_ceiling'):
        value = payload.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise RuntimeStoreError('invalid_params')


def worker_archive(db, conn, sid, payload):
    from hermes_state_worker_context import SIDECAR_KEYS
    _fields(payload, ('messages', 'model_config_patch', 'watermark', 'lock_holder', 'tail_count'))
    _handoff_messages(conn, sid, payload['messages'])
    _watermarks(payload)
    _text(payload['lock_holder'])
    patch = payload['model_config_patch']
    if patch is not None and (not isinstance(patch, dict) or set(patch) - SIDECAR_KEYS):
        raise RuntimeStoreError('invalid_params')
    if type(payload['tail_count']) is not int or payload['tail_count'] < 0:
        raise RuntimeStoreError('invalid_params')
    value = archive_on_connection(db, conn, sid, payload['messages'], model_config_patch=patch,
        watermark=payload['watermark'], lock_holder=payload['lock_holder'], tail_count=payload['tail_count'])
    return {'value': value, 'row_ids': [message['_row_id'] for message in payload['messages']]}


def _migrate_worker_automation(db, conn, parent, child):
    from hermes_cli.goals import GoalState
    from hermes_cli.heartbeat import HeartbeatState
    from hermes_cli.loops import LoopState
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
        db.set_meta(new_key, state.to_json(), cursor=conn)
        state.status = 'cleared'
        db.set_meta(old_key, state.to_json(), cursor=conn)


def worker_publish(db, conn, sid, payload):
    from hermes_state_sessions import _parse_model_config
    _fields(payload, ('child_session_id', 'source', 'messages', 'model', 'model_config', 'system_prompt',
                      'cwd', 'profile_name', 'compression_lock_holder', 'require_compression_lease',
                      'require_lease_refresh', 'lease_ttl_seconds', 'watermark', 'watermark_ceiling'))
    child = _text(payload['child_session_id'])
    _text(payload['compression_lock_holder'])
    _watermarks(payload)
    if payload['require_compression_lease'] is not True or type(payload['require_lease_refresh']) is not bool:
        raise RuntimeStoreError('invalid_params')
    if not 0.1 <= _number(payload['lease_ttl_seconds']) <= 3600:
        raise RuntimeStoreError('invalid_params')
    _handoff_messages(conn, sid, payload['messages'])
    parent = CompressionSnapshot(db, conn).get_session(sid)
    for key in ('source', 'cwd', 'profile_name'):
        if payload[key] is not None and payload[key] != parent[key]:
            raise RuntimeStoreError('permission_denied')
    for key in ('model', 'system_prompt'):
        if payload[key] is not None and not isinstance(payload[key], str):
            raise RuntimeStoreError('invalid_params')
    cfg = payload['model_config']
    if cfg is not None and not isinstance(cfg, dict):
        raise RuntimeStoreError('invalid_params')
    cfg = dict(cfg or {})
    prior = _parse_model_config(parent['model_config'])
    for key in ('_branched_from', '_delegate_from', '_reset_from'):
        if key in cfg and cfg[key] != prior.get(key):
            raise RuntimeStoreError('permission_denied')
        if key in prior:
            cfg[key] = prior[key]
    if conn.execute('SELECT 1 FROM sessions WHERE id=?', (child,)).fetchone():
        raise RuntimeStoreError('permission_denied')
    worker = conn.execute("SELECT * FROM worker_executions WHERE session_id=? AND status IN ('registered','running')", (sid,)).fetchone()
    if worker is None:
        raise RuntimeStoreError('stale_generation')
    publish_on_connection(db, conn, parent_session_id=sid, **dict(payload, model_config=cfg))
    _migrate_worker_automation(db, conn, sid, child)
    conn.execute('UPDATE sessions SET runtime_generation=? WHERE id=?', (worker['generation'], child))
    conn.execute('UPDATE worker_executions SET session_id=? WHERE execution_id=?', (child, worker['execution_id']))
    # The FIFO/retry key is immutable; only physical lineage follows rotation.
    conn.execute("UPDATE session_admissions SET lineage_json=json_insert(lineage_json,'$[#]',?) "
                 "WHERE json_extract(lineage_json,'$[#-1]')=? AND generation=? AND owner_epoch=? "
                 "AND status IN ('started','unknown')",
                 (child, sid, worker['generation'], worker['owner_epoch']))
    return {'value': None, 'worker_assignment': {'parent': sid, 'session_id': child},
            'row_ids': [message['_row_id'] for message in payload['messages']]}


def worker_receipt_assignment(conn, execution_id, session_id, generation, sequence, digest):
    """Only an identical already-committed publication can retry its old assignment."""
    from hermes_state_runtime import _worker_assignment
    receipt = conn.execute('SELECT payload_digest,result_json FROM worker_receipts WHERE execution_id=? AND sequence=?',
                           (execution_id, sequence)).fetchone()
    if receipt and receipt['payload_digest'] == digest:
        assignment = json.loads(receipt['result_json']).get('worker_assignment')
        if assignment and assignment['parent'] == session_id:
            current = conn.execute('SELECT session_id FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
            return _worker_assignment(conn, execution_id, current['session_id'], generation)
    return _worker_assignment(conn, execution_id, session_id, generation)


def worker_claim_origin(db, params):
    """Keep the producer claim bound to its original identity after physical rotation."""
    with db._read_ctx() as conn:
        row = conn.execute("SELECT result_json FROM worker_receipts WHERE execution_id=? "
                           "AND json_extract(result_json,'$.worker_assignment.parent') IS NOT NULL ORDER BY sequence LIMIT 1",
                           (params['execution_id'],)).fetchone()
    return json.loads(row[0])['worker_assignment']['parent'] if row else params['session_id']


def worker_retry_target(db, params):
    """Transport verification accepts old targets only for exact publication replay."""
    from gateway.session_admission import admission_fingerprint
    if params.get('operation') != 'compression.publish':
        return params['session_id']
    digest = admission_fingerprint(canonical_target=params['session_id'],
        payload={'operation': params['operation'], 'payload': params['payload']})
    with db._read_ctx() as conn:
        row = worker_receipt_assignment(conn, params['execution_id'], params['session_id'],
                                        params['generation'], params['sequence'], digest)
    return row['session_id']


from hermes_state_common import _ENDED_ROW_SQL, _ended_by_compression

def reopen_on_connection(db, conn, session_id):
    if not _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()):
        return False
    child = conn.execute('\n                SELECT 1\n                FROM sessions\n                WHERE parent_session_id = ?\n                ' + db._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias='') + '\n                LIMIT 1\n                ', (session_id, session_id, session_id)).fetchone()
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


def worker_reopen(db, conn, sid, payload):
    _fields(payload, ())
    return {'value': reopen_on_connection(db, conn, sid)}


def worker_compression_cleanup(db, conn, sid, payload):
    _fields(payload, ('target', 'holder'))
    target = _text(payload['target'])
    holder = _text(payload['holder'])
    view = CompressionSnapshot(db, conn)
    view.authorize(sid, target)
    if db._session_turn_lease_key_on_conn(conn, sid) != db._session_turn_lease_key_on_conn(conn, target):
        raise RuntimeStoreError('permission_denied')
    conn.execute('DELETE FROM compression_locks WHERE session_id=? AND holder=?', (target, holder))
    return {'value': None}


def worker_compression_append(db, conn, sid, payload):
    _fields(payload, ('messages', 'compression_lock_holder', 'turn_lease_holder', 'turn_lease_ttl_seconds'))
    _handoff_messages(conn, sid, payload['messages'])
    for key in ('compression_lock_holder', 'turn_lease_holder'):
        if payload[key] is not None:
            _text(payload[key])
    if not 0.1 <= _number(payload['turn_lease_ttl_seconds']) <= 3600:
        raise RuntimeStoreError('invalid_params')
    count = db._append_messages_in_transaction(conn, sid, **payload)
    return {'count': count, 'annotations': [
        {key: msg[key] for key in ('_row_id', '_canonical_content') if key in msg} for msg in payload['messages']]}


def worker_turn_cleanup(db, conn, sid, payload):
    _fields(payload, ('target', 'holder'))
    target = _text(payload['target'])
    holder = _text(payload['holder'])
    CompressionSnapshot(db, conn).authorize(sid, target)
    key = db._session_turn_lease_key_on_conn(conn, sid)
    if key != db._session_turn_lease_key_on_conn(conn, target):
        raise RuntimeStoreError('permission_denied')
    conn.execute('DELETE FROM session_turn_leases WHERE conversation_id=? AND holder=?', (key, holder))
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
       for action in ('acquire', 'renew', 'release', 'holder')},
    'compression.cooldown.record': worker_cooldown_record,
    'compression.cooldown.restore': worker_cooldown_restore,
    'compression.cooldown.clear': worker_cooldown_clear,
    **{'compression.' + name: partial(worker_compression_counter, column=column)
       for name, column in (('fallback', 'compression_fallback_streak'),
                            ('ineffective', 'compression_ineffective_count'),
                            ('recovery', 'compression_recovery_deadline'))},
}
