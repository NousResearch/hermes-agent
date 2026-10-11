"""Canonical runtime transactions; callers authorize and resolve session identity.

No execution, transport, or authority objects live here. Every mutation uses the
SessionDB transaction owner, including its inode guard and SQLite retry policy.
"""
import json
import uuid

from agent.conversation_compression_archive import ABSORBED_ROW_IDS
from agent.message_metadata import (
    CANONICAL_ROW, DB_ROW_SNAPSHOT, MESSAGE_UID, PERSISTENCE_ONLY_MESSAGE_FIELDS, TOOL_CALL_UID, TOOL_CALL_UIDS,
)
from hermes_state_keys import admission_fingerprint


class RuntimeStoreError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _text(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise RuntimeStoreError('invalid_params')
    return value


def _json(value):
    if not isinstance(value, dict):
        raise RuntimeStoreError('invalid_params')
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeStoreError('invalid_params') from exc


def _epoch(conn, epoch):
    row = conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()
    if type(epoch) is not int or row is None or row[0] != epoch:
        raise RuntimeStoreError('stale_epoch')


def _session(conn, session_id):
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    return row


def _admission(conn, admission_id):
    row = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()
    if row is None:
        from hermes_state_terminal import terminal_admission
        row = terminal_admission(conn, admission_id)
    if row is None:
        raise RuntimeStoreError('not_found')
    return row


def _retire_admission_workers(conn, row, owner_epoch):
    """Retire the worker rows this admission's generation owns on BOTH the logical
    FIFO owner and its current physical transcript (reset/compression move the private
    worker assignment to the child while target_session_id stays the logical ID)."""
    lineage = json.loads(row['lineage_json'])
    targets = list({row['target_session_id'], lineage[-1] if lineage else row['target_session_id']})
    scope = f"session_id IN ({','.join('?' * len(targets))}) AND generation=? AND owner_epoch=?"
    # Settlement closes an execution that never wrote execution.finish: its read copies go too.
    from hermes_state_runtime_workers import compact_terminal_receipts
    for (execution_id,) in conn.execute(f"SELECT execution_id FROM worker_executions WHERE {scope} "
                                        "AND status!='terminal'", (*targets, row['generation'], owner_epoch)).fetchall():
        compact_terminal_receipts(conn, execution_id)
    conn.execute(f"UPDATE worker_executions SET status='terminal' WHERE {scope}",
                 (*targets, row['generation'], owner_epoch))


def _row(row):
    if row is None:
        return None
    result = dict(row)
    result['payload'] = json.loads(result.pop('payload_json'))
    result.pop('payload_digest')
    result.pop('lineage_json')
    return result


def begin_runtime_epoch(db, *, instance_id: str) -> int:
    """Caller MUST hold the profile ownership lock, not merely an endpoint ticket."""
    _text(instance_id)
    def write(conn):
        conn.execute('''INSERT INTO runtime_epoch(singleton,epoch,instance_id) VALUES(1,1,?)
            ON CONFLICT(singleton) DO UPDATE SET epoch=runtime_epoch.epoch+1, instance_id=excluded.instance_id''', (instance_id,))
        return conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()[0]
    return db._execute_write(write)


def admit_session_input(db, *, epoch: int, principal_id: str, session_id: str,
                        request_id: str, payload: dict, intent: str = 'queue',
                        _authorize_write=None) -> dict:
    """Admit input; the trusted private guard raises to refuse a NEW write.

    The guard receives the owning transaction connection, must not commit it or
    perform external effects, and may run again on SQLite retry. Exact existing
    and terminal replays bypass it: they cannot create or change accepted work.
    """
    for value in (principal_id, session_id, request_id):
        _text(value)
    if intent not in ('queue', 'steer', 'redirect'):
        raise RuntimeStoreError('invalid_params')
    encoded = _json(payload)
    from hermes_state_terminal import retry_terminal_admission
    retired = retry_terminal_admission(db, epoch=epoch, principal_id=principal_id, session_id=session_id,
        request_id=request_id, payload=payload, intent=intent)
    if retired is not None:
        return retired
    digest = admission_fingerprint(canonical_target=session_id, payload={'input': json.loads(encoded), 'intent': intent})
    def write(conn):
        _epoch(conn, epoch)
        _session(conn, session_id)
        old = conn.execute('''SELECT * FROM session_admissions
            WHERE principal_id=? AND target_session_id=? AND request_id=?''', (principal_id, session_id, request_id)).fetchone()
        if old is not None:
            if old['payload_digest'] != digest:
                raise RuntimeStoreError('admission_conflict')
            return _row(old)
        if _authorize_write is not None:
            _authorize_write(conn)
        admission_id = uuid.uuid4().hex
        conn.execute('''INSERT INTO session_admissions(admission_id,request_id,principal_id,
            target_session_id,lineage_json,payload_json,payload_digest,intent,status,owner_epoch)
            VALUES(?,?,?,?,?,?,?,?,'queued',?)''',
            (admission_id, request_id, principal_id, session_id, json.dumps([session_id]), encoded, digest, intent, epoch))
        return _row(_admission(conn, admission_id))
    return db._execute_write(write)


def get_session_admission(db, *, admission_id: str) -> dict | None:
    with db._read_ctx() as conn:
        from hermes_state_terminal import terminal_admission
        row = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()
        return _row(row if row is not None else terminal_admission(conn, admission_id))


def list_session_admissions(db, *, session_id: str, pending_only: bool = True) -> list[dict]:
    with db._read_ctx() as conn:
        # Pending is an explicit index-range IN, not ``!='terminal'``: a negated status cannot
        # bound the (target, status, seq) index, so every probe scanned the session's history.
        sql = ("SELECT * FROM session_admissions WHERE target_session_id=? "
               + ("AND status IN ('queued','started','unknown') " if pending_only else "") + "ORDER BY seq")
        return [_row(row) for row in conn.execute(sql, (session_id,))]


_ANY_HEAD = object()


def claim_session_input(db, *, epoch: int, session_id: str, _guard=None, head=_ANY_HEAD) -> dict | bool | None:
    """Claim the FIFO head. ``_guard(conn)`` (owner admission gate) raises to refuse inside the
    write transaction, so a drain that began while this waited on the writer still wins.
    ``head`` is the admission id the caller's preclaim checks validated (None: it saw nothing
    queued); if the head is another row by the time this write runs, nothing is claimed and False
    tells the caller to re-read and validate the row that is now first."""
    def write(conn):
        _epoch(conn, epoch)
        if _guard is not None:
            _guard(conn)
        session = _session(conn, session_id)
        blocked = conn.execute("SELECT status FROM session_admissions WHERE target_session_id=? AND status IN ('started','unknown')", (session_id,)).fetchall()
        if any(row[0] == 'unknown' for row in blocked):
            raise RuntimeStoreError('unknown_execution')
        from hermes_state_runtime_workers import worker_states
        workers = worker_states(conn, session_id)
        if 'unknown' in workers:
            raise RuntimeStoreError('unknown_execution')
        if blocked or workers:
            return None
        row = conn.execute("SELECT * FROM session_admissions WHERE target_session_id=? AND status='queued' ORDER BY seq LIMIT 1", (session_id,)).fetchone()
        if head is not _ANY_HEAD and (row['admission_id'] if row is not None else None) != head:
            return False
        if row is None:
            return None
        generation = session['runtime_generation'] + 1
        conn.execute('UPDATE sessions SET runtime_generation=? WHERE id=?', (generation, session_id))
        changed = conn.execute("UPDATE session_admissions SET status='started',owner_epoch=?,generation=? WHERE admission_id=? AND status='queued'", (epoch, generation, row['admission_id']))
        if changed.rowcount != 1:
            raise RuntimeStoreError('stale_generation')
        return _row(_admission(conn, row['admission_id']))
    return db._execute_write(write)


def settle_session_input(db, *, epoch: int, admission_id: str, generation: int, outcome: str,
                         result: dict | None = None) -> dict:
    if outcome not in ('completed', 'interrupted', 'rejected', 'failed'):
        raise RuntimeStoreError('invalid_params')
    from hermes_state_terminal import compact_result
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        session = _session(conn, row['target_session_id'])
        if (row['status'] != 'started' or row['owner_epoch'] != epoch
                or type(generation) is not int or row['generation'] != generation
                or session['runtime_generation'] != generation):
            raise RuntimeStoreError('stale_generation')
        _retire_admission_workers(conn, row, epoch)
        if result is not None:
            from hermes_state_terminal import RESULT_PREFIX
            encoded = _json(compact_result(result, user_message=json.loads(row['payload_json']).get('text')))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                         'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         (RESULT_PREFIX + admission_id, encoded))
        conn.execute("UPDATE session_admissions SET status='terminal',outcome=? WHERE admission_id=?", (outcome, admission_id))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return _row(_admission(conn, admission_id))
    return db._execute_write(write)


def cancel_session_input(db, *, epoch: int, admission_id: str) -> dict:
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if row['status'] == 'unknown':
            raise RuntimeStoreError('unknown_execution')
        if row['status'] == 'started':
            raise RuntimeStoreError('stale_generation')
        if row['status'] == 'queued':
            conn.execute("UPDATE session_admissions SET status='terminal',outcome='cancelled' WHERE admission_id=?", (admission_id,))
        return _row(_admission(conn, admission_id))
    return db._execute_write(write)


def recover_session_inputs(db, *, epoch: int) -> int:
    """Never replay started work. Live worker adoption is a separate explicit operation."""
    def write(conn):
        _epoch(conn, epoch)
        conn.execute("UPDATE worker_executions SET status='unknown' WHERE status IN ('registered','running') AND owner_epoch!=?", (epoch,))
        return conn.execute("UPDATE session_admissions SET status='unknown' WHERE status='started' AND owner_epoch!=?", (epoch,)).rowcount
    return db._execute_write(write)


def mutate_runtime_session(db, *, epoch: int, principal_id: str, session_id: str,
                           request_id: str, expected_revision: int,
                           operation: str, payload: dict, expected_generation: int | None = None,
                           _live_guard=None, _authorize_write=None, _prepare_only=False, _prepared=None) -> dict:
    """Commit a closed metadata edit and its retry receipt in the same transaction.

    Caller authorizes the principal and resolves the canonical session. These
    metadata edits do not stop execution. Existing direct writers must migrate
    before this seam can provide universal revision fencing.
    """
    for value in (principal_id, session_id, request_id):
        _text(value)
    if type(expected_revision) is not int or expected_revision < 0:
        raise RuntimeStoreError('invalid_params')
    from hermes_state_mutations import validate_action, apply_action
    validate_action(operation, payload)
    if expected_generation is not None and (type(expected_generation) is not int or expected_generation < 0):
        raise RuntimeStoreError('invalid_params')
    if operation in {'delete', 'rewind', 'reset', 'branch', 'model', 'compress'} and expected_generation is None:
        raise RuntimeStoreError('invalid_params')
    # Snapshot caller data before waiting for the writer lock.
    payload = json.loads(_json(payload))
    key = 'gateway.mutation.v1.' + admission_fingerprint(
        canonical_target=session_id, payload={'principal': principal_id, 'request': request_id})
    fingerprint = {'operation': operation, 'payload': payload, 'expected_revision': expected_revision}
    if expected_generation is not None:
        fingerprint['expected_generation'] = expected_generation
    digest = admission_fingerprint(canonical_target=session_id, payload=fingerprint)

    def write(conn):
        _epoch(conn, epoch)
        old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if old is not None:
            receipt = json.loads(old[0])
            if receipt['digest'] != digest:
                raise RuntimeStoreError('admission_conflict')
            return receipt['result']
        session = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
        if session is None and operation != 'import':
            raise RuntimeStoreError('not_found')
        revision = session['runtime_revision'] if session else 0
        if revision != expected_revision:
            raise RuntimeStoreError('revision_conflict')
        if expected_generation is not None and (session is None or session['runtime_generation'] != expected_generation):
            raise RuntimeStoreError('stale_generation')
        if _live_guard is not None:
            from hermes_state_mutation_guards import delete_targets
            targets = delete_targets(conn, session_id) if operation == 'delete' else [session_id]
            _live_guard(targets)
        if _authorize_write is not None:
            _authorize_write(conn)
        if _prepare_only:
            from hermes_state_mutation_prepared import local_snapshot
            return {'snapshot': local_snapshot(db, conn, session_id)}
        affected, projection = apply_action(db, conn, session_id, operation, payload, prepared=_prepared)
        conn.executemany('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?',
                         [(target,) for target in affected])
        updated = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
        result = {'session_id': session_id, 'revision': updated['runtime_revision'] if updated else expected_revision + 1,
                  'execution_generation': updated['runtime_generation'] if updated else expected_generation,
                  'operation': operation, **projection}
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (key, _json({'digest': digest, 'result': result})))
        return result
    return db._execute_write(write)


_IMPORT_KEY = 'gateway.prompt_admissions_import.v1'


def _canonical_chain(conn, session_id):
    # Same selector as SessionDB.get_compression_chain, on OUR transaction connection.
    from hermes_state_compression import _CHAIN_STEP_SQL
    _session(conn, session_id)
    chain = [session_id]
    for _ in range(100):
        child = conn.execute(_CHAIN_STEP_SQL, (chain[-1],)).fetchone()
        if child is None:
            return chain
        if child['id'] in chain:
            raise RuntimeStoreError('admission_conflict')
        chain.append(child['id'])
    raise RuntimeStoreError('admission_conflict')


def _import_legacy_row(conn, row, epoch, principal_id):
    for field in ('admission_id', 'target_session_id', 'root'):
        _text(row[field])
    lineage = json.loads(row['lineage'])
    if not isinstance(lineage, list) or row['target_session_id'] not in lineage:
        raise RuntimeStoreError('invalid_params')
    payload = json.loads(row['payload'])
    encoded = _json(payload)
    intent = payload.get('intent', 'queue')
    intent = 'redirect' if intent == 'interrupt' else intent
    if intent not in ('queue', 'steer', 'redirect') or row['status'] not in ('queued', 'started', 'unknown', 'terminal'):
        raise RuntimeStoreError('invalid_params')
    chain = _canonical_chain(conn, row['target_session_id'])
    target = chain[-1]
    status = 'unknown' if row['status'] == 'started' else row['status']
    generation = row['generation']
    if generation is None and status == 'unknown':
        generation = _session(conn, target)['runtime_generation'] + 1
    if generation is not None and (type(generation) is not int or generation < 0):
        raise RuntimeStoreError('invalid_params')
    digest = admission_fingerprint(canonical_target=target, payload={'input': payload, 'intent': intent})
    conn.execute("""INSERT INTO session_admissions(admission_id,request_id,principal_id,
        target_session_id,lineage_json,payload_json,payload_digest,intent,status,outcome,owner_epoch,generation)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (row['admission_id'], row['admission_id'], principal_id,
        target, json.dumps(chain), encoded, digest, intent, status, row['outcome'], epoch, generation))
    if generation is not None:
        conn.execute('UPDATE sessions SET runtime_generation=MAX(runtime_generation,?) WHERE id=?', (generation, target))


def import_legacy_session_admissions(db, *, epoch: int, source_path, principal_id: str,
                                     writers_drained: bool) -> int:
    """Import a frozen legacy snapshot; original file remains rollback evidence.

    The caller proves the old writers are drained before invoking this function.
    Fingerprinting uses SQL snapshot data, never a raw open/close on a live inode.
    """
    from contextlib import closing
    from pathlib import Path
    import hashlib
    import sqlite3
    from hermes_cli.sqlite_safe_read import connect_tracked
    if writers_drained is not True:
        raise RuntimeStoreError('invalid_params')
    _text(principal_id)
    source_path = Path(source_path).resolve(strict=True)
    with closing(connect_tracked(source_path.as_uri() + '?mode=ro', uri=True)) as source:
        source.row_factory = sqlite3.Row
        source.execute('PRAGMA query_only=ON')
        source.execute('BEGIN')
        rows = [dict(r) for r in source.execute('SELECT * FROM admissions ORDER BY seq')]
        executions = []
        if source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='executions'").fetchone():
            executions = [dict(r) for r in source.execute('SELECT * FROM executions ORDER BY root')]
        fingerprint = hashlib.sha256(_json({'rows': rows, 'executions': executions}).encode('utf-8')).hexdigest()
        marker = _json({'source': str(source_path), 'fingerprint': fingerprint, 'imported_id_count': len(rows), 'principal_id': principal_id})
        def write(conn):
            _epoch(conn, epoch)
            old = conn.execute('SELECT value FROM state_meta WHERE key=?', (_IMPORT_KEY,)).fetchone()
            if old is not None:
                if old[0] != marker:
                    raise RuntimeStoreError('admission_conflict')
                return 0
            for execution in executions:
                generation = execution['generation']
                if type(generation) is not int or generation < 0:
                    raise RuntimeStoreError('invalid_params')
                target = _canonical_chain(conn, _text(execution['root']))[-1]
                conn.execute('UPDATE sessions SET runtime_generation=MAX(runtime_generation,?) WHERE id=?', (generation, target))
            for row in rows:
                _import_legacy_row(conn, row, epoch, principal_id)
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (_IMPORT_KEY, marker))
            return len(rows)
        return db._execute_write(write)


def resolve_unknown_session_input(db, *, epoch: int, admission_id: str, generation: int,
                                  _terminal_write=None, _captured=None) -> dict:
    """Explicit operator acknowledgement; resolves uncertainty, never requeues it.

    ``_terminal_write(conn, row)`` is the owner's same-transaction effect (the discarded turn's
    transcript boundary): it must not commit, may run again on SQLite retry, and raising rolls
    the whole resolution back, so the FIFO can never advance past an unclosed lost turn.
    ``_captured=(outcome, stored_result)`` is the exact result this owner captured for the turn
    before its settlement write failed; it commits with the resolution (only for an ``unknown``
    row stamped by this same owner epoch), so the finished answer is not thrown away."""
    from hermes_state_terminal import RESULT_PREFIX
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if (type(generation) is int and row['generation'] == generation and row['status'] == 'terminal'
                and (row['outcome'] == 'interrupted' or conn.execute(
                    'SELECT 1 FROM state_meta WHERE key=?', (RESULT_PREFIX + admission_id,)).fetchone())):
            return _row(row)
        if type(generation) is not int or row['status'] != 'unknown' or row['generation'] != generation:
            raise RuntimeStoreError('stale_generation')
        outcome = 'interrupted'
        if _captured is not None and row['owner_epoch'] == epoch:
            outcome, stored = _captured
            if outcome not in ('completed', 'interrupted', 'failed'):
                raise RuntimeStoreError('invalid_params')
            from hermes_state_terminal import compact_result
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                         'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         (RESULT_PREFIX + admission_id, _json(compact_result(
                             stored, user_message=json.loads(row['payload_json']).get('text')))))
        if _terminal_write is not None:
            _terminal_write(conn, row)
        _retire_admission_workers(conn, row, row['owner_epoch'])
        conn.execute("UPDATE session_admissions SET status='terminal',outcome=? WHERE admission_id=?",
                     (outcome, admission_id))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return _row(_admission(conn, admission_id))
    return db._execute_write(write)


def _worker_public(row):
    result = dict(row)
    result.pop('adoption_digest')
    return result


def _worker_assignment(conn, execution_id, session_id, generation):
    row = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    if row['session_id'] != session_id:
        raise RuntimeStoreError('permission_denied')
    if (type(generation) is not int or row['generation'] != generation
            or _session(conn, session_id)['runtime_generation'] != generation):
        raise RuntimeStoreError('stale_generation')
    return row


def _linked_worker_admission(conn, session_id, generation, owner_epoch, statuses):
    placeholders = ','.join('?' for _ in statuses)
    rows = conn.execute(f"""SELECT * FROM session_admissions
        WHERE generation=? AND owner_epoch=? AND status IN ({placeholders})
          AND (target_session_id=? OR EXISTS (
              SELECT 1 FROM json_each(session_admissions.lineage_json) WHERE value=?
          )) ORDER BY seq""",
        (generation, owner_epoch, *statuses, session_id, session_id)).fetchall()
    if len(rows) > 1:
        raise RuntimeStoreError('admission_conflict')
    return rows[0] if rows else None


def _secret_digest(secret):
    import hashlib
    _text(secret)
    return hashlib.sha256(secret.encode('utf-8')).hexdigest()


def register_worker_execution(db, *, epoch: int, execution_id: str, session_id: str,
                              generation: int, kind: str, adoption_secret: str, require_idle: bool = False) -> dict:
    for value in (execution_id, session_id):
        _text(value)
    if kind not in ('cron', 'child', 'compute', 'kanban'):
        raise RuntimeStoreError('invalid_params')
    digest = _secret_digest(adoption_secret)
    def write(conn):
        _epoch(conn, epoch)
        session = _session(conn, session_id)
        if type(generation) is not int or session['runtime_generation'] != generation:
            raise RuntimeStoreError('stale_generation')
        old = conn.execute('SELECT * FROM worker_executions WHERE execution_id=?', (execution_id,)).fetchone()
        if old is not None:
            if (old['session_id'], old['generation'], old['kind'], old['owner_epoch'], old['adoption_digest']) != (session_id, generation, kind, epoch, digest):
                raise RuntimeStoreError('admission_conflict')
            return _worker_public(old)
        from hermes_state_runtime_workers import runtime_lineage, worker_states
        targets = runtime_lineage(conn, session_id)
        if require_idle and any(conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown')", (sid,)).fetchone() for sid in targets):
            raise RuntimeStoreError('stale_generation')
        if worker_states(conn, session_id):
            raise RuntimeStoreError('stale_generation')
        if any(conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='unknown'", (sid,)).fetchone() for sid in targets):
            raise RuntimeStoreError('unknown_execution')
        conn.execute("""INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest)
            VALUES(?,?,?,?,?,'registered',?)""", (execution_id, session_id, kind, epoch, generation, digest))
        return _worker_public(_worker_assignment(conn, execution_id, session_id, generation))
    return db._execute_write(write)


def adopt_worker_execution(db, *, epoch: int, execution_id: str, session_id: str,
                           generation: int, adoption_secret: str) -> dict:
    """Authority first verifies the original producer claim and live worker proof."""
    import hmac
    digest = _secret_digest(adoption_secret)
    def write(conn):
        _epoch(conn, epoch)
        row = _worker_assignment(conn, execution_id, session_id, generation)
        if row['status'] == 'terminal':
            raise RuntimeStoreError('stale_generation')
        if not hmac.compare_digest(row['adoption_digest'], digest):
            raise RuntimeStoreError('permission_denied')
        # An admission-reserved worker reports to the reserving owner over that owner's pipes and
        # dies with them, so only its same-epoch handshake adopts it. A late handshake reaching a
        # restarted owner would re-arm the recovered `unknown` admission as `started` with nobody
        # left to settle it (and Discard refuses `started`); refuse it so recovery's verdict stands.
        if execution_id.startswith('admission-worker:') and row['owner_epoch'] != epoch:
            raise RuntimeStoreError('stale_epoch')
        linked = _linked_worker_admission(
            conn, session_id, generation, row['owner_epoch'], ('started', 'unknown'))
        if linked is not None:
            changed = conn.execute("""UPDATE session_admissions SET owner_epoch=?,status='started'
                WHERE admission_id=? AND generation=? AND owner_epoch=?
                AND status IN ('started','unknown')""",
                (epoch, linked['admission_id'], generation, row['owner_epoch']))
            if changed.rowcount != 1:
                raise RuntimeStoreError('stale_generation')
        changed = conn.execute(
            "UPDATE worker_executions SET owner_epoch=?,status='running' WHERE execution_id=?",
            (epoch, execution_id))
        if changed.rowcount != 1:
            raise RuntimeStoreError('stale_generation')
        return _worker_public(_worker_assignment(conn, execution_id, session_id, generation))
    return db._execute_write(write)


_MESSAGE_FIELDS = frozenset({
    'role', 'content', 'tool_name', 'tool_calls', 'tool_call_id', 'token_count',
    'finish_reason', 'reasoning', 'reasoning_content', 'reasoning_details',
    'codex_reasoning_items', 'codex_message_items', 'platform_message_id', 'message_id',
    'observed', 'effect_disposition', '_compressed_summary', 'timestamp', 'api_content',
    'display_kind', 'display_metadata', '_row_id', '_canonical_content', DB_ROW_SNAPSHOT, CANONICAL_ROW,
    # Main's durable message identity (message_uid / merge witness / tool-call uids) and the alternation
    # repair's row counts ride every persisted dict; the owner's insert and coverage read them.
}) | PERSISTENCE_ONLY_MESSAGE_FIELDS | {ABSORBED_ROW_IDS}
# Compaction handoffs (archive/publish) also carry the summary's user-turn flag. This is the closed
# boundary the owner validates AND the projection the worker facade applies to live dicts first.
HANDOFF_MESSAGE_FIELDS = _MESSAGE_FIELDS | {'_compressed_summary_has_user_turn'}
# Row state the owner's insert and transcript repair stamp on each message (main mutates the caller's
# dict in place; a worker gets it back as an annotation). None = absent, so a stale adoption is cleared.
# The owner-minted identity rides too: a worker dict without it re-mints a new uid on every later
# archive/rotation copy, so the logical message changes identity across generations.
_ROW_ANNOTATION_KEYS = ('_row_id', 'timestamp', DB_ROW_SNAPSHOT, CANONICAL_ROW,
                        MESSAGE_UID, TOOL_CALL_UIDS, TOOL_CALL_UID)


def row_annotations(messages):
    return [{key: msg.get(key) for key in _ROW_ANNOTATION_KEYS} for msg in messages]


def _worker_append(db, conn, session_id, payload):
    if set(payload) - {'messages', 'turn_lease_holder'} or 'messages' not in payload:
        raise RuntimeStoreError('invalid_params')
    messages = payload['messages']
    if not isinstance(messages, list) or len(messages) > 1000:
        raise RuntimeStoreError('invalid_params')
    for msg in messages:
        if (not isinstance(msg, dict) or set(msg) - _MESSAGE_FIELDS
                or msg.get('role') not in ('user', 'assistant', 'system', 'tool')):
            raise RuntimeStoreError('invalid_params')
        # Existing row annotations can adopt only rows in THIS transcript.
        if '_row_id' in msg and not conn.execute(
                'SELECT 1 FROM messages WHERE id=? AND session_id=?',
                (msg['_row_id'], session_id)).fetchone():
            raise RuntimeStoreError('permission_denied')
    holder = payload.get('turn_lease_holder')
    if holder is not None:
        _text(holder)
    count = db._append_messages_in_transaction(conn, session_id, messages, turn_lease_holder=holder)
    return {'count': count, 'annotations': row_annotations(messages)}


def _worker_turn(db, conn, session_id, payload, operation):
    import time
    from hermes_state_compression import _claim_lease_row
    from hermes_state import _compression_lock_holder_process_is_dead
    allowed = {'holder'} if operation == 'turn.release' else {'holder', 'ttl_seconds'}
    if set(payload) != allowed:
        raise RuntimeStoreError('invalid_params')
    holder = _text(payload['holder'])
    ttl = payload.get('ttl_seconds', 300)
    if type(ttl) not in (float, int) or not 0.1 <= ttl <= 3600:
        raise RuntimeStoreError('invalid_params')
    key = db._session_turn_lease_key_on_conn(conn, session_id)
    now = time.time()
    if operation == 'turn.acquire':
        value = _claim_lease_row(conn, 'session_turn_leases', 'conversation_id', key,
            holder, now, now + ttl,
            lambda h, e: float(e) <= now or _compression_lock_holder_process_is_dead(h))[0]
    elif operation == 'turn.renew':
        value = conn.execute('UPDATE session_turn_leases SET expires_at=? WHERE conversation_id=? AND holder=?',
                             (now + ttl, key, holder)).rowcount > 0
    else:
        conn.execute('DELETE FROM session_turn_leases WHERE conversation_id=? AND holder=?', (key, holder))
        value = None
    return {'value': value}


def _worker_usage(db, conn, session_id, payload, *, auxiliary=False):
    from hermes_state_usage import _MODEL_USAGE_FIELDS, _TOKEN_COUNTERS
    if not auxiliary:
        # ``source`` feeds the legacy path's row-existence guard (#111999); an authority-owned
        # session row was minted with its real surface at admission, so nothing to repair here.
        payload = {k: v for k, v in payload.items() if k != 'source'}
    allowed = (_MODEL_USAGE_FIELDS - {'billing_mode', 'actual_cost_usd', 'cost_status', 'cost_source'} | {'task'}) if auxiliary else (_MODEL_USAGE_FIELDS | {'pricing_version', 'absolute', 'task'})
    if set(payload) - allowed:
        raise RuntimeStoreError('invalid_params')
    for key, value in payload.items():
        if key in (*_TOKEN_COUNTERS, 'api_call_count'):
            valid = type(value) is int and 0 <= value <= 2**53
        elif key in ('estimated_cost_usd', 'actual_cost_usd'):
            valid = value is None or (type(value) in (int, float) and 0 <= value <= 1e12)
        elif key == 'absolute':
            valid = type(value) is bool
        else:
            valid = value is None or (isinstance(value, str) and len(value) <= 4096)
        if not valid:
            raise RuntimeStoreError('invalid_params')
    if auxiliary:
        if not payload.get('task'):
            raise RuntimeStoreError('invalid_params')
        db._record_model_usage(conn, session_id, **({'api_call_count': 1} | payload))
    else:
        db._update_token_counts_in_transaction(conn, session_id, **payload)
    return {'value': None}


def _worker_finish(db, conn, session_id, payload):
    if payload:
        raise RuntimeStoreError('invalid_params')
    return {'status': 'terminal'}


def mutate_worker_execution(db, *, epoch, execution_id, session_id, generation,
                            sequence, operation, payload):
    """One closed durable mutation and receipt; never call a self-committing API here."""
    from hermes_state_worker_context import (worker_context, worker_prompt, worker_row_role, worker_sidecars,
                                             worker_tail_role, worker_tool_names)
    from hermes_state_worker_compression import WORKER_COMPRESSION_HANDLERS
    from hermes_state_worker_lifecycle import WORKER_LIFECYCLE_HANDLERS
    handlers = {
        **WORKER_LIFECYCLE_HANDLERS,
        **WORKER_COMPRESSION_HANDLERS,
        'session.context': worker_context,
        'session.prompt': worker_prompt,
        'session.sidecars': worker_sidecars,
        'session.tools': worker_tool_names,
        'session.tail_role': worker_tail_role,
        'session.row_role': worker_row_role,
        'transcript.append': _worker_append,
        'execution.finish': _worker_finish,
        'usage.main': _worker_usage,
        'usage.auxiliary': lambda db, conn, sid, p: _worker_usage(db, conn, sid, p, auxiliary=True),
        **{name: (lambda db, conn, sid, p, op=name: _worker_turn(db, conn, sid, p, op))
           for name in ('turn.acquire', 'turn.renew', 'turn.release')},
    }
    return apply_worker_receipt(db, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                generation=generation, sequence=sequence, operation=operation,
                                payload=payload, handlers=handlers)


def apply_worker_receipt(db, *, epoch, execution_id, session_id, generation, sequence,
                         operation, payload, handlers):
    """The one worker receipt transaction every ``worker.persist`` family shares.

    An exact (execution, sequence, digest) retry returns the stored result; otherwise the
    epoch, assignment, live status and next sequence are fenced, ``handlers[operation]`` runs on
    the transaction connection (it must not commit), and its result is stored as the receipt.
    ``execution.finish`` is the only operation that makes the execution terminal.
    """
    from hermes_state_worker_compression import worker_receipt_assignment
    if type(sequence) is not int or sequence < 1 or not isinstance(operation, str) or operation not in handlers:
        raise RuntimeStoreError('invalid_params')
    encoded = _json(payload)
    if len(encoded.encode('utf-8', errors='surrogatepass')) > 4 * 1024 * 1024:
        raise RuntimeStoreError('invalid_params')
    digest = admission_fingerprint(canonical_target=session_id,
                                  payload={'operation': operation, 'payload': json.loads(encoded)})
    def write(conn):
        _epoch(conn, epoch)
        row = worker_receipt_assignment(conn, execution_id, session_id, generation, sequence, digest)
        if row['owner_epoch'] != epoch:
            raise RuntimeStoreError('stale_epoch')
        old = conn.execute('SELECT * FROM worker_receipts WHERE execution_id=? AND sequence=?',
                           (execution_id, sequence)).fetchone()
        if old is not None:
            if old['payload_digest'] != digest:
                raise RuntimeStoreError('admission_conflict')
            from hermes_state_runtime_workers import retired_receipt
            if retired_receipt(old['result_json']):
                raise RuntimeStoreError('stale_generation')  # a terminal execution's read copy
            return json.loads(old['result_json'])
        if row['status'] not in ('registered', 'running'):
            raise RuntimeStoreError('stale_generation')
        if sequence != row['last_sequence'] + 1:
            raise RuntimeStoreError('invalid_params')
        # Each SQLite retry gets fresh rows; rolled-back annotations must not escape.
        changes = conn.total_changes
        result = handlers[operation](db, conn, session_id, json.loads(encoded))
        # The CAS fence moves only when the handler changed durable state (or the execution
        # closes): a read-only receipt (context, history, lineage, lifecycle) must not make a
        # client's mutation built on the pre-read revision fail revision_conflict.
        changed = conn.total_changes != changes or operation == 'execution.finish'
        conn.execute('INSERT INTO worker_receipts(execution_id,sequence,payload_digest,result_json) VALUES(?,?,?,?)',
                     (execution_id, sequence, digest, json.dumps(result, ensure_ascii=True, allow_nan=False)))
        conn.execute("UPDATE worker_executions SET last_sequence=?,status=? WHERE execution_id=?",
                     (sequence, 'terminal' if operation == 'execution.finish' else 'running', execution_id))
        if changed:
            conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (session_id,))
        if operation == 'execution.finish':
            from hermes_state_runtime_workers import compact_terminal_receipts
            compact_terminal_receipts(conn, execution_id)
        return result
    return db._execute_write(write, patience_s=db._TRANSCRIPT_WRITE_PATIENCE_S)
