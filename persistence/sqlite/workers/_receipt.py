"""The worker receipt transaction and the runtime operations it runs.

:func:`apply_worker_receipt` is the one receipt transaction every receipted worker verb runs; :func:`mutate_worker_execution`
builds the handler table of the closed worker operations and runs it. The operations ``session.context``,
``turn.release`` and ``compression.lock.release`` have no producer and no handler, so the dispatcher refuses them as it
refuses any unknown operation. Handlers take the store holder and the transaction's connection, and never commit.
"""

import json

from persistence.authority.vocabulary import admission_fingerprint
from persistence.core.errors import AuthorityRefusal
from persistence.workers.vocabulary import WORKER_MESSAGE_FIELDS, row_annotations

from ..authority._runtime import _epoch, _json, _text

# The contract vocabulary keeps the one copy.
_MESSAGE_FIELDS = WORKER_MESSAGE_FIELDS


def _worker_append(holder, conn, session_id, payload):
    if set(payload) - {'messages', 'turn_lease_holder'} or 'messages' not in payload:
        raise AuthorityRefusal('invalid_params')
    messages = payload['messages']
    if not isinstance(messages, list) or len(messages) > 1000:
        raise AuthorityRefusal('invalid_params')
    for msg in messages:
        if (not isinstance(msg, dict) or set(msg) - _MESSAGE_FIELDS
                or msg.get('role') not in ('user', 'assistant', 'system', 'tool')):
            raise AuthorityRefusal('invalid_params')
        # Existing row annotations can adopt only rows in THIS transcript.
        if '_row_id' in msg and not conn.execute(
                'SELECT 1 FROM messages WHERE id=? AND session_id=?',
                (msg['_row_id'], session_id)).fetchone():
            raise AuthorityRefusal('permission_denied')
    lease_holder = payload.get('turn_lease_holder')
    if lease_holder is not None:
        _text(lease_holder)
    from ..conversation._store.messages import _append_messages_in_transaction
    count = _append_messages_in_transaction(holder, conn, session_id, messages, turn_lease_holder=lease_holder)
    return {'count': count, 'annotations': row_annotations(messages)}


def _worker_turn(holder, conn, session_id, payload, operation):
    import time
    from ..authority.coordination import _claim_lease_row
    from ..conversation._store.compression import _session_turn_lease_key_on_conn
    from process_identity import holder_process_is_dead
    if set(payload) != {'holder', 'ttl_seconds'}:
        raise AuthorityRefusal('invalid_params')
    lease_holder = _text(payload['holder'])
    ttl = payload.get('ttl_seconds', 300)
    if type(ttl) not in (float, int) or not 0.1 <= ttl <= 3600:
        raise AuthorityRefusal('invalid_params')
    key = _session_turn_lease_key_on_conn(holder, conn, session_id)
    now = time.time()
    if operation == 'turn.acquire':
        value = _claim_lease_row(conn, 'session_turn_leases', 'conversation_id', key,
            lease_holder, now, now + ttl,
            lambda h, e: float(e) <= now or holder_process_is_dead(h))[0]
    else:
        value = conn.execute('UPDATE session_turn_leases SET expires_at=? WHERE conversation_id=? AND holder=?',
                             (now + ttl, key, lease_holder)).rowcount > 0
    return {'value': value}


def _worker_usage(holder, conn, session_id, payload, *, auxiliary=False):
    from ..conversation._store.usage import (
        _MODEL_USAGE_FIELDS, _TOKEN_COUNTERS, _record_model_usage, _update_token_counts_in_transaction,
    )
    source = None
    if not auxiliary:
        # ``source`` is the worker platform (turn_usage._agent_session_source). A Desktop
        # managed worker passes ``gui``; created_surface maps that to ``desktop``. The row
        # is usually stamped at admission. A compression child can still be NULL (or an
        # auxiliary-usage ``unknown``), and this worker never create_session's it, so fill
        # only an unset column, as the creator upsert does.
        raw = payload.get('source')
        source = raw if isinstance(raw, str) else None
        payload = {k: v for k, v in payload.items() if k != 'source'}
    allowed = (_MODEL_USAGE_FIELDS - {'billing_mode', 'actual_cost_usd', 'cost_status', 'cost_source'} | {'task'}) if auxiliary else (_MODEL_USAGE_FIELDS | {'pricing_version', 'absolute', 'task'})
    if set(payload) - allowed:
        raise AuthorityRefusal('invalid_params')
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
            raise AuthorityRefusal('invalid_params')
    if auxiliary:
        if not payload.get('task'):
            raise AuthorityRefusal('invalid_params')
        _record_model_usage(holder, conn, session_id, **({'api_call_count': 1} | payload))
    else:
        _update_token_counts_in_transaction(holder, conn, session_id, **payload)
        # The managed worker's platform is the policy label (gui), not the in-process
        # agent surface (desktop). Stamp the same created_surface the insert uses.
        from session_rules.surfaces import created_surface
        from ..conversation._store.compression import _UNSET_CREATED_SQL, stamp_compression_ancestors
        surfaced = created_surface(source)
        if surfaced:
            conn.execute(
                "UPDATE sessions SET created_source=? WHERE id=? AND "
                + _UNSET_CREATED_SQL.format(col="created_source"),
                (surfaced, session_id))
            stamp_compression_ancestors(conn, session_id)
    return {'value': None}


def _worker_finish(holder, conn, session_id, payload):
    if payload:
        raise AuthorityRefusal('invalid_params')
    return {'status': 'terminal'}


def mutate_worker_execution(holder, *, epoch, execution_id, session_id, generation,
                            sequence, operation, payload):
    """One closed durable mutation and receipt; never call a self-committing API here."""
    from ._context import worker_prompt, worker_row_role, worker_sidecars, worker_tail_role, worker_tool_names
    from ._compression import WORKER_COMPRESSION_HANDLERS
    from ._lifecycle import WORKER_LIFECYCLE_HANDLERS
    handlers = {
        **WORKER_LIFECYCLE_HANDLERS,
        **WORKER_COMPRESSION_HANDLERS,
        'session.prompt': worker_prompt,
        'session.sidecars': worker_sidecars,
        'session.tools': worker_tool_names,
        'session.tail_role': worker_tail_role,
        'session.row_role': worker_row_role,
        'transcript.append': _worker_append,
        'execution.finish': _worker_finish,
        'usage.main': _worker_usage,
        'usage.auxiliary': lambda holder, conn, sid, p: _worker_usage(holder, conn, sid, p, auxiliary=True),
        **{name: (lambda holder, conn, sid, p, op=name: _worker_turn(holder, conn, sid, p, op))
           for name in ('turn.acquire', 'turn.renew')},
    }
    return apply_worker_receipt(holder, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                generation=generation, sequence=sequence, operation=operation,
                                payload=payload, handlers=handlers)


def apply_worker_receipt(holder, *, epoch, execution_id, session_id, generation, sequence,
                         operation, payload, handlers):
    """The one worker receipt transaction every ``worker.persist`` family shares.

    An exact (execution, sequence, digest) retry returns the stored result; otherwise the
    epoch, assignment, live status and next sequence are fenced, ``handlers[operation]`` runs on
    the transaction connection (it must not commit), and its result is stored as the receipt.
    ``execution.finish`` is the only operation that makes the execution terminal.
    """
    from ._compression import worker_receipt_assignment
    if type(sequence) is not int or sequence < 1 or not isinstance(operation, str) or operation not in handlers:
        raise AuthorityRefusal('invalid_params')
    encoded = _json(payload)
    if len(encoded.encode('utf-8', errors='surrogatepass')) > 4 * 1024 * 1024:
        raise AuthorityRefusal('invalid_params')
    digest = admission_fingerprint(canonical_target=session_id,
                                  payload={'operation': operation, 'payload': json.loads(encoded)})
    def write(conn):
        _epoch(conn, epoch)
        row = worker_receipt_assignment(conn, execution_id, session_id, generation, sequence, digest)
        if row['owner_epoch'] != epoch:
            raise AuthorityRefusal('stale_epoch')
        old = conn.execute('SELECT * FROM worker_receipts WHERE execution_id=? AND sequence=?',
                           (execution_id, sequence)).fetchone()
        if old is not None:
            if old['payload_digest'] != digest:
                raise AuthorityRefusal('admission_conflict')
            from ._runtime_workers import retired_receipt
            if retired_receipt(old['result_json']):
                raise AuthorityRefusal('stale_generation')  # a terminal execution's read copy
            return json.loads(old['result_json'])
        if row['status'] not in ('registered', 'running'):
            raise AuthorityRefusal('stale_generation')
        if sequence != row['last_sequence'] + 1:
            raise AuthorityRefusal('invalid_params')
        # Each SQLite retry gets fresh rows; rolled-back annotations must not escape.
        changes = conn.total_changes
        result = handlers[operation](holder, conn, session_id, json.loads(encoded))
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
            from ._runtime_workers import compact_terminal_receipts
            compact_terminal_receipts(conn, execution_id)
        return result
    return holder._execute_write(write, patience_s=holder._TRANSCRIPT_WRITE_PATIENCE_S)
