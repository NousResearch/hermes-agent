"""Private terminal-only tombstone queries; never reconstruct executable history.

The authenticated edge supplies the original principal/target/request and frozen
payload. A retired target permits only that exact digest, never fresh admission.
Worker receipt reads additionally require the original private adoption proof.
"""
import hmac
import json

from hermes_state_runtime import RuntimeStoreError, _epoch, _json, _row, _secret_digest, _text, admission_fingerprint

RESULT_PREFIX = 'gateway.admission.result.v1.'
ADMISSION_PREFIX = 'gateway.terminal_admission.v1.'
IDENTITY_PREFIX = 'gateway.terminal_identity.v1.'
WORKER_PREFIX = 'gateway.terminal_worker.v1.'


def compact_result(result, *, user_message=None):
    """Retain replay/accounting and current-turn API output, never cumulative history."""
    result = dict(result)
    value = result.get('result')
    if not isinstance(value, dict):
        return result
    value = dict(value)
    # The tool schema list rides every turn result for in-process transcript logging; it is the
    # same session-wide copy each turn (tens of KB, more with MCP) and no stored-result reader uses it.
    value.pop('tools', None)
    messages = value.get('messages')
    if isinstance(messages, list):
        # Agent results include all previous turns. APIs only project this turn's
        # assistant/tool suffix; previous_response_id stores its own history. Compression
        # is the exception: its replacement snapshot is the only immutable record of the
        # rewritten prefix for this turn. A later live transcript may include a successor.
        if not value.get('_compressed') and not value.get('_messages_are_turn_suffix'):
            from agent.turn_context import reanchor_current_turn_user_idx
            anchor = value.get('current_turn_user_idx')
            if not (type(anchor) is int and 0 <= anchor < len(messages)
                    and isinstance(messages[anchor], dict) and messages[anchor].get('role') == 'user'):
                anchor = reanchor_current_turn_user_idx(messages, user_message)
            value['messages'] = messages[anchor + 1:]
            if anchor >= 0 and isinstance(messages[anchor], dict) and messages[anchor].get('api_content'):
                # The user row as the model saw it (memory prefetch, plugin context): a chained
                # Responses turn replays it instead of a row rebuilt from the raw request text.
                value['_turn_user_row'] = {k: messages[anchor][k] for k in ('role', 'content', 'api_content')}
            # Recovery/verification nudges are later user-role rows within this same turn.
            # Consumers must not infer a new boundary from them after the real anchor is gone.
            value['_messages_are_turn_suffix'] = True
            value.pop('current_turn_user_idx', None)
    result['result'] = value
    return result


def identity_key(principal_id, session_id, request_id):
    return IDENTITY_PREFIX + admission_fingerprint(canonical_target=session_id,
        payload={'principal': principal_id, 'request': request_id})


# One consumed API request id (the Idempotency-Key's admission id). It outlives the session
# row: ``session_admissions`` is deleted with the chat, and ``_one_target`` only reads that
# table. The runs store's 24h ``RETENTION_SECONDS`` is a different namespace
# (``runs_idempotency.db``) and does not hold this result. This key is pruned with the
# exact-retry tombstones (``RETRY_HORIZON_S``).
API_REQUEST_PREFIX = 'gateway.api_request.v1.'


def api_request_key(request_id):
    return API_REQUEST_PREFIX + admission_fingerprint(
        canonical_target='api', payload={'request': request_id})


# Set once a database's existing terminal-admission rows have been projected onto the pointer.
# Not a schema version: the rows already live in ``state_meta``.
API_REQUEST_BACKFILL_KEY = 'gateway.api_request.backfill.v1'


def terminal_api_admissions(conn, request_id=None):
    """API rows in ``gateway.terminal_admission.v1.*``, the tombstone the responses door already reads.

    ``request_id`` limits the scan to one Idempotency-Key. Omitting it returns every API
    terminal admission, for the one-time pointer backfill.
    """
    sql = ("SELECT value FROM state_meta WHERE key GLOB ? "
           "AND json_extract(value, '$.principal_id')='api'")
    params = [ADMISSION_PREFIX + '*']
    if request_id is not None:
        sql += " AND json_extract(value, '$.request_id')=?"
        params.append(request_id)
    return [json.loads(row[0]) for row in conn.execute(sql, params)]


def _pointer_from_terminal(row):
    pointer = {'target_session_id': row.get('target_session_id'), 'admission_id': row.get('admission_id')}
    if row.get('retired_at') is not None:
        pointer['retired_at'] = row['retired_at']
    return pointer


def consumed_api_request(conn, request_id):
    """The consumed-request pointer, or the terminal-admission row when that pointer was never written.

    Deletes from before the pointer still have ``gateway.terminal_admission.v1.*``. That is the
    same scan ``_request_admission_rows`` uses; a miss here is not a fresh request.
    """
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (api_request_key(request_id),)).fetchone()
    if saved:
        return json.loads(saved[0])
    rows = terminal_api_admissions(conn, request_id)
    if not rows:
        return None
    if len({row.get('target_session_id') for row in rows}) > 1:
        raise RuntimeStoreError('admission_conflict')
    newest = max(rows, key=_retired_rank)
    return _pointer_from_terminal(newest)


def _retired_rank(row):
    retired_at = row.get('retired_at')
    return float('-inf') if retired_at is None else float(retired_at)


def _backfill_consumed_api_requests(conn):
    """Project unexpired API terminal admissions onto the pointer. First writer wins."""
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (API_REQUEST_BACKFILL_KEY,)).fetchone():
        return 0
    import time
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    cutoff = time.time() - RETRY_HORIZON_S
    rows = [row for row in terminal_api_admissions(conn)
            if row.get('request_id') and row.get('admission_id')
            and (row.get('retired_at') is None or float(row['retired_at']) >= cutoff)]
    for row in sorted(rows, key=_retired_rank, reverse=True):
        conn.execute(
            'INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO NOTHING',
            (api_request_key(row['request_id']), _json(_pointer_from_terminal(row))))
    conn.execute(
        'INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO NOTHING',
        (API_REQUEST_BACKFILL_KEY, _json({'backfilled_at': time.time()})))
    return len(rows)


def consumed_api_requests_backfilled(db):
    with db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                            (API_REQUEST_BACKFILL_KEY,)).fetchone() is not None


def backfill_consumed_api_requests(db):
    """One write per store: existing terminal admissions gain a request-id pointer.

    ``ON CONFLICT DO NOTHING`` so an older tombstone cannot replace a key this process
    already recorded for a newer delete. A read of the marker skips the scan afterwards.
    """
    if consumed_api_requests_backfilled(db):
        return 0
    return db._execute_write(_backfill_consumed_api_requests)


def replay_consumed_api_request(conn, *, principal_id, session_id, request_id, digest):
    """The original admission when this API request id was already consumed, else None.

    The same target replays that admission when the digest matches. Any other target, or a
    changed payload, is ``admission_conflict`` and must not execute again. Other principals
    are not in this namespace.
    """
    if principal_id != 'api':
        return None
    consumed = consumed_api_request(conn, request_id)
    if consumed is None:
        return None
    if consumed.get('target_session_id') != session_id:
        raise RuntimeStoreError('admission_conflict')
    row = terminal_admission(conn, consumed.get('admission_id'))
    if row is None or row.get('payload_digest') != digest:
        raise RuntimeStoreError('admission_conflict')
    return _row(row)


def terminal_admission(conn, admission_id):
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (ADMISSION_PREFIX + admission_id,)).fetchone()
    return json.loads(saved[0]) if saved else None


def retry_terminal_admission(db, *, epoch, principal_id, session_id, request_id, payload, intent='queue'):
    """None means a live target that has not consumed this API request id.

    A retired miss/conflict is a definitive refusal. A consumed API request id still
    resolves after its session is deleted: the original target replays, any other
    target conflicts. This is a read-only identity lookup, not authorization to
    attach, restore a route, or write. HTTP authentication/grant checks must precede it.
    """
    for value in (principal_id, session_id, request_id):
        _text(value)
    digest = admission_fingerprint(canonical_target=session_id,
        payload={'input': json.loads(_json(payload)), 'intent': intent})
    with db._read_ctx() as conn:
        _epoch(conn, epoch)
        from hermes_state_mutation_retirement import RETIRED_PREFIX
        retired = conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                               (RETIRED_PREFIX + session_id,)).fetchone()
        if not retired:
            # A live admission for this session is its own replay, even when an older delete
            # already consumed the same request id (the pre-upgrade double execution). The
            # live-row path checks the digest. A session with no row still consults the tombstone.
            live = conn.execute(
                'SELECT 1 FROM session_admissions WHERE principal_id=? AND target_session_id=? AND request_id=?',
                (principal_id, session_id, request_id)).fetchone()
            if live is not None:
                return None
            return replay_consumed_api_request(
                conn, principal_id=principal_id, session_id=session_id,
                request_id=request_id, digest=digest)
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (identity_key(principal_id, session_id, request_id),)).fetchone()
        if saved is None:
            replay = replay_consumed_api_request(
                conn, principal_id=principal_id, session_id=session_id,
                request_id=request_id, digest=digest)
            if replay is None:
                raise RuntimeStoreError('not_found')
            return replay
        row = terminal_admission(conn, json.loads(saved[0]))
        if row['payload_digest'] != digest:
            raise RuntimeStoreError('admission_conflict')
        return _row(row)


def terminal_worker_receipt(db, *, execution_id, session_id, generation, sequence,
                            adoption_secret, payload_digest):
    """Exact retired producer evidence; possession never reactivates the worker."""
    proof = _secret_digest(adoption_secret)
    with db._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (WORKER_PREFIX + execution_id,)).fetchone()
        if saved is None:
            raise RuntimeStoreError('not_found')
        worker = json.loads(saved[0])
        if worker['session_id'] != session_id or not hmac.compare_digest(worker['adoption_digest'], proof):
            raise RuntimeStoreError('permission_denied')
        if type(generation) is not int or generation != worker['generation']:
            raise RuntimeStoreError('stale_generation')
        receipt = next((r for r in worker['receipts'] if r['sequence'] == sequence), None)
        if receipt is None:
            raise RuntimeStoreError('stale_generation')
        if receipt['payload_digest'] != payload_digest:
            raise RuntimeStoreError('admission_conflict')
        if receipt['result_json'] is None:
            raise RuntimeStoreError('stale_generation')
        return json.loads(receipt['result_json'])
