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


def terminal_admission(conn, admission_id):
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (ADMISSION_PREFIX + admission_id,)).fetchone()
    return json.loads(saved[0]) if saved else None


def retry_terminal_admission(db, *, epoch, principal_id, session_id, request_id, payload, intent='queue'):
    """None means a live target; a retired miss/conflict is a definitive refusal.

    This is a read-only identity lookup, not authorization to attach, restore a
    route, or write. HTTP authentication/grant checks must precede this query.
    """
    for value in (principal_id, session_id, request_id):
        _text(value)
    digest = admission_fingerprint(canonical_target=session_id,
        payload={'input': json.loads(_json(payload)), 'intent': intent})
    with db._read_ctx() as conn:
        _epoch(conn, epoch)
        from hermes_state_mutation_retirement import RETIRED_PREFIX
        if not conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + session_id,)).fetchone():
            return None
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (identity_key(principal_id, session_id, request_id),)).fetchone()
        if saved is None:
            raise RuntimeStoreError('not_found')
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


# A live session's settled result holds its transcript copies (the turn's messages, tool payloads,
# reasoning, the user row as the model saw it, a second copy of the reply) only while an exact retry
# may replay them (``RETRY_HORIZON_S``). After that a completed turn keeps its one reply
# (``final_response``), its outcome flags, failure fields and accounting, and is marked
# ``reclaimed_at``. A failed, interrupted or partial turn and a cron run stay whole: they are rare.
# The stamp key carries the expiry, so a batch is one key-range scan. A deleted session's results
# age with its tombstones.
HORIZON_PREFIX = 'gateway.admission.result_horizon.v2.'
_BACKFILL_KEY = 'gateway.admission.result_horizon.backfill.v2'
# The end of the result key range: a bounded scan, never from the prefix start (``backfill``).
_RESULT_END = RESULT_PREFIX[:-1] + chr(ord(RESULT_PREFIX[-1]) + 1)


def _horizon_key(admission_id, expires_at):
    return f'{HORIZON_PREFIX}{int(expires_at):012d}:{admission_id}'


def _reducible(value):
    result = value.get('result')
    return (isinstance(result, dict) and result.get('completed') is not False and 'cron_result' not in result
            and not any(result.get(flag) for flag in ('failed', 'interrupted', 'partial')))


def stamp_result_horizon(conn, admission_id, outcome, value):
    """In the settlement transaction: a reducible result's retry horizon starts now."""
    import time
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    if outcome == 'completed' and _reducible(value):
        conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                     (_horizon_key(admission_id, time.time() + RETRY_HORIZON_S), '1'))


def backfill_result_horizon(conn, *, now, limit):
    """Stamp up to *limit* results settled before horizon stamps existed, in key order, from *now*: an
    upgrade never reduces anything at once. Returns the rows read; fewer than *limit* means done."""
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (_BACKFILL_KEY,)).fetchone()
    after = json.loads(saved[0]) if saved else ''
    if after is None:
        return 0
    rows = conn.execute('SELECT key, value FROM state_meta WHERE key > ? AND key < ? AND json_valid(value) '
                        'ORDER BY key LIMIT ?', (RESULT_PREFIX + after, _RESULT_END, limit)).fetchall()
    for key, raw in rows:
        admission_id = key[len(RESULT_PREFIX):]
        value = json.loads(raw)
        # A result settled since the upgrade already has its own stamp; a second one only finds it reduced.
        if (conn.execute("SELECT 1 FROM session_admissions WHERE admission_id=? AND status='terminal' "
                         "AND outcome='completed'", (admission_id,)).fetchone()
                and 'reclaimed_at' not in value and _reducible(value)):
            conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                         (_horizon_key(admission_id, now + RETRY_HORIZON_S), '1'))
    cursor = rows[-1][0][len(RESULT_PREFIX):] if len(rows) == limit else None
    conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)', (_BACKFILL_KEY, json.dumps(cursor)))
    return len(rows)


def reclaim_settled_results(conn, *, now, limit):
    """Reduce the results of up to *limit* expired horizon stamps: a key-range scan that examines at
    most *limit* stamps. Returns the stamps consumed; fewer than *limit* means none are left. A result
    whose admission left the live ledger (its session was deleted) is left to its tombstone."""
    from hermes_state_mutation_retirement import _TRANSCRIPT_FIELDS
    stamps = conn.execute('SELECT key FROM state_meta WHERE key > ? AND key < ? ORDER BY key LIMIT ?',
                          (HORIZON_PREFIX, f'{HORIZON_PREFIX}{int(now):012d}', limit)).fetchall()
    for (stamp,) in stamps:
        admission_id = stamp.split(':', 1)[1]
        conn.execute('DELETE FROM state_meta WHERE key=?', (stamp,))
        if not conn.execute("SELECT 1 FROM session_admissions WHERE admission_id=? AND status='terminal' "
                            "AND outcome='completed'", (admission_id,)).fetchone():
            continue
        saved = conn.execute('SELECT value FROM state_meta WHERE key=? AND json_valid(value)',
                             (RESULT_PREFIX + admission_id,)).fetchone()
        value = json.loads(saved[0]) if saved else {}
        if 'reclaimed_at' not in value and _reducible(value):
            result = {k: v for k, v in value['result'].items() if k not in _TRANSCRIPT_FIELDS}
            conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                         (_json({**value, 'result': result, 'reclaimed_at': now}), RESULT_PREFIX + admission_id))
    return len(stamps)
