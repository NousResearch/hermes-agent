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
