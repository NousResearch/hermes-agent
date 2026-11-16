"""Atomic record-only transcript writes and durable, session-local replay receipts."""
import json

from hermes_state_runtime import RuntimeStoreError, _epoch, _session
from hermes_state_keys import admission_fingerprint


def validate_messages(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 100:
        raise RuntimeStoreError('invalid_params', field='messages')
    result = []
    for message in messages:
        if not isinstance(message, dict) or set(message) - {'role', 'content', 'client_id', 'observed'}:
            raise RuntimeStoreError('invalid_params', field='messages')
        role, content, client_id = (message.get(key) for key in ('role', 'content', 'client_id'))
        if role not in ('user', 'assistant'):
            error = RuntimeStoreError('invalid_params', field='role')
            error.wire_code, error.wire_message = 4000, "role must be 'user' or 'assistant'"
            raise error
        if not isinstance(content, str) or not content.strip() or len(content) > 1_000_000:
            raise RuntimeStoreError('invalid_params', field='content')
        if client_id is not None and (not isinstance(client_id, str) or not client_id.strip() or len(client_id) > 1024):
            raise RuntimeStoreError('invalid_params', field='client_id')
        observed = message.get('observed', False)
        if type(observed) is not bool:
            raise RuntimeStoreError('invalid_params', field='observed')
        result.append(dict(role=role, content=content, client_id=client_id, observed=observed))
    return result


def transcript_target(db, conn, session_id):
    from hermes_state_local import POLICY_PREFIX
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
    target = json.loads(saved[0])['entry']['session_id'] if saved else session_id
    return db._compression_tip_on(conn, target)


def append_external_messages(db, *, epoch, session_id, messages, memory_interval=0):
    """Commit the batch and its identities together. The authority owns writer admission.

    A live worker keeps its current prompt snapshot; these rows enter the next turn's
    stored history. No old row is rewritten, and no foreground turn is admitted.
    Compression uses its existing watermark/tail-copy protocol for concurrent appends.
    """
    messages = validate_messages(messages)

    def write(conn):
        _epoch(conn, epoch)
        _session(conn, session_id)
        target = transcript_target(db, conn, session_id)
        db._check_transcript_write_guards(conn, target, None)
        before = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user' AND active=1",
                              (target,)).fetchone()[0]
        results, inserted, users = [], [], 0
        for message in messages:
            digest = admission_fingerprint(canonical_target=session_id, payload=message)
            client_id = message['client_id']
            previous = None if client_id is None else conn.execute(
                'SELECT payload_digest, result_json FROM session_external_messages WHERE session_id=? AND client_id=?',
                (session_id, client_id)).fetchone()
            if previous is not None:
                if previous[0] != digest:
                    raise RuntimeStoreError('client_id_conflict', field='client_id')
                results.append(json.loads(previous[1]))
                continue
            row = {key: value for key, value in message.items() if key != 'client_id'}
            db._insert_message_rows(conn, target, [row])
            receipt = {'id': row['_row_id'], 'message_uid': row['message_uid'], 'session_id': target,
                       'role': row['role'], 'client_id': client_id}
            if client_id is not None:
                conn.execute('INSERT INTO session_external_messages VALUES(?,?,?,?)',
                             (session_id, client_id, digest, json.dumps(receipt)))
            results.append(receipt)
            inserted.append(row)
            users += row['role'] == 'user'
        if inserted:
            db._bump_session_counters(conn, target, len(inserted), 0, unit=False)
        review = bool(memory_interval > 0 and users and before // memory_interval < (before + users) // memory_interval)
        return {'session_id': session_id, 'messages': results}, inserted, review

    return db._execute_write(write, patience_s=db._TRANSCRIPT_WRITE_PATIENCE_S)
