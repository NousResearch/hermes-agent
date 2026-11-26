"""Atomic external-speech rows and receipts, owned by the current runtime epoch."""
import json
from dataclasses import asdict

from persistence.authority.external_messages import ExternalAppendOutcome, ExternalMessageReceipt
from persistence.authority.vocabulary import admission_fingerprint
from persistence.core.errors import AuthorityRefusal
from ._runtime import _epoch, _session
from ..conversation._store.compression import _compression_tip_on
from ..conversation._store.messages import _check_transcript_write_guards, _insert_message_rows, _bump_session_counters


def transcript_target(db, conn, session_id):
    from ..conversation._store.local import POLICY_PREFIX
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
    target = json.loads(saved[0])['entry']['session_id'] if saved else session_id
    return _compression_tip_on(db, conn, target)


def append_external_messages(db, *, epoch, session_id, messages, memory_interval=0):
    """Commit the batch and its identities together. The authority owns writer admission.

    A live worker keeps its current prompt snapshot; these rows enter the next turn's
    stored history. No old row is rewritten, and no foreground turn is admitted.
    Compression uses its existing watermark/tail-copy protocol for concurrent appends.
    """

    def write(conn):
        _epoch(conn, epoch)
        _session(conn, session_id)
        target = transcript_target(db, conn, session_id)
        _check_transcript_write_guards(db, conn, target, None)
        before = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user' AND active=1",
                              (target,)).fetchone()[0]
        results, inserted, users = [], [], 0
        for item in messages:
            message = asdict(item)
            digest = admission_fingerprint(canonical_target=session_id, payload=message)
            client_id = message['client_id']
            previous = None if client_id is None else conn.execute(
                'SELECT payload_digest, result_json FROM session_external_messages WHERE session_id=? AND client_id=?',
                (session_id, client_id)).fetchone()
            if previous is not None:
                if previous[0] != digest:
                    raise AuthorityRefusal('client_id_conflict', field='client_id')
                results.append(json.loads(previous[1]))
                continue
            row = {key: value for key, value in message.items() if key != 'client_id'}
            _insert_message_rows(db, conn, target, [row])
            receipt = {'id': row['_row_id'], 'message_uid': row['message_uid'], 'session_id': target,
                       'role': row['role'], 'client_id': client_id}
            if client_id is not None:
                conn.execute('INSERT INTO session_external_messages VALUES(?,?,?,?)',
                             (session_id, client_id, digest, json.dumps(receipt)))
            results.append(receipt)
            inserted.append(row)
            users += row['role'] == 'user'
        if inserted:
            _bump_session_counters(conn, target, len(inserted), 0, unit=False)
        review = bool(memory_interval > 0 and users and before // memory_interval < (before + users) // memory_interval)
        return ExternalAppendOutcome(session_id, tuple(ExternalMessageReceipt(**receipt) for receipt in results),
                                     tuple(row['_row_id'] for row in inserted), review)

    return db._execute_write(write, patience_s=db._TRANSCRIPT_WRITE_PATIENCE_S)
