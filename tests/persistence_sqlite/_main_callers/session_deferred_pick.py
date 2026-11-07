"""Main's ``gateway/session_deferred_pick.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

import json
from hermes_state_mutation_prepared import DEFERRED_PICK_PREFIX as _PREFIX
from persistence.core.errors import AuthorityRefusal


def defer_pick(authority, actor, ref, params):
    """Record *params* (an authorized ``model`` or ``reasoning`` mutation refused only because a turn
    is running).

    Blocking SQLite: callers run it off the event loop. Acceptance writes the request's durable
    mutation receipt in the same transaction, so an exact retry returns this result instead of
    overwriting a newer pick, and the stored pick carries a stable application id."""
    from hermes_state_mutations import validate_action
    from hermes_state_runtime import _epoch, _json, mutation_receipt_key, replay_mutation_receipt
    operation = params.get('operation', 'model')
    validate_action(operation, params['payload'])
    key, digest = mutation_receipt_key(actor.subject, ref.session_id, params['request_id'], operation=operation,
                                       payload=params['payload'], expected_revision=params.get('expected_revision'),
                                       expected_generation=params.get('expected_generation'))
    pick = {'principal_id': actor.subject, 'request_id': params['request_id'], 'payload': params['payload'],
            'application_id': 'deferred-pick:' + params['request_id']}

    def write(conn):
        _epoch(conn, authority.epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            return replay
        row = conn.execute('SELECT runtime_revision, runtime_generation FROM sessions WHERE id=?',
                           (ref.session_id,)).fetchone()
        if row is None:
            raise AuthorityRefusal('not_found')
        stored = conn.execute('SELECT value FROM state_meta WHERE key=?', (_PREFIX + ref.session_id,)).fetchone()
        current = json.loads(stored[0]) if stored else {}
        if operation == 'reasoning':
            current.get('payload', {}).pop('reasoning', None)  # this level is newer than the one picked with the model
            value, picked = {**current, 'reasoning': pick}, {'effort': params['payload']['effort']}
        else:
            # A level picked with this model is newer than one deferred before it, which it drops: the
            # slot never holds both, so committing the slot's level never strips the model's own.
            older = 'reasoning' in current and 'reasoning' not in params['payload']
            value = {**pick, **({'reasoning': current['reasoning']} if older else {})}
            picked = {'model': params['payload']['model']}
        result = {'session_id': ref.session_id, 'operation': operation, 'status': 'deferred', **picked,
                  'revision': row[0], 'execution_generation': row[1]}
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (_PREFIX + ref.session_id, _json(value)))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, _json({'digest': digest, 'result': result})))
        return result
    return authority.db._execute_write(write)
