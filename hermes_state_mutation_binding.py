"""Control-only ownership for unbound legacy history; never an execution policy."""
import json
from hermes_state_runtime import RuntimeStoreError

BINDING_PREFIX = 'gateway.history_control.v1.'


def import_history_control(conn, actor, session_ids):
    binding = json.dumps([actor.profile_id, actor.subject], separators=(',', ':'))
    for sid in session_ids:
        if conn.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone() is None:
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (BINDING_PREFIX + sid, binding))


def authorize_history(conn, actor, session_id, *, claim=False):
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    if row is None:
        return False
    # Import intentionally strips routes. Never reinterpret a cold native/API
    # transcript as imported history, even if stale import metadata exists.
    if row['session_key'] or row['chat_id'] or row['origin_json']:
        return False
    key = BINDING_PREFIX + session_id
    binding = json.dumps([actor.profile_id, actor.subject], separators=(',', ':'))
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if saved is not None:
        if saved[0] != binding:
            raise RuntimeStoreError('permission_denied')
        return True
    if row['source'] not in {'cli', 'tui', 'gui', 'import'}:
        return False
    if row['user_id'] and row['user_id'] != actor.subject:
        raise RuntimeStoreError('permission_denied')
    if 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if claim:
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, binding))
    return True
