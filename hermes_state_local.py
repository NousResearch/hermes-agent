"""Private local creation receipts in the canonical runtime transaction owner."""
import json

from hermes_state_runtime import RuntimeStoreError, _epoch, _json

POLICY_PREFIX = 'gateway.local_policy.v1:'


def commit_local_session(db, *, epoch, receipt):
    """Reserve the row, route and immutable policy together, before publication."""
    encoded = _json(receipt)
    receipt = json.loads(encoded)
    sid, route = receipt['session_id'], receipt['route']
    key = POLICY_PREFIX + sid
    def write(conn):
        _epoch(conn, epoch)
        old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if old is not None:
            try:
                saved = json.loads(old[0])
                same = all(saved[k] == receipt[k] for k in (
                    'profile_id', 'principal_id', 'request_id', 'session_id', 'route'))
                same = same and saved['policy']['request_json'] == receipt['policy']['request_json']
            except (ValueError, TypeError, KeyError) as exc:
                raise RuntimeStoreError('storage_unavailable') from exc
            if not same:
                raise RuntimeStoreError('invalid_params')
            return saved
        if ('legacy_session_id' not in receipt and
                conn.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone()):
            raise RuntimeStoreError('storage_unavailable')
        if conn.execute("SELECT 1 FROM gateway_routing WHERE scope='' AND session_key=?", (route,)).fetchone():
            raise RuntimeStoreError('admission_conflict')
        if 'legacy_session_id' in receipt:
            from hermes_state_local_migration import bind_legacy_target
            bind_legacy_target(db, conn, receipt)
        policy, entry = receipt['policy'], receipt['entry']
        from datetime import datetime
        started = datetime.fromisoformat(entry['created_at']).timestamp()
        conn.execute('''INSERT INTO sessions(id,source,user_id,session_key,chat_id,chat_type,
            model,cwd,profile_name,origin_json,started_at) VALUES(?,?,?,?,?,'dm',?,?,?,?,?)
            ON CONFLICT(id) DO NOTHING''',
            (sid, policy['source'], receipt['principal_id'], route, entry['origin']['chat_id'],
             policy['model'], policy['cwd'], db._own_profile_name(), _json(entry['origin']), started))
        conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?)",
                     (route, _json(entry), started))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, encoded))
        return receipt
    return db._execute_write(write)


def local_receipt(db, session_id):
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('storage_unavailable')
    try:
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise ValueError('invalid receipt')
        return value
    except (TypeError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc
