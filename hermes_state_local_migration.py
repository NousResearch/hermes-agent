"""Transactional legacy binding; never a second transcript writer."""
import json
import time

from hermes_state_runtime import RuntimeStoreError, _json

LEGACY_PREFIX = 'gateway.local_legacy.v1:'


def validate_legacy_row(db, row):
    # Unrouted old CLI/TUI/GUI history only. Existing identity, foreign profile,
    # API, messaging, delegate and canonical rows require their own authority.
    if (row['source'] not in {'cli', 'tui', 'gui'}
            or any(row.get(k) for k in ('session_key', 'chat_id', 'user_id', 'origin_json'))
            or row.get('profile_name') not in (None, '', db._own_profile_name() or 'default')):
        raise RuntimeStoreError('not_found')


def bind_legacy_target(db, conn, receipt):
    target = receipt['legacy_session_id']
    if target != receipt['session_id']:
        raise RuntimeStoreError('storage_unavailable')
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    validate_legacy_row(db, dict(row))
    from hermes_state_mutation_binding import BINDING_PREFIX
    imported = conn.execute('SELECT value FROM state_meta WHERE key=?', (BINDING_PREFIX + target,)).fetchone()
    if imported and json.loads(imported[0]) != [receipt['profile_id'], receipt['principal_id']]:
        raise RuntimeStoreError('permission_denied')
    from hermes_state_compression import _CHAIN_STEP_SQL
    if conn.execute(_CHAIN_STEP_SQL, (target,)).fetchone() is not None:
        raise RuntimeStoreError('admission_conflict')
    from hermes_state_mutation_guards import require_idle
    require_idle(db, conn, [target])
    root = db._session_turn_lease_key_on_conn(conn, target)
    if conn.execute('SELECT 1 FROM session_turn_leases WHERE conversation_id=? AND expires_at>?',
                    (root, time.time())).fetchone():
        raise RuntimeStoreError('runtime_coordination_required')
    key = LEGACY_PREFIX + target
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (key,)).fetchone():
        raise RuntimeStoreError('storage_unavailable')
    source = receipt['entry']['origin']
    conn.execute('UPDATE sessions SET ended_at=NULL,end_reason=NULL,session_key=?,chat_id=?,user_id=?,chat_type=?,origin_json=? WHERE id=?',
                 (receipt['route'], receipt['session_id'], receipt['principal_id'], 'dm', _json(source), target))
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, _json({
        'session_id': receipt['session_id'], 'profile_id': receipt['profile_id'],
        'principal_id': receipt['principal_id'], 'legacy_session_id': target})))


def legacy_lineage_root(conn, receipt):
    root = receipt.get('legacy_session_id', receipt['session_id'])
    if 'legacy_session_id' in receipt:
        if root != receipt['session_id']:
            raise RuntimeStoreError('storage_unavailable')
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (LEGACY_PREFIX + root,)).fetchone()
        expected = {k: receipt[k] for k in ('session_id', 'profile_id', 'principal_id', 'legacy_session_id')}
        if row is None or json.loads(row[0]) != expected:
            raise RuntimeStoreError('storage_unavailable')
    return root
