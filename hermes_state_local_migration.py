"""Transactional legacy binding; never a second transcript writer."""
import json
import time

from hermes_state_runtime import RuntimeStoreError, _json

LEGACY_PREFIX = 'gateway.local_legacy.v1:'


# Sources the pre-authority local CLI/TUI/GUI wrote. ``oneshot`` (finite ``chat -q`` / ``-z``) and ``tool``
# (``chat --source tool``) are CLI runs that printed ``hermes --resume <id>``: adopted for resume like
# ``cli`` and keeping their label, so human pickers still hide them (INTERNAL_LISTING_SOURCES).
# ``desktop`` is the label main's Desktop stored; it is adopted as the ``gui`` creation source.
# ``acp`` is main's in-process editor server (Zed, VS Code, JetBrains): adopted under its own source.
LEGACY_LOCAL_SOURCES = frozenset({'cli', 'tui', 'gui', 'desktop', 'oneshot', 'tool', 'acp'})
# Legacy labels that are not creation sources of their own (session_policy.SURFACES).
LEGACY_CREATION_SOURCES = {'desktop': 'gui'}


def legacy_account_label(row):
    # main's Desktop stamped the signed-in Nous account on its chats. That is a label, not a route or
    # an authenticated owner, so it does not block adoption of an otherwise unrouted Desktop row.
    user_id = row.get('user_id')
    return row['source'] == 'desktop' and isinstance(user_id, str) and user_id.startswith('nous:')


def validate_legacy_row(db, row):
    # Unrouted old local history only. Existing identity, foreign profile,
    # API, messaging, delegate and canonical rows require their own authority.
    if (row['source'] not in LEGACY_LOCAL_SOURCES
            or any(row.get(k) for k in ('session_key', 'chat_id', 'origin_json'))
            or (row.get('user_id') and not legacy_account_label(row))
            or row.get('profile_name') not in (None, '', db._own_profile_name() or 'default')):
        raise RuntimeStoreError('not_found')


# Local divergence (NousResearch/hermes-agent#134889): upstream binds only unrouted local history. With
# ``gateway.session.viewer_replies_stay_local`` a local viewer who continues a historical messaging row (one
# its lane's route has moved past) adopts it as its own local session (gateway/session_viewer_adoption.py),
# as main's in-process TUI did by resuming any row under ``platform="tui"``.
# Pre-adoption columns of a viewer-adopted row (``scripts/local_adopt_rollback.py`` restores them): the
# identity ``bind_legacy_target`` rewrites and the end stamp the first turn's ``reopen_local_session``
# clears, so a rolled-back row is history of its old lane again, not an open session.
PREIMAGE_PREFIX = 'local_adopt_preimage:'
IDENTITY_COLUMNS = ('session_key', 'chat_id', 'user_id', 'chat_type', 'origin_json', 'ended_at', 'end_reason')


def adopted_away_from(db, session_id, session_key):
    """True when *session_id* is a viewer-adopted row and *session_key* is not its local route: a
    repoint (``/resume``, handoff, branch) would hand the local session back to another lane."""
    if not isinstance(db.get_meta(PREIMAGE_PREFIX + session_id), str):
        return False
    row = db.get_session(session_id)
    return row is not None and row.get('session_key') != session_key


def validate_viewer_row(db, row):
    """A messaging row a local viewer may adopt: this profile's and keyed. That its lane's route no
    longer names it is checked by the caller against the gateway's routing index
    (``gateway/session_viewer_adoption.py``), which may live in another profile's store.

    ``validate_legacy_row``'s other refusals do not apply, deliberately: a routed messaging row with
    ``session_key``/``chat_id``/``origin_json`` IS what is adopted (its identity goes to the preimage,
    and the lane it belonged to already routes elsewhere), and ``legacy_account_label`` vets a Desktop
    account stamp, whereas this ``user_id`` is the platform sender the viewer principal replaces."""
    if (row['source'] in LEGACY_LOCAL_SOURCES or not row.get('session_key')
            or row.get('profile_name') not in (None, '', db._own_profile_name() or 'default')):
        raise RuntimeStoreError('not_found')


def _validate_bind_target(db, conn, row, receipt):
    if not receipt.get('viewer_adoption'):
        validate_legacy_row(db, row)
        return
    validate_viewer_row(db, row)
    conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)', (PREIMAGE_PREFIX + row['id'], _json({
        'columns': {column: row.get(column) for column in IDENTITY_COLUMNS},
        'adopted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'actor': receipt['principal_id']})))


def bind_legacy_target(db, conn, receipt):
    target = receipt['legacy_session_id']
    if target != receipt['session_id']:
        raise RuntimeStoreError('storage_unavailable')
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    if row is None:
        raise RuntimeStoreError('not_found')
    _validate_bind_target(db, conn, dict(row), receipt)
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
    # Binding is a read-side claim: the row keeps main's ``source`` label and its end stamp. The first
    # admitted turn reopens it (``reopen_local_session``), exactly as the classic surfaces did.
    conn.execute('UPDATE sessions SET session_key=?,chat_id=?,user_id=?,chat_type=?,origin_json=? WHERE id=?',
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
