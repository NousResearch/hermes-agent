"""The legacy arm of a local reservation: binding an unrouted pre-authority row to its new receipt inside the
reservation's write. Its owner and a client app's label read through the caller's ``OwnerRule``, ``owner``."""

import json
import time

from persistence.core.errors import AuthorityRefusal
from session_rules.legacy_local import binding_names, client_history_row, validate_legacy_row

from ._runtime import _json


def bind_legacy_target(holder, conn, receipt, owner=None):
    target = receipt['legacy_session_id']
    if target != receipt['session_id']:
        raise AuthorityRefusal('storage_unavailable')
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    if row is None:
        raise AuthorityRefusal('not_found')
    from ..conversation._store.sessions import _own_profile_name
    validate_legacy_row(_own_profile_name(holder), dict(row), subject=receipt['principal_id'], rule=owner)
    from ._binding_history import BINDING_PREFIX
    imported = conn.execute('SELECT value FROM state_meta WHERE key=?', (BINDING_PREFIX + target,)).fetchone()
    if imported and not binding_names(imported[0], receipt['profile_id'], receipt['principal_id'], owner):
        raise AuthorityRefusal('permission_denied')
    from ..conversation._store.compression import _CHAIN_STEP_SQL
    if conn.execute(_CHAIN_STEP_SQL, (target,)).fetchone() is not None:
        raise AuthorityRefusal('admission_conflict')
    from ._runtime_guards import require_idle
    require_idle(holder, conn, [target])
    from ..conversation._store.compression import _session_turn_lease_key_on_conn
    root = _session_turn_lease_key_on_conn(holder, conn, target)
    if conn.execute('SELECT 1 FROM session_turn_leases WHERE conversation_id=? AND expires_at>?',
                    (root, time.time())).fetchone():
        raise AuthorityRefusal('runtime_coordination_required')
    from ..conversation._store.local_migration import LEGACY_PREFIX
    key = LEGACY_PREFIX + target
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (key,)).fetchone():
        raise AuthorityRefusal('storage_unavailable')
    source = receipt['entry']['origin']
    # Binding is a read-side claim: the row keeps main's ``source`` label and its end stamp. The first
    # admitted turn reopens it (``reopen_local_session``), exactly as the classic surfaces did. A client
    # app's own label is the exception: bound, the row has a route, so the label alone would read it as
    # a messaging chat; it becomes the Desktop chat it resumes as (``created_source`` keeps provenance).
    label = 'desktop' if client_history_row(dict(row), owner) else row['source']
    conn.execute('UPDATE sessions SET source=?,session_key=?,chat_id=?,user_id=?,chat_type=?,origin_json=? WHERE id=?',
                 (label, receipt['route'], receipt['session_id'], receipt['principal_id'], 'dm', _json(source), target))
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, _json({
        'session_id': receipt['session_id'], 'profile_id': receipt['profile_id'],
        'principal_id': receipt['principal_id'], 'legacy_session_id': target})))
