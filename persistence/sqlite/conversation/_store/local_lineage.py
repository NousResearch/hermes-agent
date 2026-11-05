"""Local lineage validation and the routed target advance."""


import json
from persistence.core.errors import AuthorityRefusal
from persistence.core.canonical import canonical_json as _json
import itertools


def validate_local_lineage(conn, receipt):
    from persistence.sqlite.conversation._store.compression import _CHAIN_STEP_SQL
    from persistence.authority.bindings import local_session_id
    if ('legacy_session_id' not in receipt and
            receipt['session_id'] != local_session_id(receipt['profile_id'], receipt['principal_id'], receipt['request_id'])):
        raise AuthorityRefusal('storage_unavailable')
    from persistence.sqlite.conversation._store.local_migration import legacy_lineage_root
    root = legacy_lineage_root(conn, receipt)
    lineage = receipt.get('lineage', [root])
    if (not isinstance(lineage, list) or not lineage or len(set(lineage)) != len(lineage)
            or lineage[0] != root or lineage[-1] != receipt['entry']['session_id']):
        raise AuthorityRefusal('storage_unavailable')
    for parent_id, child_id in itertools.pairwise(lineage):
        parent = conn.execute('SELECT end_reason FROM sessions WHERE id=?', (parent_id,)).fetchone()
        child = conn.execute('SELECT parent_session_id,model_config FROM sessions WHERE id=?', (child_id,)).fetchone()
        if parent is None or child is None or child['parent_session_id'] != parent_id:
            raise AuthorityRefusal('storage_unavailable')
        if parent['end_reason'] == 'session_reset':
            valid = json.loads(child['model_config'] or '{}').get('_reset_from') == parent_id
        else:
            tip = conn.execute(_CHAIN_STEP_SQL, (parent_id,)).fetchone()
            valid = parent['end_reason'] == 'compression' and tip is not None and tip['id'] == child_id
        if not valid:
            raise AuthorityRefusal('storage_unavailable')
    if conn.execute(_CHAIN_STEP_SQL, (lineage[-1],)).fetchone() is not None:
        raise AuthorityRefusal('storage_unavailable')
    return lineage[-1]


def _routed_entry(conn, route, tip):
    """*route*'s routing entry while it still targets *tip*, else None."""
    row = conn.execute("SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=?", (route,)).fetchone()
    routed = json.loads(row[0]) if row else None
    return routed if routed is not None and routed['session_id'] == tip else None


def _commit_local_target(conn, receipt, routed):
    """Store *receipt* and its route's entry *routed* and bump the logical revision, in one transaction."""
    conn.execute("UPDATE gateway_routing SET entry_json=? WHERE scope='' AND session_key=?",
                 (_json(routed), receipt['route']))
    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(receipt), POLICY_PREFIX + receipt['session_id']))
    conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (receipt['session_id'],))


def advance_local_target(conn, parent_session_id, child_session_id, *, entry=None):
    parent = conn.execute('SELECT * FROM sessions WHERE id=?', (parent_session_id,)).fetchone()
    logical_id = parent['chat_id'] if parent else None
    if not logical_id:
        return
    key = POLICY_PREFIX + logical_id
    raw = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if raw is None:
        from persistence.sqlite.conversation._store.local_migration import LEGACY_PREFIX
        legacy = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (LEGACY_PREFIX + logical_id,)).fetchone()
        if logical_id.startswith('local-') or legacy:
            raise AuthorityRefusal('storage_unavailable')
        return
    try:
        receipt = json.loads(raw[0])
        if (receipt['session_id'] != logical_id or receipt['entry']['session_id'] != parent_session_id
                or receipt['principal_id'] != parent['user_id'] or receipt['route'] != parent['session_key']):
            raise AuthorityRefusal('storage_unavailable')
        routed = _routed_entry(conn, receipt['route'], parent_session_id)
        if routed is None:
            raise AuthorityRefusal('admission_conflict')
        candidate = dict(entry or routed, session_id=child_session_id)
        receipt['entry'] = candidate
        receipt['lineage'] = [*receipt.get('lineage', [logical_id]), child_session_id]
        validate_local_lineage(conn, receipt)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AuthorityRefusal):
            raise
        raise AuthorityRefusal('storage_unavailable') from exc
    _commit_local_target(conn, receipt, candidate)


# Sibling store modules import this module's copies in turn, so their copies are imported after these.
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
