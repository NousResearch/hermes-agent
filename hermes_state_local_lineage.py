"""Local physical-target handoff inside the existing publication transaction.

The creation ID remains the admission/execution owner. Transcript rotation must
not re-key idempotency digests, queued inputs, subscribers or uncertain claims.
"""
import json
import time

from hermes_state_local import POLICY_PREFIX
from hermes_state_runtime import RuntimeStoreError, _epoch, _json
import itertools


def validate_local_lineage(conn, receipt):
    from hermes_state_compression import _CHAIN_STEP_SQL
    from hermes_state_keys import local_identity
    if ('legacy_session_id' not in receipt and
            receipt['session_id'] != local_identity(receipt['profile_id'], receipt['principal_id'], receipt['request_id'])):
        raise RuntimeStoreError('storage_unavailable')
    from hermes_state_local_migration import legacy_lineage_root
    root = legacy_lineage_root(conn, receipt)
    lineage = receipt.get('lineage', [root])
    if (not isinstance(lineage, list) or not lineage or len(set(lineage)) != len(lineage)
            or lineage[0] != root or lineage[-1] != receipt['entry']['session_id']):
        raise RuntimeStoreError('storage_unavailable')
    for parent_id, child_id in itertools.pairwise(lineage):
        parent = conn.execute('SELECT end_reason FROM sessions WHERE id=?', (parent_id,)).fetchone()
        child = conn.execute('SELECT parent_session_id,model_config FROM sessions WHERE id=?', (child_id,)).fetchone()
        if parent is None or child is None or child['parent_session_id'] != parent_id:
            raise RuntimeStoreError('storage_unavailable')
        if parent['end_reason'] == 'session_reset':
            valid = json.loads(child['model_config'] or '{}').get('_reset_from') == parent_id
        else:
            tip = conn.execute(_CHAIN_STEP_SQL, (parent_id,)).fetchone()
            valid = parent['end_reason'] == 'compression' and tip is not None and tip['id'] == child_id
        if not valid:
            raise RuntimeStoreError('storage_unavailable')
    if conn.execute(_CHAIN_STEP_SQL, (lineage[-1],)).fetchone() is not None:
        raise RuntimeStoreError('storage_unavailable')
    return lineage[-1]


def local_physical_target(conn, session_id):
    """Resolve a local owner's current transcript on the caller's transaction."""
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (POLICY_PREFIX + session_id,)).fetchone()
    if saved is None:
        return session_id
    receipt = json.loads(saved[0])
    if receipt['session_id'] != session_id:
        raise RuntimeStoreError('storage_unavailable')
    return validate_local_lineage(conn, receipt)


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
        from hermes_state_local_migration import LEGACY_PREFIX
        legacy = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (LEGACY_PREFIX + logical_id,)).fetchone()
        if logical_id.startswith('local-') or legacy:
            raise RuntimeStoreError('storage_unavailable')
        return
    try:
        receipt = json.loads(raw[0])
        if (receipt['session_id'] != logical_id or receipt['entry']['session_id'] != parent_session_id
                or receipt['principal_id'] != parent['user_id'] or receipt['route'] != parent['session_key']):
            raise RuntimeStoreError('storage_unavailable')
        routed = _routed_entry(conn, receipt['route'], parent_session_id)
        if routed is None:
            raise RuntimeStoreError('admission_conflict')
        candidate = dict(entry or routed, session_id=child_session_id)
        receipt['entry'] = candidate
        receipt['lineage'] = [*receipt.get('lineage', [logical_id]), child_session_id]
        validate_local_lineage(conn, receipt)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, RuntimeStoreError):
            raise
        raise RuntimeStoreError('storage_unavailable') from exc
    _commit_local_target(conn, receipt, candidate)


def _untracked_steps(conn, tip, lineage):
    """Every compression step past *tip*, in order; None when the chain leads back into itself.

    One ``_walk_compression_chain`` stops after ``_CHAIN_CAP`` steps, which is not the end of the
    chain: walk on from the last step until a walk finds nothing more. Each batch only adds ids never
    seen, so the walk ends; a step back into the lineage or an earlier batch is a cycle.
    """
    from hermes_state_compression import _CHAIN_STEP_SQL, _walk_compression_chain
    steps, seen, current = [], {*lineage, tip}, tip
    while True:
        batch = _walk_compression_chain(current, lambda sid: conn.execute(_CHAIN_STEP_SQL, (sid,)).fetchone())[1:]
        if not batch:
            return steps
        if seen.intersection(batch):
            return None
        seen.update(batch)
        steps.extend(batch)
        current = batch[-1]


def adoptable_compression(conn, receipt):
    """``(steps, routed)``: the untracked compression steps *receipt* would adopt and its route's
    entry, else None. Read-only; ``adopt_untracked_compression`` repeats it inside its write."""
    tip = receipt['entry']['session_id']
    children = _untracked_steps(conn, tip, receipt.get('lineage', [receipt['session_id']]))
    if not children:
        return None
    identity = (receipt['route'], receipt['session_id'], receipt['principal_id'])
    for child_id in children:
        child = conn.execute('SELECT session_key,chat_id,user_id FROM sessions WHERE id=?', (child_id,)).fetchone()
        if tuple(child) != identity:
            return None
    routed = _routed_entry(conn, receipt['route'], tip)
    return (children, routed) if routed is not None else None


def adopt_untracked_compression(db, *, epoch, session_id, expected):
    """Record, once, compression steps a receipt-unaware build published past the receipt's tip.

    0.22.2 (the classic line) resumes a 0.22.1 local session by id. Its compaction ends the tip with
    ``compression`` and copies the parent's route columns onto the child, but never advances the
    receipt, so ``validate_local_lineage`` refused the session forever (#140060). On this owner a
    compaction advances the receipt in its own transaction, so such a step comes only from an older
    writer. A child carrying the receipt's own route, owner and principal is that session's
    continuation: append it to the lineage and move the receipt's and the routing entry's target to
    it, nothing else. The stored receipt must still be *expected*, the copy the caller checked
    against its owner. Anything else is left for validation to refuse, unchanged.
    """
    def write(conn):
        _epoch(conn, epoch)
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
        receipt = json.loads(saved[0])
        if receipt != expected:
            return
        adoptable = adoptable_compression(conn, receipt)
        if adoptable is None:
            return
        children, routed = adoptable
        receipt['entry'] = dict(receipt['entry'], session_id=children[-1])
        receipt['lineage'] = [*receipt.get('lineage', [session_id]), *children]
        validate_local_lineage(conn, receipt)
        _commit_local_target(conn, receipt, dict(routed, session_id=children[-1]))
    db._execute_write(write)


def reset_local_target(db, *, epoch, parent_session_id, entry):
    def write(conn):
        _epoch(conn, epoch)
        parent = conn.execute('SELECT * FROM sessions WHERE id=?', (parent_session_id,)).fetchone()
        if parent is None or parent['end_reason'] == 'compression':
            raise RuntimeStoreError('admission_conflict')
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (POLICY_PREFIX + parent['chat_id'],)).fetchone()
        if saved is None:
            raise RuntimeStoreError('storage_unavailable')
        receipt = json.loads(saved[0])
        logical_id = receipt['session_id']
        if validate_local_lineage(conn, receipt) != parent_session_id:
            raise RuntimeStoreError('admission_conflict')
        from hermes_state_mutation_guards import require_not_executing
        require_not_executing(conn, list({logical_id, parent_session_id}))
        policy = receipt['policy']
        db._publish_child_session_row(conn, parent, parent_session_id=parent_session_id,
            child_session_id=entry['session_id'], source=policy['source'], model=policy['model'],
            model_config={'_reset_from': parent_session_id}, system_prompt=None,
            cwd=policy['cwd'], profile_name=parent['profile_name'])
        conn.execute("UPDATE sessions SET ended_at=?,end_reason='session_reset' WHERE id=?",
                     (time.time(), parent_session_id))
        db._bump_conversation_generation(conn, parent_session_id, 'session_reset')
        advance_local_target(conn, parent_session_id, entry['session_id'], entry=entry)
        conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (logical_id,))
    db._execute_write(write)
