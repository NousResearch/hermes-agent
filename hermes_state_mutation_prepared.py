"""Optimistic preparation snapshots revalidated inside the receipt transaction."""
import json
from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import validate_local_lineage
from hermes_state_mutation_guards import require_idle
from hermes_state_runtime import RuntimeStoreError


def local_snapshot(db, conn, session_id):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('runtime_coordination_required')
    saved = json.loads(row[0])
    target = validate_local_lineage(conn, saved)
    require_idle(db, conn, list({session_id, target}))
    revision = conn.execute('SELECT runtime_revision FROM sessions WHERE id=?', (target,)).fetchone()[0]
    return {'receipt': saved, 'target': target, 'target_revision': revision}


def validate_prepared(db, conn, session_id, prepared):
    if not isinstance(prepared, dict) or 'snapshot' not in prepared:
        raise RuntimeStoreError('runtime_coordination_required')
    current = local_snapshot(db, conn, session_id)
    if current != prepared['snapshot']:
        raise RuntimeStoreError('revision_conflict')
    return current


def model_in_transaction(db, conn, session_id, payload, prepared):
    from hermes_state_runtime import _json
    snapshot = validate_prepared(db, conn, session_id, prepared)
    saved = snapshot['receipt']
    saved['policy'] = prepared['policy']
    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(saved), POLICY_PREFIX + session_id))
    model = prepared['policy']['model']
    config = json.loads(prepared['policy']['config_json'])['model']
    target = snapshot['target']
    # Keep the transcript and cached prompt byte-stable; the explicit model switch
    # is a client/provider boundary, not an implicit reload of launch defaults.
    row = conn.execute('SELECT model_config FROM sessions WHERE id=?', (target,)).fetchone()
    model_config = json.loads(row[0] or '{}')
    model_config['provider'] = config['provider']
    conn.execute('UPDATE sessions SET model=?,model_config=? WHERE id=?', (model, _json(model_config), target))
    conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (session_id,))
    generation = conn.execute('SELECT runtime_generation FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
    return {session_id, target}, {'model': model, 'provider': config['provider'], 'execution_generation': generation}
