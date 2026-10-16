"""Optimistic preparation snapshots revalidated inside the receipt transaction."""
import json
from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import validate_local_lineage
from hermes_state_write_guards import REVIEW_LEASE_IGNORE
from hermes_state_mutation_guards import require_target_advanceable
from hermes_state_runtime import RuntimeStoreError

# A model pick made while a turn ran, committed at the next turn boundary (gateway/session_deferred_pick.py).
DEFERRED_PICK_PREFIX = 'gateway.deferred_pick.v1.'


def local_snapshot(db, conn, session_id):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('runtime_coordination_required')
    saved = json.loads(row[0])
    target = validate_local_lineage(conn, saved)
    # A compress/model rotation rebinds a background review's ownership; its snapshot stays valid.
    require_target_advanceable(db, conn, list({session_id, target}), review_lease_policy=REVIEW_LEASE_IGNORE)
    revision = conn.execute('SELECT runtime_revision FROM sessions WHERE id=?', (target,)).fetchone()[0]
    return {'receipt': saved, 'target': target, 'target_revision': revision}


def validate_prepared(db, conn, session_id, prepared):
    if not isinstance(prepared, dict) or 'snapshot' not in prepared:
        raise RuntimeStoreError('runtime_coordination_required')
    current = local_snapshot(db, conn, session_id)
    if current != prepared['snapshot']:
        raise RuntimeStoreError('revision_conflict')
    return current


def _commit_policy(db, conn, session_id, prepared):
    """Write the prepared policy over the snapshot it was derived from and bump the generation."""
    from hermes_state_runtime import _json
    snapshot = validate_prepared(db, conn, session_id, prepared)
    saved = snapshot['receipt']
    saved['policy'] = prepared['policy']
    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(saved), POLICY_PREFIX + session_id))
    conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (session_id,))
    generation = conn.execute('SELECT runtime_generation FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
    return snapshot, generation


def model_in_transaction(db, conn, session_id, payload, prepared):
    from hermes_state_runtime import _json
    snapshot, generation = _commit_policy(db, conn, session_id, prepared)
    model = prepared['policy']['model']
    config = json.loads(prepared['policy']['config_json'])['model']
    target = snapshot['target']
    # Keep the transcript and cached prompt byte-stable; the explicit model switch
    # is a client/provider boundary, not an implicit reload of launch defaults.
    row = conn.execute('SELECT model_config FROM sessions WHERE id=?', (target,)).fetchone()
    model_config = json.loads(row[0] or '{}')
    model_config['provider'] = config['provider']
    conn.execute('UPDATE sessions SET model=?,model_config=? WHERE id=?', (model, _json(model_config), target))
    # This selection is newer than any pick still waiting for a turn boundary: that pick is superseded.
    conn.execute('DELETE FROM state_meta WHERE key=?', (DEFERRED_PICK_PREFIX + session_id,))
    return {session_id, target}, {'model': model, 'provider': config['provider'], 'execution_generation': generation,
                                  **({'effort': payload['reasoning']} if 'reasoning' in payload else {})}


def reasoning_in_transaction(db, conn, session_id, payload, prepared):
    snapshot, generation = _commit_policy(db, conn, session_id, prepared)
    return {session_id, snapshot['target']}, {'effort': payload['effort'], 'execution_generation': generation}


def workspace_in_transaction(db, conn, session_id, payload, prepared):
    """The policy (execution cwd + TERMINAL_CWD) and every row of the conversation move together:
    the sidebar groups by the row's cwd/git identity, which a move REPLACES (main's update_session_cwd
    with replace_git_meta), never merges."""
    from hermes_state_local import local_conversation_ids
    snapshot, generation = _commit_policy(db, conn, session_id, prepared)
    cwd, git = prepared['policy']['cwd'], prepared['git']
    rows = list(dict.fromkeys([*local_conversation_ids(conn, session_id), snapshot['target']]))
    conn.executemany('UPDATE sessions SET cwd=?, git_branch=?, git_repo_root=?, '
                     'git_metadata_generation=COALESCE(git_metadata_generation, 0)+1 WHERE id=?',
                     [(cwd, git['branch'] or None, git['root'] or None, sid) for sid in rows])
    return {session_id, snapshot['target']}, {'cwd': cwd, 'branch': git['branch'] or None,
                                              'git_repo_root': git['root'] or None, 'execution_generation': generation}


def editor_mcp_in_transaction(db, conn, session_id, payload, prepared):
    snapshot, generation = _commit_policy(db, conn, session_id, prepared)
    return {session_id, snapshot['target']}, {'tools_changed': prepared['tools_changed'],
                                              'mcp_servers': prepared['mcp_servers'], 'execution_generation': generation}


POLICY_HANDLERS = {'model': model_in_transaction, 'reasoning': reasoning_in_transaction,
                   'workspace': workspace_in_transaction, 'editor_mcp': editor_mcp_in_transaction}
