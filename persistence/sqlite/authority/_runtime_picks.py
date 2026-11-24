"""Picks and the frozen-policy refresh: the deferred model or reasoning slot, a messaging chat's pick receipts, the
policy commands a mutation commits, the policy operations a session picked and the owner's refresh of an idle
session's policy."""

import json

from persistence.authority.runtime import parse_mutation
from persistence.core.errors import AuthorityRefusal

from ._runtime import _epoch, _json, replay_mutation_receipt

DEFERRED_PICK_PREFIX = 'gateway.deferred_pick.v1.'


def local_snapshot(holder, conn, session_id):
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.local_lineage import validate_local_lineage
    from ._runtime_guards import require_target_advanceable
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal('runtime_coordination_required')
    saved = json.loads(row[0])
    target = validate_local_lineage(conn, saved)
    require_target_advanceable(holder, conn, list({session_id, target}))
    revision = conn.execute('SELECT runtime_revision FROM sessions WHERE id=?', (target,)).fetchone()[0]
    return {'receipt': saved, 'target': target, 'target_revision': revision}


def validate_prepared(holder, conn, session_id, prepared):
    if not isinstance(prepared, dict) or 'snapshot' not in prepared:
        raise AuthorityRefusal('runtime_coordination_required')
    current = local_snapshot(holder, conn, session_id)
    if current != prepared['snapshot']:
        raise AuthorityRefusal('revision_conflict')
    return current


def _commit_policy(holder, conn, session_id, prepared):
    """Write the prepared policy over the snapshot it was derived from and bump the generation."""
    from ..conversation._store.local import POLICY_PREFIX
    snapshot = validate_prepared(holder, conn, session_id, prepared)
    saved = snapshot['receipt']
    saved['policy'] = prepared['policy']
    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(saved), POLICY_PREFIX + session_id))
    conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (session_id,))
    generation = conn.execute('SELECT runtime_generation FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
    return snapshot, generation


def model_in_transaction(holder, conn, session_id, payload, prepared):
    snapshot, generation = _commit_policy(holder, conn, session_id, prepared)
    model = prepared['policy']['model']
    config = json.loads(prepared['policy']['config_json'])['model']
    target = snapshot['target']
    # Keep the transcript and cached prompt byte-stable; the explicit model switch
    # is a client/provider boundary, not an implicit reload of launch defaults.
    row = conn.execute('SELECT model_config FROM sessions WHERE id=?', (target,)).fetchone()
    model_config = json.loads(row[0] or '{}')
    model_config['provider'] = config['provider']
    conn.execute('UPDATE sessions SET model=?,model_config=? WHERE id=?', (model, _json(model_config), target))
    # This selection is newer than any model pick still waiting for a turn boundary: that pick is superseded.
    # A level deferred beside it stays unless this selection carries its own, as defer_pick keeps it.
    key = DEFERRED_PICK_PREFIX + session_id
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    level = json.loads(row[0]).get('reasoning') if row is not None and 'reasoning' not in payload else None
    if level is None:
        conn.execute('DELETE FROM state_meta WHERE key=?', (key,))
    else:
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json({'reasoning': level}), key))
    return {session_id, target}, {'model': model, 'provider': config['provider'], 'execution_generation': generation,
                                  **({'effort': payload['reasoning']} if 'reasoning' in payload else {})}


def reasoning_in_transaction(holder, conn, session_id, payload, prepared):
    snapshot, generation = _commit_policy(holder, conn, session_id, prepared)
    # This level is newer than any still waiting for a turn boundary, beside or inside a deferred model pick.
    key = DEFERRED_PICK_PREFIX + session_id
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if row is not None:
        pick = json.loads(row[0])
        pick.pop('reasoning', None)
        pick.get('payload', {}).pop('reasoning', None)
        if 'application_id' in pick:
            conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(pick), key))
        else:
            conn.execute('DELETE FROM state_meta WHERE key=?', (key,))
    return {session_id, snapshot['target']}, {'effort': payload['effort'], 'execution_generation': generation}


def workspace_in_transaction(holder, conn, session_id, payload, prepared):
    """The policy (execution cwd + TERMINAL_CWD) and every row of the conversation move together:
    the sidebar groups by the row's cwd/git identity, which a move REPLACES (main's update_session_cwd
    with replace_git_meta), never merges."""
    from ..conversation._store.local import local_conversation_ids
    snapshot, generation = _commit_policy(holder, conn, session_id, prepared)
    cwd, git = prepared['policy']['cwd'], prepared['git']
    rows = list(dict.fromkeys([*local_conversation_ids(conn, session_id), snapshot['target']]))
    conn.executemany('UPDATE sessions SET cwd=?, git_branch=?, git_repo_root=?, '
                     'git_metadata_generation=COALESCE(git_metadata_generation, 0)+1 WHERE id=?',
                     [(cwd, git['branch'] or None, git['root'] or None, sid) for sid in rows])
    return {session_id, snapshot['target']}, {'cwd': cwd, 'branch': git['branch'] or None,
                                              'git_repo_root': git['root'] or None, 'execution_generation': generation}


def editor_mcp_in_transaction(holder, conn, session_id, payload, prepared):
    snapshot, generation = _commit_policy(holder, conn, session_id, prepared)
    return {session_id, snapshot['target']}, {'tools_changed': prepared['tools_changed'],
                                              'mcp_servers': prepared['mcp_servers'], 'execution_generation': generation}


def edit_policy_in_transaction(holder, conn, session_id, payload, prepared):
    snapshot, generation = _commit_policy(holder, conn, session_id, prepared)
    return {session_id, snapshot['target']}, {'edit_approval_policy': payload['policy'],
                                              'execution_generation': generation}


POLICY_HANDLERS = {'model': model_in_transaction, 'reasoning': reasoning_in_transaction,
                   'workspace': workspace_in_transaction, 'editor_mcp': editor_mcp_in_transaction,
                   'edit_policy': edit_policy_in_transaction}


_RECEIPTS = "FROM state_meta WHERE key GLOB 'gateway.mutation.v1.*' AND json_valid(value) "


def picks_on(conn, session_id, before=None):
    """The policy edits (``model``, ``reasoning``, ``workspace``) a ``session.mutate`` committed for *session_id* (below
    revision *before*): their receipts (kept for exact retries, also past deletion) are the durable record of
    the user's pick. A deferred ``/model`` (``status: deferred``) is only a request until a turn boundary
    applies it under a receipt of its own, so it is no pick. A branch copied its parent's policy, picks
    included, under its own id: its local receipt's ``picks``; a branch stored before that field has its
    parent's as they stood at its branch."""
    from ..conversation._store.local import POLICY_PREFIX
    bound = '' if before is None else " AND json_extract(value, '$.result.revision')<?"
    picked = {row[0] for row in conn.execute(
        "SELECT DISTINCT json_extract(value, '$.result.operation') " + _RECEIPTS +
        "AND json_extract(value, '$.result.operation') IN ('model', 'reasoning', 'workspace') "
        "AND COALESCE(json_extract(value, '$.result.status'), '') != 'deferred' "
        "AND json_extract(value, '$.result.session_id')=?" + bound, (session_id, *(() if before is None else (before,))))}
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
    if saved is None:
        return picked
    if 'picks' in (receipt := json.loads(saved[0])):
        return picked | set(receipt['picks'])
    # The parent's branch receipt names this session and the parent's revision the branch committed at.
    branched = conn.execute(
        "SELECT json_extract(value, '$.result.session_id'), json_extract(value, '$.result.revision') " + _RECEIPTS +
        "AND json_extract(value, '$.result.operation')='branch' "
        "AND json_extract(value, '$.result.branched_session_id')=?", (session_id,)).fetchone()
    return picked | picks_on(conn, *branched) if branched else picked


def policy_picks(holder, session_id):
    with holder._read_ctx() as conn:
        return picks_on(conn, session_id)


def commit_refreshed_policy(holder, *, epoch, session_id, old, new, generation):
    """The owner's config refresh of an idle session: *new* replaces exactly *old* at *generation* (a
    concurrent mutation or turn wins) and opens the next policy generation. Returns that generation."""
    from ._runtime_guards import require_idle

    def write(conn):
        _epoch(conn, epoch)
        snapshot = local_snapshot(holder, conn, session_id)
        require_idle(holder, conn, list({session_id, snapshot['target']}))
        current = conn.execute('SELECT runtime_generation FROM sessions WHERE id=?', (session_id,)).fetchone()
        if snapshot['receipt']['policy'] != old or current is None or current[0] != generation:
            raise AuthorityRefusal('revision_conflict')
        return _commit_policy(holder, conn, session_id, {'snapshot': snapshot, 'policy': new})[1]
    return holder._execute_write(write)


def _fenced_row(conn, session_id, params):
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal('not_found')
    if row['runtime_revision'] != params['expected_revision']:
        raise AuthorityRefusal('revision_conflict')
    if row['runtime_generation'] != params['expected_generation']:
        raise AuthorityRefusal('stale_generation')
    return row


def prepare_chat_pick(holder, *, epoch, key, digest, session_id, params):
    """The stored receipt of an exact retry (``{'receipt': ...}``), else the fenced mount row
    (``{'row': ...}``); a reused request id with another payload refuses ``admission_conflict``."""
    with holder._read_ctx() as conn:
        _epoch(conn, epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            return {'receipt': replay}
        return {'row': dict(_fenced_row(conn, session_id, params))}


def commit_chat_pick(holder, *, epoch, key, digest, session_id, params, result):
    """Bump the mount's revision and store *result* (plus the new revision) as the receipt, in one
    write that re-checks the fences. Returns the committed receipt result."""
    def write(conn):
        _epoch(conn, epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            return replay
        _fenced_row(conn, session_id, params)
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (session_id,))
        revision = conn.execute('SELECT runtime_revision FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
        committed = {**result, 'revision': revision}
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (key, _json({'digest': digest, 'result': committed})))
        return committed
    return holder._execute_write(write)


def defer_pick(holder, *, epoch, key, digest, session_id, operation, principal_id, request_id, payload, shown=None):
    """Record the ``model`` or ``reasoning`` pick *payload* (an authorized mutation refused only because a turn
    is running) under the receipt *key* and *digest*.

    Acceptance writes the request's durable mutation receipt in the same transaction, so an exact retry
    returns this result instead of overwriting a newer pick, and the stored pick carries a stable
    application id. A model pick's receipt names *shown* when given (the configured default a queued reset
    adopts), else the payload's model."""
    parse_mutation(operation, payload)
    pick = {'principal_id': principal_id, 'request_id': request_id, 'payload': payload,
            'application_id': 'deferred-pick:' + request_id}

    def write(conn):
        _epoch(conn, epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            return replay
        row = conn.execute('SELECT runtime_revision, runtime_generation FROM sessions WHERE id=?',
                           (session_id,)).fetchone()
        if row is None:
            raise AuthorityRefusal('not_found')
        stored = conn.execute('SELECT value FROM state_meta WHERE key=?', (DEFERRED_PICK_PREFIX + session_id,)).fetchone()
        current = json.loads(stored[0]) if stored else {}
        if operation == 'reasoning':
            current.get('payload', {}).pop('reasoning', None)  # this level is newer than the one picked with the model
            value, picked = {**current, 'reasoning': pick}, {'effort': payload['effort']}
        else:
            # A level picked with this model is newer than one deferred before it, which it drops: the
            # slot never holds both, so committing the slot's level never strips the model's own.
            older = 'reasoning' in current and 'reasoning' not in payload
            value = {**pick, **({'reasoning': current['reasoning']} if older else {})}
            picked = {'model': shown or payload['model']}
        result = {'session_id': session_id, 'operation': operation, 'status': 'deferred', **picked,
                  'revision': row[0], 'execution_generation': row[1]}
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (DEFERRED_PICK_PREFIX + session_id, _json(value)))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, _json({'digest': digest, 'result': result})))
        return result
    return holder._execute_write(write)


def deferred_pick(holder, session_id):
    """The session's deferred slot as stored, else None."""
    with holder._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (DEFERRED_PICK_PREFIX + session_id,)).fetchone()
    return row[0] if row else None


def forget_deferred_pick(holder, session_id, stored):
    """Delete the deferred slot while it still holds exactly *stored*."""
    holder._execute_write(lambda conn: conn.execute(
        'DELETE FROM state_meta WHERE key=? AND value=?', (DEFERRED_PICK_PREFIX + session_id, stored)))


def replace_deferred_pick(holder, session_id, stored, pick):
    """Store *pick* in the deferred slot while it still holds exactly *stored*; returns the text written, matched or not."""
    rest = _json(pick)
    holder._execute_write(lambda conn: conn.execute(
        'UPDATE state_meta SET value=? WHERE key=? AND value=?', (rest, DEFERRED_PICK_PREFIX + session_id, stored)))
    return rest


def mark_deferred_pick_asked(holder, session_id, stored, confirm):
    """Store the slot as asked (``awaiting_confirm``) while it still holds exactly *stored*."""
    try:
        pick = json.loads(stored)
    except (TypeError, ValueError) as exc:
        raise AuthorityRefusal('invalid_params') from exc
    if not isinstance(pick, dict):
        raise AuthorityRefusal('invalid_params')
    value = _json({**pick, 'awaiting_confirm': confirm})
    holder._execute_write(lambda conn: conn.execute(
        'UPDATE state_meta SET value=? WHERE key=? AND value=?',
        (value, DEFERRED_PICK_PREFIX + session_id, stored)))
