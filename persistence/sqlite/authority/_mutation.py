"""The receipted session mutation: one closed command and its exact-retry receipt in one write, its prepare phase, and
the command handlers that run inside that write.

The owner's checks arrive as data: ``live_guard`` is the gateway's runtime-coordination refusal, raised after the
revision and generation checks (and a delete's target walk), and ``history`` is the gateway's history-control
write, run at that point when no live guard refuses.
"""

import json
import time
from functools import partial

from persistence.authority.runtime import COMPACTED_TARGET_DETAIL, ClaimHistory, SessionGone, mutation_receipt_key, parse_mutation
from persistence.core.errors import AuthorityRefusal
from session_rules.turns import typed_prompt_content

from .._schema import TITLE_SOURCE_USER
from ._runtime import _epoch, _json, _text, replay_mutation_receipt


def authorize_write(conn, history, session_id, owner=None):
    """The history-control write: claim the session's unbound legacy history, or bind an import's new ids."""
    from ._binding_history import authorize_history, import_history_control
    if not isinstance(history, ClaimHistory):
        import_history_control(conn, history.actor, history.session_ids)
        return
    if not authorize_history(conn, history.actor, session_id, claim=True, owner=owner):
        raise AuthorityRefusal('permission_denied')


def mutate_runtime_session(holder, *, epoch: int, principal_id: str, session_id: str,
                           request_id: str, expected_revision: int,
                           operation: str, payload: dict, expected_generation: int | None = None,
                           live_guard=None, history=None, prepare_only=False, prepared=None, media_root=None,
                           owner=None):
    """Commit a closed metadata edit and its retry receipt in the same transaction.

    Returns ``(result, gone, replayed)``: the receipt result, the session-gone event the commit owes its caller
    (None on a replay) and whether an exact retry returned its stored result. With ``prepare_only`` the same checks
    run and the result is ``{'snapshot': ...}``, the local snapshot a prepared commit is derived from. Only a delete
    reads ``media_root``: the folder whose hosted documents the retirement of the deleted sessions' admissions records.
    ``owner`` is the owner rule the history write and the delete walk compare stored owners by.
    """
    for value in (principal_id, session_id, request_id):
        _text(value)
    if type(expected_revision) is not int or expected_revision < 0:
        raise AuthorityRefusal('invalid_params')
    parse_mutation(operation, payload)
    if expected_generation is not None and (type(expected_generation) is not int or expected_generation < 0):
        raise AuthorityRefusal('invalid_params')
    if operation in {'delete', 'rewind', 'reset', 'branch', 'model', 'compress'} and expected_generation is None:
        raise AuthorityRefusal('invalid_params')
    # Snapshot caller data before waiting for the writer lock.
    payload = json.loads(_json(payload))
    key, digest = mutation_receipt_key(principal_id, session_id, request_id, operation=operation, payload=payload,
                                       expected_revision=expected_revision, expected_generation=expected_generation)

    def write(conn):
        _epoch(conn, epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            return replay, None, True
        session = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
        if session is None and operation != 'import':
            raise AuthorityRefusal('not_found')
        revision = session['runtime_revision'] if session else 0
        if revision != expected_revision:
            raise AuthorityRefusal('revision_conflict')
        if expected_generation is not None and (session is None or session['runtime_generation'] != expected_generation):
            raise AuthorityRefusal('stale_generation')
        if live_guard is not None:
            # The owner's verdict refuses after the delete target walk, which may refuse first.
            if operation == 'delete':
                from ._runtime_guards import delete_targets
                delete_targets(conn, session_id, owner)
            raise live_guard.refusal()
        if history is not None:
            authorize_write(conn, history, session_id, owner)
        if prepare_only:
            from ._runtime_picks import local_snapshot
            snapshot = local_snapshot(holder, conn, session_id)
            if operation == 'branch':
                # The branch commit's own idle rule (queued input too), before the owner starts servers.
                from ._runtime_guards import require_idle
                require_idle(holder, conn, list({session_id, snapshot['target']}))
            return {'snapshot': snapshot}, None, False
        affected, projection = apply_action(holder, conn, session_id, operation, payload, prepared=prepared,
                                            media_root=media_root, owner=owner)
        gone = None
        if operation == 'delete':
            gone = SessionGone(tuple(projection.get('deleted_ids') or (session_id,)), 'delete')
        elif payload.get('archived') is True and session is not None and not session['archived']:
            gone = SessionGone(tuple(affected) or (session_id,), 'archive')
        conn.executemany('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?',
                         [(target,) for target in affected])
        updated = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
        result = {'session_id': session_id, 'revision': updated['runtime_revision'] if updated else expected_revision + 1,
                  'execution_generation': updated['runtime_generation'] if updated else expected_generation,
                  'operation': operation, **projection}
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (key, _json({'digest': digest, 'result': result})))
        return result, gone, False
    return holder._execute_write(write)


def apply_action(holder, conn, session_id, operation, payload, *, prepared=None, media_root=None, owner=None):
    if operation == 'compress':
        from ._mutation_publish import compress_in_transaction
        return compress_in_transaction(holder, conn, session_id, payload, prepared)
    if operation in {'model', 'reasoning', 'workspace', 'editor_mcp', 'edit_policy'}:
        from ._runtime_picks import POLICY_HANDLERS
        return POLICY_HANDLERS[operation](holder, conn, session_id, payload, prepared)
    if operation == 'branch':
        from ._mutation_publish import branch_in_transaction
        return branch_in_transaction(holder, conn, session_id, payload, prepared)
    from ._mutation_publish import reset_in_transaction
    handlers = {'delete': partial(_delete, media_root=media_root, owner=owner), 'import': _import, 'rewind': _rewind,
                'reset': reset_in_transaction}
    if operation in handlers:
        return handlers[operation](holder, conn, session_id, payload)
    from ..conversation._store.sessions import _set_lineage_column_in_transaction
    from ..conversation._store.titles import _set_session_title_in_transaction
    affected = set()
    result = {}
    for key, value in payload.items():
        if key == 'title':
            affected.update(_set_session_title_in_transaction(
                holder, conn, session_id, value, source=TITLE_SOURCE_USER))
            result[key] = conn.execute('SELECT title FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
        else:
            column = 'last_read_at' if key == 'unread' else key
            stored = (0.0 if value else time.time()) if key == 'unread' else int(value)
            affected.update(_set_lineage_column_in_transaction(holder, conn, column, session_id, stored))
            result[key] = value
    return affected, result


def _delete(holder, conn, session_id, payload, *, media_root, owner):
    # Terminal admission/worker rows become digest tombstones so exact retries and
    # adoption checks still resolve after the transcript is gone.
    from ._runtime_retirement import delete_in_transaction
    return delete_in_transaction(holder, conn, session_id, payload, media_root=media_root, owner=owner)


def _rewind(holder, conn, session_id, payload):
    from ._runtime_guards import require_idle
    from ..conversation._store.messages import _row_to_message_dict
    from ..conversation._store.mutation_transcript import rewind_in_transaction
    from ._local import local_physical_target
    physical = local_physical_target(conn, session_id)
    require_idle(holder, conn, list({session_id, physical}))
    # The client names the row: like a branch boundary it must be an active row of the physical
    # target. A row compaction archived (still shown, still carrying its id) would deactivate the
    # summary and every later turn, since both have larger ids. That refusal says so (``compacted:``
    # detail); a row a rewind or undo already took back, an assistant row or an unknown id does not.
    row = conn.execute('SELECT role FROM messages WHERE id=? AND session_id=? AND active=1',
                       (payload['target_message_id'], physical)).fetchone()
    if row is None or row['role'] != 'user':
        if _summarized_by_compaction(conn, physical, payload['target_message_id']):
            raise AuthorityRefusal('invalid_params', field='target_message_id', detail=COMPACTED_TARGET_DETAIL)
        raise AuthorityRefusal('invalid_params', field='target_message_id',
                                detail='target user message is no longer in session history')
    # A composite compaction carrier (summary + the next ask in one row) keeps its handoff, as
    # main's rewind does; the receipt reports the live ask, so /retry resubmits it, not the summary.
    live = _carrier_live_view(holder, conn, physical, payload['target_message_id'])
    target, ids, head, replacement = rewind_in_transaction(holder, conn, physical, payload['target_message_id'],
        preserve_compaction_handoff=live is not None or payload.get('preserve_compaction_handoff', False))
    # Receipts are JSON: publish the same public message shape get_messages() returns (decoded
    # content/tool_calls, no internal display_identity BLOB / display_order).
    target = _row_to_message_dict(holder, target, warn_context='rewind receipt', summary_flag=True)
    if live is not None:
        target['content'] = live.get('content')
    target['content'] = typed_prompt_content(target.get('content'), target)  # /retry, /undo: the typed ask, not the placeholder
    conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (session_id,))
    return {session_id, physical}, {'rewound_count': len(ids), 'target_message': target,
        'new_head_id': head, 'replacement_message_id': replacement}


def _summarized_by_compaction(conn, physical, message_id):
    """Whether *message_id* is a user turn compaction took out of *physical*'s live transcript, as
    opposed to one a rewind/undo deactivated, an assistant row, or no row at all. In-place compaction
    archives with ``active=0, compacted=1`` (``_ARCHIVE_ACTIVE_SQL``) while rewind, undo and replace
    mark ``active=0, compacted=0``; a rotating compaction leaves the turn on a compression ancestor of
    *physical*, still displayed (``active=1 OR compacted=1``)."""
    row = conn.execute('SELECT session_id, role, active, compacted FROM messages WHERE id=?',
                       (message_id,)).fetchone()
    if row is None or row['role'] != 'user':
        return False
    if row['session_id'] == physical:
        return row['active'] == 0 and row['compacted'] == 1
    if not (row['active'] == 1 or row['compacted'] == 1):
        return False
    from ..conversation._store.sessions import _expand_compression_lineage_ids
    from ..conversation._store.compression import _CHAIN_CAP
    lineage, frontier = {physical}, [physical]
    for _ in range(_CHAIN_CAP):
        frontier = [sid for sid in _expand_compression_lineage_ids(conn, frontier) if sid not in lineage]
        if not frontier:
            break
        lineage.update(frontier)
    return row['session_id'] in lineage


def _carrier_live_view(holder, conn, session_id, message_id):
    from session_rules.turns import split_user_originated_turn
    from ..conversation._store.messages import _row_to_message_dict
    row = conn.execute('SELECT * FROM messages WHERE id=? AND session_id=? AND active=1',
                       (message_id, session_id)).fetchone()
    message = row and _row_to_message_dict(holder, row, warn_context='rewind target', summary_flag=True)
    handoff, live = split_user_originated_turn(message) if message else (None, None)
    return live if handoff is not None else None


def _import(holder, conn, session_id, payload):
    from ..conversation._store.core import _IMPORT_MAX_SESSIONS
    from ..housekeeping._portability import _attach_import_parents, _import_session_row, _validate_import_payload
    rows = payload['sessions']
    if len(rows) > _IMPORT_MAX_SESSIONS:
        raise AuthorityRefusal('invalid_params')
    normalized, errors = _validate_import_payload(rows)
    if errors or not normalized or session_id != normalized[0]['session']['id']:
        raise AuthorityRefusal('invalid_params')
    imported, skipped, parents = [], [], []
    for item in normalized:
        raw = item['session']
        sid = raw['id']
        from ..conversation._store.retirement import RETIRED_PREFIX
        if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + sid,)).fetchone():
            # Refused by name at import time: a row restored beside the fence would refuse every turn.
            raise AuthorityRefusal('admission_conflict', field='sessions',
                                    detail=f'session {sid} was deleted here; a deleted session is not restored')
        if conn.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone():
            skipped.append(sid)
            continue
        _import_session_row(holder, conn, raw, item['messages'], sid)
        if raw.get('parent_session_id'):
            parents.append((sid, raw['parent_session_id']))
        imported.append(sid)
    # A parent outside this payload is kept only when it is history this importer already
    # controls: its import binding equals the child's (written by import_history_control in this
    # transaction). Linking under any other row would graft the import onto a foreign chain,
    # which a later chain-aware delete of the import would then remove.
    from ._binding_history import history_binding
    # Rows imported now, not payload ids: a payload naming an existing foreign id skips that row.
    fresh = set(imported)
    foreign = [(sid, parent) for sid, parent in parents if parent not in fresh and (
        history_binding(conn, parent) is None or history_binding(conn, parent) != history_binding(conn, sid))]
    parents = [link for link in parents if link not in foreign]
    detached = _attach_import_parents(conn, parents) + len(foreign)
    return set(imported), {'ok': True, 'imported': len(imported), 'skipped': len(skipped),
        'imported_ids': imported, 'skipped_ids': skipped, 'detached': detached, 'errors': []}
