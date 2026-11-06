"""Discover every session ID a canonical mutation can change before its receipt commits."""

import json


def _title_conflict(db, conn, title, excluded):
    title = db.sanitize_title(title)
    if title:
        row = conn.execute('SELECT id FROM sessions WHERE title=? AND id!=?', (title, excluded)).fetchone()
        if row is not None:
            return [row[0]]
    return []


def _compression_lineage(conn, session_id):
    from hermes_state_sessions import _LINEAGE_CTE_SQL
    return [row[0] for row in conn.execute(
        _LINEAGE_CTE_SQL + ' SELECT id FROM lineage', (session_id, session_id))]


def lineage_column_ids(conn, session_id):
    """The exact selector used by direct pin, archive, visibility and read markers."""
    return _compression_lineage(conn, session_id)


def direct_delete_write_ids(db, conn, session_ids, *, include_compression_chain=False):
    """Direct deletion's lineage, delegate cascade and FK-detached children."""
    from hermes_state_local import owned_lineage_ids
    from hermes_state_sessions import _collect_delegate_child_ids
    from hermes_state_common import _id_chunks, _placeholders
    requested = list(session_ids)
    existing = [row[0] for chunk in _id_chunks(requested) for row in conn.execute(
        f'SELECT id FROM sessions WHERE id IN ({_placeholders(chunk)})', chunk)]
    if include_compression_chain:
        scope = set(db._expand_compression_lineage(conn, existing))
    else:
        scope = {member for sid in existing for member in owned_lineage_ids(conn, sid)}
    doomed = scope | set(_collect_delegate_child_ids(conn, list(scope)))
    detached = {row[0] for chunk in _id_chunks(doomed) for row in conn.execute(
        f'SELECT id FROM sessions WHERE parent_session_id IN ({_placeholders(chunk)})', chunk)}
    return set(requested) | doomed | detached


def reopen_write_ids(conn, session_id):
    """Reopen's legacy reset-marker targets and auto-archive compression rows."""
    from hermes_state_common import _legacy_reset_child_sql, _sql_json_extract
    from hermes_state_sessions import _RESET_END_REASONS, _session_ids_placeholders
    ids = set(_compression_lineage(conn, session_id))
    ids.add(session_id)
    ids.update(row[0] for row in conn.execute(
        "SELECT child.id FROM sessions AS child WHERE child.parent_session_id=? "
        f"AND {_sql_json_extract('child.model_config', '$._reset_from')} IS NULL "
        f"AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL "
        f"AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL "
        "AND COALESCE(child.source,'') != 'tool' "
        "AND child.started_at >= (SELECT p.started_at FROM sessions p WHERE p.id=child.parent_session_id) "
        f"AND {_legacy_reset_child_sql('child', _session_ids_placeholders(_RESET_END_REASONS))}",
        (session_id, *_RESET_END_REASONS)))
    return ids


def compression_publication_ids(conn, parent_session_id, child_session_id):
    """Publication can stamp/unarchive ancestors and advance a local root's receipt."""
    ids = {parent_session_id, child_session_id, *_compression_lineage(conn, parent_session_id)}
    parent = conn.execute('SELECT chat_id FROM sessions WHERE id=?', (parent_session_id,)).fetchone()
    if parent is not None and parent['chat_id']:
        ids.add(parent['chat_id'])
    return ids


def _import_ids(db, conn, session_id, operation, payload, prepared, reserved):
    return {row['id'] for row in payload.get('sessions', ())
            if isinstance(row, dict) and isinstance(row.get('id'), str)}


def _metadata_ids(db, conn, session_id, operation, payload, prepared, reserved):
    ids = set()
    if operation == 'archive' or any(field != 'title' for field in payload):
        ids.update(_compression_lineage(conn, session_id))
    if 'title' in payload:
        ids.update(_title_conflict(db, conn, payload['title'], session_id))
    return ids


def _rename_ids(db, conn, session_id, operation, payload, prepared, reserved):
    return set(_title_conflict(db, conn, payload['title'], session_id))


def _delete_ids(db, conn, session_id, operation, payload, prepared, reserved):
    from hermes_state_mutation_guards import delete_targets
    targets = delete_targets(conn, session_id)
    return set(targets) | {row[0] for target in targets for row in conn.execute(
        'SELECT id FROM sessions WHERE parent_session_id=?', (target,))}


def _workspace_ids(db, conn, session_id, target, payload, prepared, reserved):
    from hermes_state_local import local_conversation_ids
    return set(local_conversation_ids(conn, session_id))


def _reset_ids(db, conn, session_id, target, payload, prepared, reserved):
    from hermes_state_runtime_workers import runtime_lineage
    return {*runtime_lineage(conn, session_id), reserved['child']}


def _branch_ids(db, conn, session_id, target, payload, prepared, reserved):
    from hermes_state_keys import local_identity
    from hermes_state_local import POLICY_PREFIX
    row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                       (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        return set()
    saved = json.loads(row[0])
    request_id = prepared['request_id'] if prepared else reserved['request_id']
    child = local_identity(saved['profile_id'], saved['principal_id'], request_id)
    ids = {child}
    parent = conn.execute('SELECT title FROM sessions WHERE id=?', (target,)).fetchone()
    if parent is not None:
        from hermes_state_titles import lineage_title_on_conn
        title = payload.get('title') or lineage_title_on_conn(conn, parent['title'] or 'branch')
        ids.update(_title_conflict(db, conn, title, child))
    return ids


def _compress_ids(db, conn, session_id, target, payload, prepared, reserved):
    return (compression_publication_ids(conn, target, reserved['child'])
            if prepared and not prepared.get('in_place') else set())


_LOCAL_EXTRAS = {'workspace': _workspace_ids, 'reset': _reset_ids,
                 'branch': _branch_ids, 'compress': _compress_ids}


def _local_ids(db, conn, session_id, operation, payload, prepared, reserved):
    from hermes_state_local_lineage import local_physical_target
    target = local_physical_target(conn, session_id)
    extra = _LOCAL_EXTRAS.get(operation)
    return {target} | (extra(db, conn, session_id, target, payload, prepared, reserved) if extra else set())


_DISCOVERY = {'import': _import_ids, 'rename': _rename_ids,
              'sidebar': _metadata_ids, 'archive': _metadata_ids, 'delete': _delete_ids,
              **dict.fromkeys(('rewind', 'reset', 'branch', 'compress', 'model', 'reasoning',
                               'workspace', 'editor_mcp', 'edit_policy'), _local_ids)}


def mutation_write_ids(db, conn, session_id, operation, payload, *, prepared=None, reserved=None):
    """Use the handlers' SQL selectors, rechecked inside the guarded writer."""
    handler = _DISCOVERY.get(operation)
    return {session_id} | (handler(db, conn, session_id, operation, payload, prepared, reserved or {})
                           if handler else set())


def worker_receipt_write_ids(conn, session_id, operation, payload):
    """Secondary IDs changed by a registered worker's receipt handler."""
    ids = {session_id}
    if operation == 'compression.publish':
        child = payload.get('child_session_id') if isinstance(payload, dict) else None
        if isinstance(child, str) and 0 < len(child) <= 1024:
            ids.update(compression_publication_ids(conn, session_id, child))
    elif operation in {'usage.main', 'usage.auxiliary'}:
        ids.update(_compression_lineage(conn, session_id))
    elif operation in {'turn.cleanup', 'compression.cleanup'} and isinstance(payload, dict):
        target = payload.get('target')
        if isinstance(target, str):
            ids.add(target)
    return ids
