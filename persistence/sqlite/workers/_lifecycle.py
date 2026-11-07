"""Closed worker lifecycle handlers, called only inside the receipt transaction."""

import json
import time

from persistence.core.errors import AuthorityRefusal


def escape_like(text: str) -> str:
    """Escape LIKE wildcards (``%``, ``_``) so derived text matches literally; pair with ``ESCAPE '\\'``.
    ``_`` is common in branch names/titles/paths and a substring match must not silently widen."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_IDENTITY_FIELDS = frozenset({
    'user_id', 'session_key', 'chat_id', 'chat_type', 'thread_id',
    'parent_session_id', 'cwd', 'profile_name', 'git_repo_root', 'origin_json', 'display_name',
})
_CREATE_FIELDS = _IDENTITY_FIELDS | {'source', 'model', 'model_config', 'system_prompt'}
_CONSTRUCTOR_CONFIG = frozenset({'max_iterations', 'max_tokens', 'reasoning_config', 'yolo_mode'})


def worker_create(holder, conn, session_id, payload):
    if set(payload) - _CREATE_FIELDS or 'source' not in payload:
        raise AuthorityRefusal('invalid_params')
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    # A worker cannot mint a row or supply unreserved routing, lineage or cwd.
    if row is None or payload['source'] != row['source']:
        raise AuthorityRefusal('permission_denied')
    for key in _IDENTITY_FIELDS:
        value = payload.get(key)
        if value is not None and value != row[key]:
            raise AuthorityRefusal('permission_denied')
    for key in _CREATE_FIELDS - {'model_config'}:
        if payload.get(key) is not None and not isinstance(payload[key], str):
            raise AuthorityRefusal('invalid_params')
    config = payload.get('model_config')
    if config is not None and (not isinstance(config, dict) or set(config) - _CONSTRUCTOR_CONFIG):
        raise AuthorityRefusal('invalid_params')
    from ..conversation._store.worker_lifecycle import insert_session_row_in_transaction
    insert_session_row_in_transaction(holder, conn, session_id, **payload)
    if config:
        # An owner-reserved child row already holds its lineage marker, so the keep-existing upsert
        # would drop these facts; merge them beside it (keys are the constructor set only).
        conn.execute("UPDATE sessions SET model_config=json_patch(model_config,?) WHERE id=? "
                     "AND json_type(model_config,'$._delegate_from') IS NOT NULL", (json.dumps(config), session_id))
    return {'value': session_id}


def worker_end(holder, conn, session_id, payload):
    if set(payload) != {'end_reason'} or not isinstance(payload['end_reason'], str) or not payload['end_reason']:
        raise AuthorityRefusal('invalid_params')
    from ..conversation._store.sessions import _end_and_bump
    _end_and_bump(holder, conn,
        'UPDATE sessions SET ended_at=?,end_reason=? WHERE id=? AND ended_at IS NULL',
        (time.time(), payload['end_reason'], session_id), session_id, payload['end_reason'])
    return {'value': None}


def worker_lifecycle(holder, conn, session_id, payload):
    if payload:
        raise AuthorityRefusal('invalid_params')
    row = conn.execute("SELECT role,tool_calls IS NOT NULL AS has_tool_calls,finish_reason "
                       "FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
    from ..conversation._store.sessions import classify_session_status
    value = classify_session_status(role=row['role'], has_tool_calls=bool(row['has_tool_calls']),
                                    finish_reason=row['finish_reason']) if row else 'empty'
    return {'value': value}


def worker_title(holder, conn, session_id, payload):
    if (set(payload) != {'title', 'source'} or payload['source'] not in ('user', 'derived', 'llm')
            or not (payload['title'] is None or isinstance(payload['title'], str))):
        raise AuthorityRefusal('invalid_params')
    from ..conversation._store.titles import _set_session_title_in_transaction
    try:
        value = bool(_set_session_title_in_transaction(
            holder, conn, session_id, payload['title'], source=payload['source']))
    except ValueError:
        # Expected validation/conflict outcomes must not poison the retry journal.
        # Keep candidate authority in the receipt, not an in-memory worker claim.
        return {'value': False, 'error': 'Title unavailable or invalid', 'title_candidate': payload['title']}
    return {'value': value}


def worker_title_source(holder, conn, session_id, payload):
    if set(payload) != {'source'} or payload['source'] not in ('user', 'derived', 'llm'):
        raise AuthorityRefusal('invalid_params')
    return {'value': conn.execute(
        'UPDATE sessions SET title_source=? WHERE id=? AND title IS NOT NULL',
        (payload['source'], session_id)).rowcount > 0}


def worker_next_title(holder, conn, session_id, payload):
    from .._schema import _NUMBERED_TITLE_RE
    if set(payload) != {'base_title'} or not isinstance(payload['base_title'], str):
        raise AuthorityRefusal('invalid_params')
    title = conn.execute('SELECT title FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
    # Only an assigned title or a receipted title attempt authorizes dedupe.
    # Concurrent activity/usage receipts must not invalidate that title attempt.
    candidate = payload['base_title']
    attempted = conn.execute(
        "SELECT 1 FROM worker_receipts r JOIN worker_executions w ON w.execution_id=r.execution_id "
        "WHERE w.session_id=? AND w.status='running' "
        "AND json_extract(r.result_json,'$.title_candidate')=? LIMIT 1",
        (session_id, candidate)).fetchone()
    if candidate != title and attempted is None:
        raise AuthorityRefusal('permission_denied')
    match = _NUMBERED_TITLE_RE.match(candidate)
    base = match.group(1) if match else candidate
    rows = conn.execute("SELECT title FROM sessions WHERE title=? OR title LIKE ? ESCAPE '\\'",
                        (base, f'{escape_like(base)} #%')).fetchall()
    if not rows:
        return {'value': base}
    numbers = [int(m.group(2)) for m in (_NUMBERED_TITLE_RE.match(row['title']) for row in rows) if m]
    return {'value': f'{base} #{max([1, *numbers]) + 1}'}


def worker_activity(holder, conn, session_id, payload):
    import math
    from session_rules.activity import bound_activity_description, normalize_activity_provenance
    if (set(payload) != {'ts', 'description', 'provenance'}
            or any(payload[k] is not None and not isinstance(payload[k], str)
                   for k in ('description', 'provenance'))):
        raise AuthorityRefusal('invalid_params')
    when = payload['ts']
    if when is not None and (type(when) not in (float, int) or not math.isfinite(when)):
        raise AuthorityRefusal('invalid_params')
    when = time.time() if when is None else float(when)
    conn.execute('UPDATE sessions SET last_activity_at=?,last_activity_description=?,last_activity_provenance=? '
                 'WHERE id=? AND (last_activity_at IS NULL OR last_activity_at<?)',
                 (when, bound_activity_description(payload['description']),
                  normalize_activity_provenance(payload['provenance']).value, session_id, when))
    return {'value': None}


def worker_activity_clear(holder, conn, session_id, payload):
    from session_rules.activity import ActivityProvenance
    if payload:
        raise AuthorityRefusal('invalid_params')
    conn.execute('UPDATE sessions SET last_activity_description=?,last_activity_provenance=? WHERE id=?',
                 ('', ActivityProvenance.UNKNOWN.value, session_id))
    return {'value': None}


def worker_billing_route(holder, conn, session_id, payload):
    if (set(payload) != {'provider', 'base_url', 'billing_mode'}
            or any(payload[k] is not None and not isinstance(payload[k], str) for k in payload)):
        raise AuthorityRefusal('invalid_params')
    # Same contract as the usage store's update_session_billing_route: route writers never touch the
    # stored prompt; only session.prompt replaces the frozen snapshot.
    conn.execute('UPDATE sessions SET billing_provider=?,billing_base_url=?,billing_mode=COALESCE(?,billing_mode) '
                 'WHERE id=?', (payload['provider'], payload['base_url'], payload['billing_mode'], session_id))
    return {'value': None}


def worker_api_content(holder, conn, session_id, payload):
    if set(payload) != {'content', 'api_content'} or not isinstance(payload['api_content'], str):
        raise AuthorityRefusal('invalid_params')
    from ..conversation._store.messages import _encode_content
    from ..conversation._store.messages_sql import _scrub_surrogates
    value = conn.execute(
        "UPDATE messages SET api_content=? WHERE id=(SELECT id FROM messages WHERE session_id=? "
        "AND role='user' AND active=1 ORDER BY id DESC LIMIT 1) AND content IS ?",
        (_scrub_surrogates(payload['api_content']), session_id, _encode_content(payload['content']))).rowcount
    return {'value': value}


def _row_id(payload):
    row_id = payload.get('row_id')
    if type(row_id) is not int or row_id <= 0:
        raise AuthorityRefusal('invalid_params')
    return row_id


def worker_message_api_content(holder, conn, session_id, payload):
    """Row-addressed sidecar backfill, same guards as the store's ``set_message_api_content``."""
    if set(payload) != {'row_id', 'content', 'api_content'} or not isinstance(payload['api_content'], str):
        raise AuthorityRefusal('invalid_params')
    from ..conversation._store.messages import _encode_content
    from ..conversation._store.messages_sql import _scrub_surrogates
    return {'value': conn.execute(
        "UPDATE messages SET api_content=? WHERE id=? AND session_id=? AND role='user' AND active=1 AND content IS ?",
        (_scrub_surrogates(payload['api_content']), _row_id(payload), session_id,
         _encode_content(payload['content']))).rowcount}


def worker_user_content(holder, conn, session_id, payload):
    """Rewrite ONE active user row of this session, same guards as the store's ``set_user_message_content``."""
    if set(payload) - {'display_metadata'} != {'row_id', 'content'}:
        raise AuthorityRefusal('invalid_params')
    from ..conversation._store.messages import _encode_content, _encode_display_metadata
    if 'display_metadata' not in payload:
        return {'value': conn.execute(
            "UPDATE messages SET content=? WHERE id=? AND session_id=? AND role='user' AND active=1",
            (_encode_content(payload['content']), _row_id(payload), session_id)).rowcount}
    return {'value': conn.execute(
        "UPDATE messages SET content=?, api_content=NULL, display_metadata=? "
        "WHERE id=? AND session_id=? AND role='user' AND active=1",
        (_encode_content(payload['content']), _encode_display_metadata(payload['display_metadata']),
         _row_id(payload), session_id)).rowcount}


WORKER_LIFECYCLE_HANDLERS = {
    'session.title': worker_title,
    'session.title_source': worker_title_source,
    'session.next_title': worker_next_title,
    'session.activity': worker_activity,
    'session.activity_clear': worker_activity_clear,
    'session.billing_route': worker_billing_route,
    'session.api_content': worker_api_content,
    'session.message_api_content': worker_message_api_content,
    'session.user_content': worker_user_content,
    'session.create': worker_create,
    'session.end': worker_end,
    'session.lifecycle': worker_lifecycle,
}
