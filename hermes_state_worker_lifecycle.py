"""Closed worker lifecycle handlers, called only inside the receipt transaction."""
import json
import time

from hermes_state_runtime import RuntimeStoreError


_IDENTITY_FIELDS = frozenset({
    'user_id', 'session_key', 'chat_id', 'chat_type', 'thread_id',
    'parent_session_id', 'cwd', 'profile_name', 'git_repo_root', 'origin_json', 'display_name',
})
_CREATE_FIELDS = _IDENTITY_FIELDS | {'source', 'model', 'model_config', 'system_prompt'}
_CONSTRUCTOR_CONFIG = frozenset({'max_iterations', 'max_tokens', 'reasoning_config', 'yolo_mode'})


def worker_create(db, conn, session_id, payload):
    if set(payload) - _CREATE_FIELDS or 'source' not in payload:
        raise RuntimeStoreError('invalid_params')
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    # A worker cannot mint a row or supply unreserved routing, lineage or cwd.
    if row is None or payload['source'] != row['source']:
        raise RuntimeStoreError('permission_denied')
    for key in _IDENTITY_FIELDS:
        value = payload.get(key)
        if value is not None and value != row[key]:
            raise RuntimeStoreError('permission_denied')
    for key in _CREATE_FIELDS - {'model_config'}:
        if payload.get(key) is not None and not isinstance(payload[key], str):
            raise RuntimeStoreError('invalid_params')
    config = payload.get('model_config')
    if config is not None and (not isinstance(config, dict) or set(config) - _CONSTRUCTOR_CONFIG):
        raise RuntimeStoreError('invalid_params')
    insert_session_row_in_transaction(db, conn, session_id, **payload)
    return {'value': session_id}


def worker_end(db, conn, session_id, payload):
    if set(payload) != {'end_reason'} or not isinstance(payload['end_reason'], str) or not payload['end_reason']:
        raise RuntimeStoreError('invalid_params')
    db._end_and_bump(conn,
        'UPDATE sessions SET ended_at=?,end_reason=? WHERE id=? AND ended_at IS NULL',
        (time.time(), payload['end_reason'], session_id), session_id, payload['end_reason'])
    return {'value': None}


def worker_lifecycle(db, conn, session_id, payload):
    from hermes_state_sessions import classify_session_status
    if payload:
        raise RuntimeStoreError('invalid_params')
    row = conn.execute("SELECT role,tool_calls IS NOT NULL AS has_tool_calls,finish_reason "
                       "FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
    value = classify_session_status(role=row['role'], has_tool_calls=bool(row['has_tool_calls']),
                                    finish_reason=row['finish_reason']) if row else 'empty'
    return {'value': value}


def worker_title(db, conn, session_id, payload):
    if (set(payload) != {'title', 'source'} or payload['source'] not in ('user', 'derived', 'llm')
            or not (payload['title'] is None or isinstance(payload['title'], str))):
        raise RuntimeStoreError('invalid_params')
    try:
        value = bool(db._set_session_title_in_transaction(
            conn, session_id, payload['title'], source=payload['source']))
    except ValueError:
        # Expected validation/conflict outcomes must not poison the retry journal.
        # Keep candidate authority in the receipt, not an in-memory worker claim.
        return {'value': False, 'error': 'Title unavailable or invalid', 'title_candidate': payload['title']}
    return {'value': value}


def worker_title_source(db, conn, session_id, payload):
    if set(payload) != {'source'} or payload['source'] not in ('user', 'derived', 'llm'):
        raise RuntimeStoreError('invalid_params')
    return {'value': conn.execute(
        'UPDATE sessions SET title_source=? WHERE id=? AND title IS NOT NULL',
        (payload['source'], session_id)).rowcount > 0}


def worker_next_title(db, conn, session_id, payload):
    from hermes_state_titles import _NUMBERED_TITLE_RE
    from hermes_state_common import escape_like
    if set(payload) != {'base_title'} or not isinstance(payload['base_title'], str):
        raise RuntimeStoreError('invalid_params')
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
        raise RuntimeStoreError('permission_denied')
    match = _NUMBERED_TITLE_RE.match(candidate)
    base = match.group(1) if match else candidate
    rows = conn.execute("SELECT title FROM sessions WHERE title=? OR title LIKE ? ESCAPE '\\'",
                        (base, f'{escape_like(base)} #%')).fetchall()
    if not rows:
        return {'value': base}
    numbers = [int(m.group(2)) for m in (_NUMBERED_TITLE_RE.match(row['title']) for row in rows) if m]
    return {'value': f'{base} #{max([1, *numbers]) + 1}'}


def worker_activity(db, conn, session_id, payload):
    import math
    from agent.session_activity import bound_activity_description, normalize_activity_provenance
    if (set(payload) != {'ts', 'description', 'provenance'}
            or any(payload[k] is not None and not isinstance(payload[k], str)
                   for k in ('description', 'provenance'))):
        raise RuntimeStoreError('invalid_params')
    when = payload['ts']
    if when is not None and (type(when) not in (float, int) or not math.isfinite(when)):
        raise RuntimeStoreError('invalid_params')
    when = time.time() if when is None else float(when)
    conn.execute('UPDATE sessions SET last_activity_at=?,last_activity_description=?,last_activity_provenance=? '
                 'WHERE id=? AND (last_activity_at IS NULL OR last_activity_at<?)',
                 (when, bound_activity_description(payload['description']),
                  normalize_activity_provenance(payload['provenance']).value, session_id, when))
    return {'value': None}


def worker_activity_clear(db, conn, session_id, payload):
    from agent.session_activity import ActivityProvenance
    if payload:
        raise RuntimeStoreError('invalid_params')
    conn.execute('UPDATE sessions SET last_activity_description=?,last_activity_provenance=? WHERE id=?',
                 ('', ActivityProvenance.UNKNOWN.value, session_id))
    return {'value': None}


def worker_billing_route(db, conn, session_id, payload):
    if (set(payload) != {'provider', 'base_url', 'billing_mode'}
            or any(payload[k] is not None and not isinstance(payload[k], str) for k in payload)):
        raise RuntimeStoreError('invalid_params')
    conn.execute('UPDATE sessions SET billing_provider=?,billing_base_url=?,billing_mode=COALESCE(?,billing_mode),'
                 'system_prompt=NULL,system_prompt_hash=NULL WHERE id=?',
                 (payload['provider'], payload['base_url'], payload['billing_mode'], session_id))
    db._delete_unreferenced_system_prompts(conn)
    return {'value': None}


def worker_api_content(db, conn, session_id, payload):
    from hermes_state_messages import _scrub_surrogates
    if set(payload) != {'content', 'api_content'} or not isinstance(payload['api_content'], str):
        raise RuntimeStoreError('invalid_params')
    value = conn.execute(
        "UPDATE messages SET api_content=? WHERE id=(SELECT id FROM messages WHERE session_id=? "
        "AND role='user' AND active=1 ORDER BY id DESC LIMIT 1) AND content IS ?",
        (_scrub_surrogates(payload['api_content']), session_id, db._encode_content(payload['content']))).rowcount
    return {'value': value}


WORKER_LIFECYCLE_HANDLERS = {
    'session.title': worker_title,
    'session.title_source': worker_title_source,
    'session.next_title': worker_next_title,
    'session.activity': worker_activity,
    'session.activity_clear': worker_activity_clear,
    'session.billing_route': worker_billing_route,
    'session.api_content': worker_api_content,
    'session.create': worker_create,
    'session.end': worker_end,
    'session.lifecycle': worker_lifecycle,
}


def insert_session_row_in_transaction(
        self, conn, session_id, source, model=None, model_config=None,
        system_prompt=None, user_id=None, session_key=None, chat_id=None,
        chat_type=None, thread_id=None, parent_session_id=None, cwd=None,
        profile_name=None, git_repo_root=None, origin_json=None, display_name=None):
    """Connection-taking body of the keep-existing constructor upsert.

    The ordinary constructor should delegate here too. Worker identity is
    validated before entry; this helper never opens or commits a DB.
    """
    from hermes_state_sessions import _UPSERT_KEEP_EXISTING_SQL
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    # A delayed constructor/accounting backfill must not recreate a deleted session
    # beside its durable tombstone (the exact delete retry would then report a stale
    # success while a fresh delete collides with the existing marker).
    if (conn.execute('SELECT 1 FROM sessions WHERE id=?', (session_id,)).fetchone() is None
            and conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + session_id,)).fetchone()):
        raise RuntimeStoreError('not_found')
    if not (profile_name or "").strip():
        profile_name = self._own_profile_name()
    system_prompt_hash = self._store_system_prompt(conn, system_prompt)
    conn.execute(
        """INSERT INTO sessions (
           id, source, user_id, session_key, chat_id, chat_type, thread_id,
           model, model_config, system_prompt, system_prompt_hash,
           parent_session_id, cwd, profile_name, git_repo_root,
           origin_json, display_name, started_at
        )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               model = COALESCE(sessions.model, excluded.model),
               model_config = CASE
                   WHEN excluded.model_config IS NOT NULL
                        AND json_type(
                            sessions.model_config, '$._reset_from'
                        ) IS NOT NULL
                        AND json_remove(
                            sessions.model_config, '$._reset_from'
                        ) = '{}'
                   THEN json_set(
                       excluded.model_config,
                       '$._reset_from',
                       json_extract(
                           sessions.model_config, '$._reset_from'
                       )
                   )
                   ELSE COALESCE(
                       sessions.model_config, excluded.model_config
                   )
               END,
               system_prompt_hash = COALESCE(
                   sessions.system_prompt_hash,
                   excluded.system_prompt_hash
               ),
               system_prompt = CASE
                   WHEN sessions.system_prompt_hash IS NULL
                        AND excluded.system_prompt_hash IS NOT NULL
                   THEN NULL
                   ELSE sessions.system_prompt
               END,
""" + _UPSERT_KEEP_EXISTING_SQL,
        (
            session_id, source, user_id, session_key, chat_id, chat_type, thread_id, model,
            json.dumps(model_config) if model_config else None, system_prompt_hash,
            parent_session_id, cwd, profile_name, git_repo_root, origin_json, display_name,
            time.time(),
        ),
    )
    if system_prompt_hash is not None:
        self._delete_unreferenced_system_prompts(conn)
    if parent_session_id:
        self._inherit_parent_session_metadata(conn, session_id)
