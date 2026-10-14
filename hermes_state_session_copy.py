"""One-row transcript copies between profile stores; no runtime/profile state migration."""

import json
import re


# These describe the source transport/turn, not the copied conversation.
_OMITTED_SESSION_FIELDS = (
    "transport_profile", "session_key", "user_id", "chat_id", "chat_type", "thread_id",
    "display_name", "origin_json", "handoff_state", "handoff_platform", "handoff_error",
    "last_activity_at", "last_activity_description", "last_activity_provenance",
    "compression_failure_cooldown_until", "compression_failure_error", "compression_recovery_deadline",
)


def _rebind_session(values, target_db):
    # Legacy room identity survives marker removal in hidden + Group: title rows.
    # Clear only that shape on the copy; unrelated hidden transcripts stay hidden.
    if values.get('hidden') and str(values.get('title') or '').strip().startswith('Group:'):
        values['hidden'] = 0
    values['profile_name'] = target_db._own_profile_name()
    for key in _OMITTED_SESSION_FIELDS:
        values[key] = None
    for key in ('expiry_finalized', 'compression_fallback_streak', 'compression_ineffective_count',
                'compression_overload_streak'):
        values[key] = 0
    config = json.loads(values['model_config']) if values.get('model_config') else None
    if isinstance(config, dict):
        # Strip authorization, external-thread/cache state and source Bot Chat identity,
        # not the model/provider/reasoning choices used to continue this transcript.
        for key in ('_delegate_from', '_branched_from', '_reset_from', 'yolo_mode',
                    'codex_thread_id', '_proactive_prune_rearm_tokens', 'room_plumbing',
                    'follow_profile_config', 'composer_override_profile'):
            config.pop(key, None)
        values['model_config'] = json.dumps(config) if config else None


def copy_session_to(source_db, session_id, target_db, *, new_session_id=None, include_inactive=True):
    if not session_id:
        raise ValueError("session_id is required")
    if new_session_id is not None and not new_session_id.strip():
        raise ValueError("new_session_id must be non-empty")
    with source_db._lock:
        conn = source_db._conn
        if conn is None:
            raise RuntimeError("SessionDB is closed")
        # Pin session, messages and content-addressed payloads to one read snapshot.
        conn.execute("SAVEPOINT session_copy_snapshot")
        try:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                raise ValueError(f"Session '{session_id}' not found")
            session = dict(row)
            messages = [dict(row) for row in conn.execute(
                "SELECT * FROM messages WHERE session_id = ?" +
                ("" if include_inactive else " AND active = 1") + " ORDER BY id", (session_id,))]
            prompt = conn.execute("SELECT prompt FROM system_prompts WHERE hash = ?",
                                  (session.get("system_prompt_hash"),)).fetchone()
            prompt_payload = prompt[0] if prompt else session.get("system_prompt")
            if session.get('system_prompt_hash') and prompt_payload is None:
                raise ValueError("Missing source system prompt payload")
            tools = conn.execute("SELECT prompt FROM system_prompts WHERE hash = ?",
                                 (session.get("tool_names"),)).fetchone()
            tool_payload = tools[0] if tools else session.get("tool_names")
            if tools is None and re.fullmatch(r'[0-9a-f]{64}', tool_payload or ''):
                raise ValueError("Missing source tool pin payload")
        finally:
            conn.execute("RELEASE session_copy_snapshot")

    def _copy(conn):
        target_id = new_session_id if new_session_id is not None else session_id
        def _exists(candidate):
            return conn.execute("SELECT 1 FROM sessions WHERE id = ?", (candidate,)).fetchone() is not None
        if _exists(target_id):
            if new_session_id is not None:
                raise ValueError(f"Target session '{target_id}' already exists")
            target_id = f"{session_id}-copy"
            suffix = 2
            while _exists(target_id):
                target_id = f"{session_id}-copy-{suffix}"
                suffix += 1
        values = dict(session, id=target_id, parent_session_id=None)
        _rebind_session(values, target_db)
        if session.get('title'):
            title = f"{session['title']} (copy)"
            suffix = 2
            while conn.execute("SELECT 1 FROM sessions WHERE title = ?", (title,)).fetchone():
                title = f"{session['title']} (copy {suffix})"
                suffix += 1
            values['title'] = title
        values['system_prompt_hash'] = target_db._store_system_prompt(conn, prompt_payload)
        values['system_prompt'] = None
        values['tool_names'] = target_db._store_system_prompt(conn, tool_payload)
        columns = [r['name'] for r in conn.execute('PRAGMA table_info(sessions)') if r['name'] in values]
        conn.execute(f"INSERT INTO sessions ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                     [values[c] for c in columns])
        if messages:
            columns = [r['name'] for r in conn.execute('PRAGMA table_info(messages)')
                       if r['name'] not in ('id', 'display_order') and r['name'] in messages[0]]
            conn.executemany(f"INSERT INTO messages ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                             [[target_id if c == 'session_id' else m[c] for c in columns] for m in messages])
            # Source row IDs are not destination ordering coordinates. Reuse the canonical
            # display fold so carried generations share their first local visible row.
            target_db._reconcile_display_orders(conn, target_id)
        return target_id

    return target_db._execute_write(_copy)
