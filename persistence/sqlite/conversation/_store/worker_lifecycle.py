"""The session row insert run inside a write transaction."""


import json
import time
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.conversation._store.core import _delete_unreferenced_system_prompts
from persistence.sqlite.conversation._store.sessions import _inherit_parent_session_metadata, _own_profile_name


def insert_session_row_in_transaction(
        holder, conn, session_id, source, model=None, model_config=None,
        system_prompt=None, user_id=None, session_key=None, chat_id=None,
        chat_type=None, thread_id=None, parent_session_id=None, cwd=None,
        profile_name=None, git_repo_root=None, origin_json=None, display_name=None,
        transport_profile=None):
    """Connection-taking body of the keep-existing constructor upsert.

    The ordinary constructor should delegate here too. Worker identity is
    validated before entry; this helper never opens or commits a DB.
    """
    from persistence.sqlite.conversation._store.sessions import _UPSERT_KEEP_EXISTING_SQL
    from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX
    # A delayed constructor/accounting backfill must not recreate a deleted session
    # beside its durable tombstone (the exact delete retry would then report a stale
    # success while a fresh delete collides with the existing marker).
    if (conn.execute('SELECT 1 FROM sessions WHERE id=?', (session_id,)).fetchone() is None
            and conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + session_id,)).fetchone()):
        raise AuthorityRefusal('not_found')
    if not (profile_name or "").strip():
        profile_name = _own_profile_name(holder)
    system_prompt_hash = holder._store_system_prompt(conn, system_prompt)
    conn.execute(
        """INSERT INTO sessions (
           id, source, created_source, user_id, session_key, chat_id, chat_type, thread_id,
           model, model_config, system_prompt, system_prompt_hash,
           parent_session_id, cwd, profile_name, transport_profile, git_repo_root,
           origin_json, display_name, started_at
        )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               source = CASE
                   WHEN sessions.source = 'unknown'
                   THEN COALESCE(excluded.source, 'unknown')
                   ELSE sessions.source
               END,
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
            session_id, source, source, user_id, session_key, chat_id, chat_type, thread_id, model,
            json.dumps(model_config) if model_config else None, system_prompt_hash,
            parent_session_id, cwd, profile_name, transport_profile, git_repo_root, origin_json,
            display_name, time.time(),
        ),
    )
    if system_prompt_hash is not None:
        _delete_unreferenced_system_prompts(conn)
    if parent_session_id:
        _inherit_parent_session_metadata(conn, session_id)
    # The upsert may have just filled a compression tip's created_source (update_token_counts).
    from persistence.sqlite.conversation._store.compression import stamp_compression_ancestors
    stamp_compression_ancestors(conn, session_id)
