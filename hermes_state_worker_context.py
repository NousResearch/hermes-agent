"""Closed constructor-context operations inside the worker receipt transaction.

Identity and lineage remain owner-reserved. This is not arbitrary SessionDB RPC.
"""
import json

from hermes_state_runtime import RuntimeStoreError


# model_config sidecars a worker may patch: every key the agent persists via
# ``patch_session_model_config`` (``_anthropic_rejected_thinking``: main's rejected-signature
# fingerprints, agent/anthropic_thinking_replay.py) or a compaction ``model_config_patch``.
SIDECAR_KEYS = frozenset({'_usage_anchor', '_proactive_prune_rearm_tokens', '_anthropic_rejected_thinking'})


def worker_context(db, conn, session_id, payload):
    if payload:
        raise RuntimeStoreError('invalid_params')
    row = conn.execute(
        'SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, '
        'COALESCE(tp.prompt, s.tool_names) AS _tool_names_resolved '
        'FROM sessions s LEFT JOIN system_prompts sp ON sp.hash=s.system_prompt_hash '
        'LEFT JOIN system_prompts tp ON tp.hash=s.tool_names WHERE s.id=?',
        (session_id,),
    ).fetchone()
    return {'session': db._session_row_dict(row)}


def worker_prompt(db, conn, session_id, payload):
    if set(payload) != {'system_prompt'} or not (
            payload['system_prompt'] is None or isinstance(payload['system_prompt'], str)):
        raise RuntimeStoreError('invalid_params')
    conn.execute('UPDATE sessions SET system_prompt_hash=?, system_prompt=NULL WHERE id=?',
                 (db._store_system_prompt(conn, payload['system_prompt']), session_id))
    db._delete_unreferenced_system_prompts(conn)
    return {'value': None}


def worker_sidecars(db, conn, session_id, payload):
    if set(payload) != {'patch'} or not isinstance(payload['patch'], dict) or set(payload['patch']) - SIDECAR_KEYS:
        raise RuntimeStoreError('invalid_params')
    merged = db._merge_model_config_json(conn, session_id, payload['patch'], on_missing='raise')
    conn.execute('UPDATE sessions SET model_config=? WHERE id=?', (merged, session_id))
    return {'value': None}


def worker_tool_names(db, conn, session_id, payload):
    """The tools[] pin, same contract as ``SessionDB.update_session_tool_names``: any JSON pin
    (``{"version", "tools"}`` today; a legacy name list) content-addressed through ``system_prompts``,
    ``None`` clears. Bounded so a worker cannot write an unbounded blob into the shared table."""
    if set(payload) != {'tool_names'}:
        raise RuntimeStoreError('invalid_params')
    pin = payload['tool_names']
    if pin is not None:
        if not isinstance(pin, (list, dict)):
            raise RuntimeStoreError('invalid_params')
        if isinstance(pin, list) and not all(isinstance(n, str) and 0 < len(n) <= 256 for n in pin):
            raise RuntimeStoreError('invalid_params')
        if isinstance(pin, dict) and (set(pin) != {'version', 'tools'} or not isinstance(pin['version'], str)
                                      or not isinstance(pin['tools'], list)):
            raise RuntimeStoreError('invalid_params')
    encoded = None if pin is None else json.dumps(pin)
    if encoded is not None and len(encoded) > 4 * 1024 * 1024:
        raise RuntimeStoreError('invalid_params')
    conn.execute('UPDATE sessions SET tool_names=? WHERE id=?',
                 (db._store_system_prompt(conn, encoded), session_id))
    db._delete_unreferenced_system_prompts(conn)
    return {'value': None}


def worker_tail_role(db, conn, session_id, payload):
    """``SessionDB.latest_conversation_role`` on the receipt connection (failed-turn boundary)."""
    if payload:
        raise RuntimeStoreError('invalid_params')
    row = conn.execute("SELECT role FROM messages WHERE session_id=? AND active=1 "
                       "AND role NOT IN ('session_meta','system') ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
    return {'value': row[0] if row else None}


def worker_row_role(db, conn, session_id, payload):
    """``SessionDB.get_message_role``: the role of an active row of THIS transcript, else None."""
    if set(payload) != {'row_id'} or type(payload['row_id']) is not int:
        raise RuntimeStoreError('invalid_params')
    row = conn.execute('SELECT role FROM messages WHERE id=? AND session_id=? AND active=1',
                       (payload['row_id'], session_id)).fetchone()
    return {'value': row[0] if row else None}
