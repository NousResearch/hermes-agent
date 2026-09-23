"""Closed constructor-context operations inside the worker receipt transaction.

Identity and lineage remain owner-reserved. This is not arbitrary SessionDB RPC.
"""
import json

from hermes_state_runtime import RuntimeStoreError


SIDECAR_KEYS = frozenset({'_usage_anchor', '_proactive_prune_rearm_tokens'})


def worker_context(db, conn, session_id, payload):
    if payload:
        raise RuntimeStoreError('invalid_params')
    row = conn.execute(
        'SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved '
        'FROM sessions s LEFT JOIN system_prompts sp ON sp.hash=s.system_prompt_hash WHERE s.id=?',
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
    names = payload.get('tool_names') if set(payload) == {'tool_names'} else False
    if names is False or not (names is None or (isinstance(names, list) and len(names) <= 4096
                                                and all(isinstance(n, str) and 0 < len(n) <= 256 for n in names))):
        raise RuntimeStoreError('invalid_params')
    conn.execute('UPDATE sessions SET tool_names=? WHERE id=?',
                 (None if names is None else json.dumps(names), session_id))
    return {'value': None}
