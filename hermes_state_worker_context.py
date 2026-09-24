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
