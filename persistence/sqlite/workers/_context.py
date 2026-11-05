"""Closed constructor-context operations inside the worker receipt transaction.

Identity and lineage remain owner-reserved. ``worker_context`` is the session-row read the compression handlers share;
the ``session.context`` operation it served has no producer and no handler.
"""

import json

from persistence.core.errors import AuthorityRefusal
from persistence.workers.vocabulary import SIDECAR_KEYS


def worker_context(holder, conn, session_id, payload):
    if payload:
        raise AuthorityRefusal('invalid_params')
    row = conn.execute(
        'SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, '
        'COALESCE(tp.prompt, s.tool_names) AS _tool_names_resolved '
        'FROM sessions s LEFT JOIN system_prompts sp ON sp.hash=s.system_prompt_hash '
        'LEFT JOIN system_prompts tp ON tp.hash=s.tool_names WHERE s.id=?',
        (session_id,),
    ).fetchone()
    from ..conversation._store.core import _session_row_dict
    return {'session': _session_row_dict(row)}


def worker_prompt(holder, conn, session_id, payload):
    if set(payload) != {'system_prompt'} or not (
            payload['system_prompt'] is None or isinstance(payload['system_prompt'], str)):
        raise AuthorityRefusal('invalid_params')
    conn.execute('UPDATE sessions SET system_prompt_hash=?, system_prompt=NULL WHERE id=?',
                 (holder._store_system_prompt(conn, payload['system_prompt']), session_id))
    from ..conversation._store.core import _delete_unreferenced_system_prompts
    _delete_unreferenced_system_prompts(conn)
    return {'value': None}


def worker_sidecars(holder, conn, session_id, payload):
    if set(payload) != {'patch'} or not isinstance(payload['patch'], dict) or set(payload['patch']) - SIDECAR_KEYS:
        raise AuthorityRefusal('invalid_params')
    from ..conversation._store.sessions import _merge_model_config_json
    merged = _merge_model_config_json(holder, conn, session_id, payload['patch'], on_missing='raise')
    conn.execute('UPDATE sessions SET model_config=? WHERE id=?', (merged, session_id))
    return {'value': None}


def worker_tool_names(holder, conn, session_id, payload):
    """The tools[] pin, same contract as the session store's ``update_session_tool_names``: any JSON pin
    (``{"version", "tools"[, "delegation_backend"]}`` today; a legacy name list) content-addressed through ``system_prompts``,
    ``None`` clears. Bounded so a worker cannot write an unbounded blob into the shared table."""
    if set(payload) != {'tool_names'}:
        raise AuthorityRefusal('invalid_params')
    pin = payload['tool_names']
    if pin is not None:
        if not isinstance(pin, (list, dict)):
            raise AuthorityRefusal('invalid_params')
        if isinstance(pin, list) and not all(isinstance(n, str) and 0 < len(n) <= 256 for n in pin):
            raise AuthorityRefusal('invalid_params')
        # ``delegation_backend``: the backend the pinned delegate_task was built from (agent/delegation_backends.py).
        backend = pin.get('delegation_backend', '') if isinstance(pin, dict) else ''
        if isinstance(pin, dict) and (not {'version', 'tools'} <= set(pin) <= {'version', 'tools', 'delegation_backend'}
                                      or not isinstance(pin['version'], str) or not isinstance(pin['tools'], list)
                                      or not isinstance(backend, str) or len(backend) > 64):
            raise AuthorityRefusal('invalid_params')
    encoded = None if pin is None else json.dumps(pin)
    if encoded is not None and len(encoded) > 4 * 1024 * 1024:
        raise AuthorityRefusal('invalid_params')
    conn.execute('UPDATE sessions SET tool_names=? WHERE id=?',
                 (holder._store_system_prompt(conn, encoded), session_id))
    from ..conversation._store.core import _delete_unreferenced_system_prompts
    _delete_unreferenced_system_prompts(conn)
    return {'value': None}


def worker_tail_role(holder, conn, session_id, payload):
    """The message store's ``latest_conversation_role`` on the receipt connection (failed-turn boundary)."""
    if payload:
        raise AuthorityRefusal('invalid_params')
    row = conn.execute("SELECT role FROM messages WHERE session_id=? AND active=1 "
                       "AND role NOT IN ('session_meta','system') ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
    return {'value': row[0] if row else None}


def worker_row_role(holder, conn, session_id, payload):
    """The message store's ``get_message_role``: the role of an active row of THIS transcript, else None."""
    if set(payload) != {'row_id'} or type(payload['row_id']) is not int:
        raise AuthorityRefusal('invalid_params')
    row = conn.execute('SELECT role FROM messages WHERE id=? AND session_id=? AND active=1',
                       (payload['row_id'], session_id)).fetchone()
    return {'value': row[0] if row else None}
