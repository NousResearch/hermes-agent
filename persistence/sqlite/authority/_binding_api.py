"""API route bindings: the binding write, the stored binding receipt and the declared conversation key registry."""

import json
import logging
from functools import partial

from persistence.authority.bindings import api_generation_id
from persistence.core.errors import AuthorityRefusal

from ._runtime import _epoch, _json


def declared_session(holder, key):
    """The session the declared conversation key *key* is registered to, else None (the registry read only)."""
    from ..conversation._store.local import API_DECLARED_PREFIX as _DECLARED_PREFIX
    with holder._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                           (_DECLARED_PREFIX + key,)).fetchone()
    if row:
        return row[0]
    return None


def bind_api(holder, *, epoch, session_id, route, storage_source, title, declared_key, caller_key, source, entry,
             receipt, now, gate=None, walked_from=None):
    """Only the authenticated API edge may reserve an API source; never public RPC. *source* and *entry* are the
    routing origin and entry as dicts, *receipt* the binding receipt before the write stamps an adopted key,
    *now* the epoch seconds a new row starts at and *gate* the admission gate the owner decided."""
    from ..conversation._store.local import API_BINDING_PREFIX as _BINDING_PREFIX, API_DECLARED_PREFIX as _DECLARED_PREFIX
    receipt = dict(receipt)

    def write(conn):
        _epoch(conn, epoch)
        if gate is not None:
            raise AuthorityRefusal(gate.reason, field=gate.field, detail=gate.detail)
        from ..conversation._store.retirement import RETIRED_PREFIX
        from ._runtime_retirement import keep_passed_fences
        if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + session_id,)).fetchone():
            raise AuthorityRefusal('not_found')
        # The walk ran before this write: a sweep marker it passed may have been reclaimed since,
        # and binding the successor then would split the chain (the next walk stops at the root).
        if walked_from and not keep_passed_fences(conn, partial(api_generation_id, walked_from), session_id):
            raise AuthorityRefusal('chain_moved')
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (_BINDING_PREFIX + session_id,)).fetchone()
        if saved is not None:
            binding = json.loads(saved[0])
            if binding.get('storage_source', 'api_server') != storage_source:
                raise AuthorityRefusal('permission_denied')
            if declared_key and binding.get('declared_key') != declared_key:
                raise AuthorityRefusal('admission_conflict')
            return
        if declared_key:
            existing_declared = conn.execute('SELECT value FROM state_meta WHERE key=?',
                (_DECLARED_PREFIX + declared_key,)).fetchone()
            if existing_declared and existing_declared[0] != session_id:
                raise AuthorityRefusal('admission_conflict')
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO NOTHING',
                         (_DECLARED_PREFIX + declared_key, session_id))
        row = conn.execute('SELECT source,session_key,title FROM sessions WHERE id=?', (session_id,)).fetchone()
        if row is not None and row['source'] != storage_source:
            raise AuthorityRefusal('permission_denied')
        if storage_source == 'bot_room':
            if row is not None and row['title'] != title:
                raise AuthorityRefusal('admission_conflict')
            if conn.execute('SELECT 1 FROM sessions WHERE title=? AND id!=?', (title, session_id)).fetchone():
                raise AuthorityRefusal('admission_conflict')
        # Earlier API releases stored the caller-declared key directly on the session row
        # (#139835). A caller presenting that exact key adopts the row onto the canonical route;
        # the declared-key registry above already refused a key owned by another session, and any
        # other stored key (another conversation's, a foreign route) still conflicts.
        legacy_key = declared_key or caller_key
        if row is not None and row['session_key'] not in (None, '', route):
            if not legacy_key or row['session_key'] != legacy_key:
                raise AuthorityRefusal('admission_conflict')
            # The rewrite below replaces the stamped key with the route: the registry keeps it,
            # so a later request selecting the conversation by that key still finds this row.
            owner = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                 (_DECLARED_PREFIX + legacy_key,)).fetchone()
            if owner is not None and owner[0] != session_id:
                raise AuthorityRefusal('admission_conflict')
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO NOTHING',
                         (_DECLARED_PREFIX + legacy_key, session_id))
            receipt['declared_key'] = legacy_key
        existing = conn.execute("SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=?",
                                (route,)).fetchone()
        held = json.loads(existing[0])['session_id'] if existing is not None else session_id
        if held != session_id and not _heal_replacement(conn, session_id, held):
            raise AuthorityRefusal('admission_conflict')
        conn.execute('''INSERT INTO sessions(id,source,title,hidden,started_at) VALUES(?,?,?,?,?)
                        ON CONFLICT(id) DO NOTHING''', (session_id, storage_source, title,
                                                       int(storage_source == 'bot_room'), now))
        conn.execute('UPDATE sessions SET session_key=?,chat_id=?,user_id=?,chat_type=?,origin_json=? WHERE id=?',
                     (route, session_id, source['user_id'], 'dm', _json(source), session_id))
        conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?) "
                     'ON CONFLICT(scope,session_key) DO UPDATE SET entry_json=excluded.entry_json',
                     (route, _json(entry), now))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (_BINDING_PREFIX + session_id, _json(receipt)))
    holder._execute_write(write)


def _heal_replacement(conn, caller, holder):
    """The route is derived from the caller's id, so only the stale-route heal (#54878) names another
    session on it, one with no binding of its own: the caller takes the route back, used or not, and
    the replacement keeps its transcript under its own id. A holder with its own binding refuses."""
    from ..conversation._store.local import API_BINDING_PREFIX as _BINDING_PREFIX
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (_BINDING_PREFIX + holder,)).fetchone():
        return False
    logging.getLogger("gateway.session_api").warning('API route of %s taken back from heal replacement %s', caller, holder)
    return True


def api_binding_receipt(holder, session_id):
    """The stored API binding receipt, decoded, else None."""
    from ..conversation._store.local import API_BINDING_PREFIX as _BINDING_PREFIX
    with holder._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (_BINDING_PREFIX + session_id,)).fetchone()
    return json.loads(saved[0]) if saved else None
