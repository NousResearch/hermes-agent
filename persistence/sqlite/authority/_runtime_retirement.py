"""Retirement reads of the deletion fence: whether an id is retired, the first usable generation of a derived id,
the passed fences a bind keeps, the mutation receipt check, the delete a mutation runs and the sweep markers."""

import json

from persistence.authority.runtime import mutation_receipt_key

from ._runtime import _text, identity_key


def retired_session(holder, session_id):
    from ..conversation._store.retirement import retired_in
    with holder._read_ctx() as conn:
        return retired_in(conn, session_id)


def first_generation(holder, derive, *, pinned=None, start=0):
    """``(n, derive(n), row)``: the first generation of a derived session id a NEW request may use, from
    *start*, with its row's ``(title, source)`` (None when no row). A user-deleted id stays retired for
    good, so a derived id (header-less chat, Bot Chat, hosted-room member, A2A context) whose session was
    deleted moves to its next generation, never recreated. *pinned* ``(principal_id, request_id)`` keeps
    an exact retry on the retired generation holding its terminal identity. Read from durable rows only,
    so a restart and every concurrent caller pick the same id."""
    from ..conversation._store.retirement import retired_in
    generation = start
    with holder._read_ctx() as conn:
        while True:
            sid = derive(generation)
            if retired_in(conn, sid):
                if pinned is not None and conn.execute('SELECT 1 FROM state_meta WHERE key=?', (
                        identity_key(pinned[0], sid, pinned[1]),)).fetchone() is not None:
                    return generation, sid, None
            else:
                return generation, sid, conn.execute('SELECT title, source FROM sessions WHERE id=?', (sid,)).fetchone()
            generation += 1


def keep_passed_fences(conn, derive, session_id):
    """Make permanent the fences a walk (``derive`` as for :func:`first_generation`) passed on its
    way to *session_id*, inside the write that binds *session_id*: that successor now holds the
    conversation, so a sweep marker aging out would send a later walk back to the passed
    generation. False when a passed generation has lost its marker since the walk (a sweep's was
    reclaimed): a walk now stops there, so *session_id* must not be bound as its successor."""
    from ..conversation._store.retirement import RETIRED_PREFIX
    generation = 0
    while (sid := derive(generation)) != session_id:
        if not conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + sid,)).fetchone():
            return False
        conn.execute("UPDATE state_meta SET value=json_remove(value, '$.expires') WHERE key=? AND json_valid(value)",
                     (RETIRED_PREFIX + sid,))
        generation += 1
    return True


def has_mutation_receipt(holder, principal_id, session_id, request_id):
    for value in (principal_id, session_id, request_id):
        _text(value)
    key, _digest = mutation_receipt_key(principal_id, session_id, request_id, operation='model', payload={},
                                        expected_revision=0)
    with holder._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key=?', (key,)).fetchone() is not None


def delete_in_transaction(holder, conn, session_id, payload, *, media_root, owner=None):
    from ._runtime_guards import require_idle, delete_targets
    from ..conversation._store.core import _delete_unreferenced_system_prompts
    from ..conversation._store.messages import _bump_conversation_generation
    from ..conversation._store.retirement import retire_sessions
    targets = delete_targets(conn, session_id, owner)
    require_idle(holder, conn, targets)
    retire_sessions(conn, targets, media_root=media_root)
    for sid in targets:
        _bump_conversation_generation(holder, conn, sid, 'session_reset')
        conn.execute('UPDATE sessions SET parent_session_id=NULL, runtime_revision=runtime_revision+1 WHERE parent_session_id=?', (sid,))
        conn.execute('DELETE FROM messages WHERE session_id=?', (sid,))
    conn.executemany('DELETE FROM sessions WHERE id=?', [(sid,) for sid in targets])
    _delete_unreferenced_system_prompts(conn)
    return set(), {'deleted_ids': targets}


def expiring_retirements(holder, session_ids):
    """The ids among *session_ids* whose own marker in *holder*'s store is a sweep's (ages out at the horizon)."""
    from ..conversation._store.retirement import RETIRED_PREFIX
    with holder._read_ctx() as conn:
        return {key[len(RETIRED_PREFIX):] for (key,) in conn.execute(
            "SELECT key FROM state_meta WHERE key IN (SELECT ? || value FROM json_each(?)) "
            "AND json_valid(value) AND json_extract(value, '$.expires')",
            (RETIRED_PREFIX, json.dumps(list(session_ids))))}
