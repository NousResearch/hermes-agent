"""``subagent.transcript``: a delegated child's stored transcript, for a viewer of its parent.

A child is a session of its own in the owning profile's store, written message by message while it
runs, but ``session.resume`` refuses it (it is not a conversation of the viewer's) and
``subagent.list`` rows are live-registry projections. This read joins the two under the parent's
authority: a live child is the one the parent's roster names (the ``subagent.list`` rule, nested
children included); a finished one is found by the ``subagent_id`` stamped on its row
(``tools/delegate_tool_sessions.py``), and only when its ancestry reaches this conversation.

The first read is the ``session.resume`` rows (``stored_history``). The opaque ``cursor`` names the
segment, its newest row and what ``transcript_cursor_state`` said of the session; while that still
holds, ``after: cursor`` reads only the rows appended since. Anything else (an in-place compaction,
a rotation, a rewind, an in-place rewrite) answers the whole transcript with ``reset``.
"""
import asyncio

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError

_FIELDS = frozenset({'session_id', 'profile', 'subagent_id', 'after'})


def _cursor(session_id, row_id, state):
    return '.'.join(map(str, (row_id, *state, session_id)))


def _held(after, session_id):
    """``(row_id, (live, inactive, revision))`` of a cursor this reply's segment issued, else None."""
    parts = after.split('.', 4) if isinstance(after, str) else []
    if len(parts) != 5 or parts[4] != session_id or not all(p.isdigit() for p in parts[:4]):
        return None
    return int(parts[0]), tuple(int(p) for p in parts[1:4])


def _read(db, session_id, after):
    """``(messages, cursor, reset)``: the rows appended after a still-valid cursor, else the whole transcript."""
    held = _held(after, session_id)
    if held is not None and db.transcript_cursor_state(session_id, held[0]) == held[1]:
        tail = db.conversation_tail(session_id, held[0])
        # Re-checked after the read: a compaction landing in between must not pass its copies as new rows.
        if db.transcript_cursor_state(session_id, held[0]) == held[1]:
            rows = [{**{k: v for k, v in row.items() if k != '_row_id'}, 'row_id': row['_row_id']} for row in tail]
            top = max([held[0], *(row['row_id'] for row in rows)])
            return rows, _cursor(session_id, top, (held[1][0] + len(rows), *held[1][1:])), False
    from gateway.session_local_recovery import stored_history
    # Read before the rows: a change landing during the read then fails the next poll's check.
    _, inactive, revision = db.transcript_cursor_state(session_id, 0)
    rows = stored_history(db, session_id)
    ids = [row['row_id'] for row in rows if isinstance(row.get('row_id'), int)]
    live = sum(1 for row in rows if not row.get('archived'))
    return rows, _cursor(session_id, max(ids, default=0), (live, inactive, revision)), True


async def subagent_transcript(connection, ref, params):
    after = params.get('after')
    if (set(params) - _FIELDS or not isinstance(ref.session_id, str) or not ref.session_id
            or not isinstance(params.get('subagent_id'), str) or not params['subagent_id']
            or not (after is None or isinstance(after, str))):
        raise RuntimeStoreError('invalid_params')
    authority = connection.authority
    # Cold adoption belongs to session.resume, never an ancillary poll.
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:read')
    from gateway.session_ancillary import _compression_lineage, _owner_objects, _require_current_route, physical_target
    live, subagent_id = authority.sessions[ref.session_id], params['subagent_id']
    with live.event_stream.lock:
        _require_current_route(authority, ref)
        _, records = _owner_objects(authority, ref, include_fallback_evicted=True)
        target = physical_target(authority, ref)
    record = next((r for r in records if r.get('subagent_id') == subagent_id), None)
    child = getattr(record.get('agent'), 'session_id', None) if record else None

    def load():
        # Off the loop, so the explicit owner scope (dispatch's is only a contextvar copy here);
        # every read goes through the owning authority's own store.
        from gateway.session_authorities import owner_scope
        from tools.delegate_tool_sessions import stored_child_session
        with owner_scope(authority):
            session_id = child if isinstance(child, str) and child else stored_child_session(
                authority.db, subagent_id, _compression_lineage(authority.db, target))
            return (session_id, *_read(authority.db, session_id, after)) if session_id else (None, [], None, True)

    session_id, rows, cursor, reset = await asyncio.to_thread(load)
    return {'subagent_id': subagent_id, 'available': session_id is not None, 'child_session_id': session_id,
            'running': record is not None, 'messages': rows, 'cursor': cursor, 'reset': reset}
