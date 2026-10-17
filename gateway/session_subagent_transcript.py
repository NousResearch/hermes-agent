"""``subagent.transcript``: a delegated child's stored transcript, for a viewer of its parent.

A child is a session of its own in the owning profile's store, written message by message while it
runs, but ``session.resume`` refuses it (it is not a conversation of the viewer's) and
``subagent.list`` rows are live-registry projections. This read joins the two under the parent's
authority: a live child is the one the parent's roster names (the ``subagent.list`` rule, nested
children included); a finished one is found by the ``subagent_id`` stamped on its row
(``tools/delegate_tool_sessions.py``), and only when its ancestry reaches this conversation.
Rows are the ``session.resume`` rows (``stored_history``); ``after`` is a ``row_id`` cursor.
"""
import asyncio

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError

_FIELDS = frozenset({'session_id', 'profile', 'subagent_id', 'after'})


async def subagent_transcript(connection, ref, params):
    after = params.get('after', 0)
    if (set(params) - _FIELDS or not isinstance(ref.session_id, str) or not ref.session_id
            or not isinstance(params.get('subagent_id'), str) or not params['subagent_id']
            or type(after) is not int or after < 0):
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
        from gateway.session_local_recovery import stored_history
        from tools.delegate_tool_sessions import stored_child_session
        with owner_scope(authority):
            session_id = child if isinstance(child, str) and child else stored_child_session(
                authority.db, subagent_id, _compression_lineage(authority.db, target))
            if session_id is None:
                return None, []
            return session_id, stored_history(authority.db, session_id)

    session_id, rows = await asyncio.to_thread(load)
    newer = [row for row in rows if not after or (isinstance(row.get('row_id'), int) and row['row_id'] > after)]
    return {'subagent_id': subagent_id, 'available': session_id is not None, 'child_session_id': session_id,
            'running': record is not None, 'messages': newer,
            'cursor': max([after, *(row['row_id'] for row in newer if isinstance(row.get('row_id'), int))])}
