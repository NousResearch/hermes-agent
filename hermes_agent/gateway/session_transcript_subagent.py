"""``session.transcript`` with ``subagent_id``: a delegated child's stored transcript, for a viewer of its
parent session (ported from #140477's ``subagent.transcript``, now one read with the stored pager).

A child is a session of its own in the owning profile's store, written message by message while it runs,
but ``session.resume`` refuses it and ``subagent.list`` rows are live-registry projections. This read joins
the two under the parent's authority: a running child is the one the parent's roster names (the
``subagent.list`` rule, nested children included); a finished one is found by the ``subagent_id`` stamped
on its row (``tools/delegate_tool_sessions.py``), and only when its ancestry reaches this conversation
through delegate-child or compression-continuation hops. Anything else is ``available: false``.
"""
import asyncio

from hermes_agent.gateway.session_busy_controls import authorize
from hermes_agent.hermes_state_runtime import RuntimeStoreError


def unavailable(session_id, subagent_id, running):
    return {'session_id': session_id, 'rows': [], 'next_cursor': None, 'reset': True,
            'subagent_id': subagent_id, 'available': False, 'running': running, 'child_session_id': None}


async def subagent_transcript(connection, ref, params, binding, held, limit, inline_images):
    authority = connection.authority
    # Cold adoption belongs to session.resume, never a child poll.
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:read')
    return await asyncio.to_thread(_read, authority, authority.sessions[ref.session_id], ref, binding, held,
                                   limit, inline_images)


def _child(authority, live, ref, subagent_id):
    """``(child session id or None, running)``. In a worker thread, like ``subagent.list``: the roster is read
    under the session's event lock, which every streamed token takes too."""
    from hermes_agent.gateway.session_ancillary import (
        _compression_lineage, _owner_objects, _require_current_route, locked_projection, physical_target)
    from tools.delegate_tool_sessions import stored_child_session

    def roster():
        _require_current_route(authority, ref)
        return _owner_objects(authority, ref, include_fallback_evicted=True)[1], physical_target(authority, ref)

    records, target = locked_projection(authority, live, ref, roster)
    record = next((r for r in records if r.get('subagent_id') == subagent_id), None)
    child = getattr(record.get('agent'), 'session_id', None) if record else None
    if isinstance(child, str) and child:
        return child, True
    return stored_child_session(authority.db, subagent_id, _compression_lineage(authority.db, target)), record is not None


def _read(authority, live, ref, binding, held, limit, inline_images):
    from hermes_agent.gateway.session_authorities import owner_scope
    from hermes_agent.gateway.session_stored_reads import read_transcript

    subagent_id = binding[1]
    with owner_scope(authority):  # every store read goes through the owning authority's own store
        child, running = _child(authority, live, ref, subagent_id)
        if child is None:
            return unavailable(ref.session_id, subagent_id, running)
        db = authority.db
        page = read_transcript(db, child, db.transcript_lineage_ids(child), binding, held, limit,
                               profile_home=authority.profile_id, inline_images=inline_images)
    return {'session_id': ref.session_id, **page, 'subagent_id': subagent_id, 'available': True,
            'running': running, 'child_session_id': child}
