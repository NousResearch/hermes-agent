"""``session.active_list`` on the session authority: the live state of the turns it owns.

Desktop's silent-turn check (``session-states.ts`` ``onEventSilence``) and its sidebar poll read
this verb. Unregistered here, the WS fell back to the legacy sidecar's handler, which lists only
the sidecar's own ``_sessions`` and never an authority-owned session: a foreground tool quiet for
45 s read as a reaped turn, and Desktop painted "Hermes ended this turn without a reply" over a
turn that was still running. Rows keep ``server._session_live_item``'s shape; ``id`` and
``session_key`` are both the session id, as ``session.resume`` reports them.
"""
from __future__ import annotations

import asyncio

from gateway.session_contract import LISTING_YIELD_EVERY, SessionRef
from hermes_state_runtime import RuntimeStoreError


def _status(live, handle, pending) -> str:
    if handle.execution_state == 'running':
        prompts = live.controls.snapshot(handle.ref.session_id, handle.execution_generation)
        return 'waiting' if prompts else 'working'
    # Admitted, not yet claimed by the drain: the turn exists, its agent has not started.
    return 'starting' if any(row['status'] == 'queued' for row in pending) else 'idle'


async def active_list(connection, ref, params):
    if set(params) - {'profile', 'current_session_id'}:
        raise RuntimeStoreError('invalid_params')
    if 'session:read' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    from gateway.session_local_close import closed_and_unviewed
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES
    authority = connection.authority
    current = str(params.get('current_session_id') or '')
    rows = []
    for index, (sid, live) in enumerate(tuple(authority.sessions.items())):
        if index and not index % LISTING_YIELD_EVERY:
            # Same reason as ``session.list``: this walks every live session on the loop that also
            # serves the turns' own I/O, so a long list must give it a turn per batch.
            await asyncio.sleep(0)
        candidate = SessionRef(connection.actor.profile_id, sid)
        try:
            handle = await authority.resolve(connection.actor, candidate)
        except RuntimeStoreError as exc:
            if exc.reason not in {'permission_denied', 'profile_mismatch', 'not_found'}:
                raise
            continue
        row = authority.db.get_session_list_row(sid) or {}
        # The same population as the live ``session.list`` feed, hidden chats included: a hidden
        # chat open in a pane still needs its turn's liveness.
        if row.get('source') in INTERNAL_LISTING_SOURCES or closed_and_unviewed(authority, sid, handle):
            continue
        started = float(row.get('started_at') or 0)
        rows.append({'current': sid == current, 'id': sid, 'session_key': sid,
                     'last_active': float(row.get('ended_at') or started), 'started_at': started,
                     'message_count': int(row.get('message_count') or 0),
                     'model': str(row.get('model') or ''), 'preview': '', 'title': row.get('title') or '',
                     'source': row.get('source'),
                     'status': _status(live, handle, authority._pending_rows(candidate))})
    return {'sessions': rows}
