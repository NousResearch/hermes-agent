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
import json

from hermes_state_runtime import RuntimeStoreError


def _status(live, handle, pending) -> str:
    if handle.execution_state == 'running':
        from gateway.session_server_requests import open_requests
        sid = handle.ref.session_id
        # A question holds the turn: a shared control (approval) or a viewer request (clarify batch,
        # sudo, secret, vault) still open on the stream.
        asked = live.controls.snapshot(sid, handle.execution_generation) or open_requests(live.event_stream, sid)
        return 'waiting' if asked else 'working'
    # Admitted, not yet claimed by the drain: the turn exists, its agent has not started.
    return 'starting' if any(row['status'] == 'queued' for row in pending) else 'idle'


async def active_list(connection, ref, params):
    if set(params) - {'profile', 'current_session_id'}:
        raise RuntimeStoreError('invalid_params')
    if 'session:read' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    from gateway.session_live_listing import live_entries
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES
    current = str(params.get('current_session_id') or '')
    rows = []
    for entry in await live_entries(connection.authority, connection.actor):
        row = entry.row
        # The same population as the live ``session.list`` feed, hidden chats included: a hidden
        # chat open in a pane still needs its turn's liveness.
        if row.get('source') in INTERNAL_LISTING_SOURCES or entry.closed:
            continue
        sid, started = entry.session_id, float(row.get('started_at') or 0)
        rows.append({'current': sid == current, 'id': sid, 'session_key': sid,
                     'last_active': float(row.get('ended_at') or started), 'started_at': started,
                     'message_count': int(row.get('message_count') or 0),
                     'model': str(row.get('model') or ''), 'preview': '', 'title': row.get('title') or '',
                     'source': row.get('source'), 'status': _status(entry.live, entry.handle, entry.pending)})
    working = [row['id'] for row in rows if row['status'] == 'working']
    if working:
        activity = await asyncio.to_thread(_working_activity, connection.authority.db, working)
        for row in rows:
            if label := activity.get(row['id']):
                row['activity'] = label
    return {'sessions': rows}


def _working_activity(db, session_ids) -> dict[str, str]:
    """Worker thread: each working turn's durable activity stamp, read off its compression TIP (the
    running agent stamps the continuation and the ended root's label is cleared). Desktop's subtitle
    needs it here because the list endpoints are not re-read on an activity heartbeat."""
    tips = {sid: db.get_compression_tip(sid) for sid in session_ids}
    ids = json.dumps(sorted(set(tips.values())))
    with db._read_ctx() as conn:
        labels = dict(conn.execute(
            "SELECT id, last_activity_description FROM sessions WHERE id IN (SELECT value FROM json_each(?))",
            (ids,)).fetchall())
    return {sid: labels[tip] for sid, tip in tips.items() if labels.get(tip)}
