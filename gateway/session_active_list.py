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

from gateway.session_contract import SessionHandle, SessionRef
from hermes_state_runtime import RuntimeStoreError, list_session_admissions


def _status(live, handle, pending) -> str:
    if handle.execution_state == 'running':
        prompts = live.controls.snapshot(handle.ref.session_id, handle.execution_generation)
        return 'waiting' if prompts else 'working'
    # Admitted, not yet claimed by the drain: the turn exists, its agent has not started.
    return 'starting' if any(row['status'] == 'queued' for row in pending) else 'idle'


def _read_state(authority, sid):
    """Storage only: never inspect or mutate the loop-owned live session here."""
    from gateway.session_local_close import _local_target
    row = authority.db.get_session(sid)
    pending = list_session_admissions(authority.db, session_id=sid)
    target = _local_target(authority, sid)
    target_row = authority.db.get_session(target) if target else None
    return row, pending, bool(target_row and target_row.get('ended_at') is not None)


async def active_list(connection, ref, params):
    if set(params) - {'profile', 'current_session_id'}:
        raise RuntimeStoreError('invalid_params')
    if 'session:read' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES
    authority = connection.authority
    current = str(params.get('current_session_id') or '')
    rows = []
    for sid, live in tuple(authority.sessions.items()):
        candidate = SessionRef(connection.actor.profile_id, sid)
        try:
            # authorize can restore missing sessions: skip stale entries before calling it.
            if authority.sessions.get(sid) is not live:
                continue
            authority.authorize(connection.actor, candidate, 'session:read')
            row, pending, closed = await asyncio.to_thread(_read_state, authority, sid)
            if row is None or authority.sessions.get(sid) is not live:
                continue
            authority.authorize(connection.actor, candidate, 'session:read')
        except RuntimeStoreError as exc:
            if exc.reason not in {'permission_denied', 'profile_mismatch', 'not_found'}:
                raise
            continue
        # Match _pending_rows' unpublished-settlement mask on the owner loop, not its worker.
        with live.event_stream.lock:
            pending = {r['admission_id']: r for r in pending}
            pending.update(live.settling)
            pending = list(pending.values())
            # A claim can advance while storage is off-loop. Its stream stamp is current
            # under this lock; an old row must never prune that generation's live controls.
            generation = live.event_stream.execution.get('execution_generation')
            state = 'unknown' if any(r['status'] == 'unknown' for r in pending) else (
                'running' if generation is not None or any(r['status'] == 'started' for r in pending) else 'idle')
            handle = SessionHandle(candidate, authority.instance_id, authority.epoch,
                                   row['runtime_revision'],
                                   generation if generation is not None else row['runtime_generation'], state)
            # Hidden chats open in a pane still need liveness; only closed, idle,
            # unviewed local chats leave the live feed.
            if row.get('source') in INTERNAL_LISTING_SOURCES or (
                    closed and not live.subscribers and state == 'idle'):
                continue
            status = _status(live, handle, pending)
        started = float(row.get('started_at') or 0)
        rows.append({'current': sid == current, 'id': sid, 'session_key': sid,
                     'last_active': float(row.get('ended_at') or started), 'started_at': started,
                     'message_count': int(row.get('message_count') or 0),
                     'model': str(row.get('model') or ''), 'preview': '', 'title': row.get('title') or '',
                     'status': status})
    return {'sessions': rows}
