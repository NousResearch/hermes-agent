"""Read-only native picker history; enumeration never adopts a transcript."""
from hermes_state_local_migration import validate_legacy_row
from hermes_state_mutation_binding import authorize_history
from hermes_state_runtime import RuntimeStoreError


def stored_picker_sessions(authority, actor, *, limit, include_hidden=False):
    """Offer legacy local rows only to the same native transport that may resume them.

    Canonical sessions are already in the live feed. Old local rows have no policy
    receipt until explicit resume; listing them must not claim or restore anything.
    """
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if (getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor
            or not {'session:read', 'session:create'} <= actor.capabilities):
        return {'sessions': [], 'scope': 'stored'}
    db = authority.db
    sessions, offset, seen = [], 0, set()
    while len(sessions) < limit:
        rows = db.list_sessions_rich(sources=['cli', 'tui', 'gui', 'desktop', 'local'], limit=200, offset=offset,
            order_by_last_active=True, compact_rows=True, include_hidden=include_hidden)
        for row in rows:
            sid = db.get_compression_tip(row['id']) or row['id']
            tip = db.get_session(sid)
            if tip is None:
                continue
            closed = _closed_local_session(authority, actor, row, tip)
            if closed is not None:
                if closed['id'] not in seen:
                    seen.add(closed['id'])
                    sessions.append(closed)
                if len(sessions) == limit:
                    break
                continue
            try:
                validate_legacy_row(db, tip)
                with db._read_ctx() as conn:
                    if not authorize_history(conn, actor, sid):
                        continue
            except RuntimeStoreError as exc:
                if exc.reason not in {'not_found', 'permission_denied'}:
                    raise
                continue
            sessions.append({'id': row['id'], 'session_id': sid, 'resolved_id': sid,
                'title': row.get('title') or '', 'source': tip['source'],
                'started_at': row.get('started_at'), 'message_count': row.get('message_count', 0),
                'model': tip.get('model'), 'running': False})
            if len(sessions) == limit:
                break
        if len(rows) < 200:
            break
        offset += len(rows)
    return {'sessions': sessions, 'scope': 'stored'}


def _closed_local_session(authority, actor, row, tip):
    """A canonical local session (created here, or adopted from main) its last viewer closed: main's
    picker listed it as resumable. None for anything else, including a session another principal owns."""
    from gateway.session_contract import SessionRef
    from gateway.session_local_close import closed_and_unviewed
    owner = tip.get('chat_id')
    live = authority.sessions.get(owner) if owner else None
    if live is None:
        return None
    try:
        handle = authority._handle(SessionRef(authority.profile_id, owner))
        authority.authorize(actor, handle.ref, 'session:read')
    except RuntimeStoreError:
        return None
    if not closed_and_unviewed(authority, owner, handle):
        return None
    return {'id': owner, 'session_id': owner, 'resolved_id': owner, 'title': tip.get('title') or row.get('title') or '',
            'source': tip['source'], 'started_at': row.get('started_at'),
            'message_count': row.get('message_count', 0), 'model': tip.get('model'), 'running': False}
