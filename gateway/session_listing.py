"""Read-only native picker history; enumeration never adopts a transcript."""
from hermes_state_local_migration import validate_legacy_row
from hermes_state_mutation_binding import authorize_history
from hermes_state_runtime import RuntimeStoreError

from gateway.session_route_recovery import _messaging_platform, mountable

# Local labels (a cron run is a canonical local session whose last viewer is its own fire).
_LOCAL_SOURCES = ('cli', 'tui', 'gui', 'desktop', 'local', 'cron')


def _excluded_sources():
    """Main's picker listed every source but the internal ones; the API transport has its own restore.
    Plugin platforms are ``Platform`` pseudo-members enum iteration never yields, so the query
    excludes rather than enumerates, and each messaging row is judged by the cold mount's own check."""
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES
    return [*INTERNAL_LISTING_SOURCES, 'api_server']


def _picker_row(row, tip, sid):
    return {'id': row['id'], 'session_id': sid, 'resolved_id': sid, 'title': row.get('title') or '',
            'source': tip['source'], 'started_at': row.get('started_at'),
            'message_count': row.get('message_count', 0), 'model': tip.get('model'), 'running': False}


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
    own_profile = db._own_profile_name() or 'default'
    sessions, offset, seen = [], 0, set()
    while len(sessions) < limit:
        rows = db.list_sessions_rich(exclude_sources=_excluded_sources(), limit=200, offset=offset,
            order_by_last_active=True, compact_rows=True, include_hidden=include_hidden)
        for row in rows:
            local = row['source'] in _LOCAL_SOURCES
            if not local and _messaging_platform(row['source']) is None:
                continue  # neither a local nor a messaging label (acp, unknown, ...): nothing this picker resumes
            sid = db.get_compression_tip(row['id']) or row['id']
            tip = db.get_session(sid)
            if tip is None:
                continue
            if not local:
                # A messaging conversation, under its lineage root (the cold mount; the listing
                # projects a compressed lineage onto its tip): this profile's rows the mount opens.
                root = row.get('_lineage_root_id') or row['id']
                if (row.get('profile_name') in (None, '', own_profile) and mountable(authority, root)
                        and root not in seen):
                    seen.add(root)
                    sessions.append(_picker_row({**row, 'id': root}, tip, root))
                if len(sessions) == limit:
                    break
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
                validate_legacy_row(db, tip, subject=actor.subject)
                with db._read_ctx() as conn:
                    if not authorize_history(conn, actor, sid):
                        continue
            except RuntimeStoreError as exc:
                if exc.reason not in {'not_found', 'permission_denied'}:
                    raise
                continue
            sessions.append(_picker_row(row, tip, sid))
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
