"""Read-only native picker history; enumeration never adopts a transcript."""
from persistence.core.errors import AuthorityRefusal
from session_rules.legacy_local import client_history_row, validate_legacy_row

from gateway.owner_identity import same_owner
from gateway.session_persistence import authority_persistence, owner_rule
from gateway.session_route_recovery import _messaging_platform, mountable

# Local labels (a cron run is a canonical local session whose last viewer is its own fire).
_LOCAL_SOURCES = ('cli', 'tui', 'gui', 'desktop', 'local', 'cron')


def _excluded_sources():
    """Main's picker listed every source but the internal ones; the API transport has its own restore.
    Plugin platforms are ``Platform`` pseudo-members enum iteration never yields, so the query
    excludes rather than enumerates, and each messaging row is judged by the cold mount's own check."""
    from persistence.conversation.sessions import INTERNAL_LISTING_SOURCES
    return (*INTERNAL_LISTING_SOURCES, 'api_server')


def _picker_row(row_id, row, tip, sid):
    return {'id': row_id, 'session_id': sid, 'resolved_id': sid, 'title': row.title or '',
            'source': tip.source, 'started_at': row.started_at,
            'message_count': row.message_count, 'model': tip.model, 'running': False}


def stored_picker_sessions(authority, actor, *, limit, include_hidden=False):
    """Offer legacy local rows only to the same native transport that may resume them.

    Canonical sessions are already in the live feed. Old local rows have no policy
    receipt until explicit resume; listing them must not claim or restore anything.
    """
    if actor.profile_id != authority.profile_id:
        raise AuthorityRefusal('profile_mismatch')
    if (getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor
            or not {'session:read', 'session:create'} <= actor.capabilities):
        return {'sessions': [], 'scope': 'stored'}
    from gateway.session_local_migration import legacy_row
    from persistence.conversation.sessions import SessionOrder, SessionQuery, SessionWindow
    from persistence_assembly import store_owner_name
    store = authority_persistence(authority)
    own_profile = store_owner_name(store.scope.profile_dir) or 'default'
    rule = owner_rule()
    query = SessionQuery(exclude_sources=_excluded_sources(), order=SessionOrder.LAST_ACTIVE, include_hidden=include_hidden)
    sessions, offset, seen = [], 0, set()
    while len(sessions) < limit:
        rows = store.sessions.list_sessions(query, SessionWindow(limit=200, offset=offset), compact_rows=True)
        for entry in rows:
            row = entry.session
            local = row.source in _LOCAL_SOURCES or client_history_row(legacy_row(row), rule)
            if not local and _messaging_platform(row.source) is None:
                continue  # neither a local nor a messaging label (acp, unknown, ...): nothing this picker resumes
            sid = store.lineage.tip(row.session_id)
            tip = store.sessions.get(sid)
            if tip is None:
                continue
            if not local:
                # A messaging conversation, under its lineage root (the cold mount; the listing
                # projects a compressed lineage onto its tip): this profile's rows the mount opens.
                root = entry.lineage_root_id or row.session_id
                if (row.profile_name in (None, '', own_profile) and mountable(authority, root)
                        and root not in seen):
                    seen.add(root)
                    sessions.append(_picker_row(root, row, tip, root))
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
                validate_legacy_row(own_profile, legacy_row(tip), subject=actor.subject, rule=rule)
                if not store.bindings.authorize_history(session_id=sid, profile_id=actor.profile_id, subject=actor.subject,
                                                        may_create='session:create' in actor.capabilities, owner=rule):
                    continue
            except AuthorityRefusal as exc:
                if exc.reason not in {'not_found', 'permission_denied'}:
                    raise
                continue
            sessions.append(_picker_row(row.session_id, row, tip, sid))
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
    owner = tip.chat_id
    live = authority.sessions.get(owner) if owner else None
    if live is None:
        if not owner or not _stored_closed(authority, actor, owner):
            return None
    else:
        try:
            handle = authority._handle(SessionRef(authority.profile_id, owner))
            authority.authorize(actor, handle.ref, 'session:read')
        except AuthorityRefusal:
            return None
        if not closed_and_unviewed(authority, owner, handle):
            return None
    return {'id': owner, 'session_id': owner, 'resolved_id': owner, 'title': tip.title or row.title or '',
            'source': tip.source, 'started_at': row.started_at,
            'message_count': row.message_count, 'model': tip.model, 'running': False}


def _stored_closed(authority, actor, owner):
    """Boot leaves an ended session unrestored and a listing never restores one (a worker thread): judge
    its stored receipt as ``authorize`` and ``closed_and_unviewed`` would after a restore. No viewer."""
    if actor.profile_id != authority.profile_id or 'session:read' not in actor.capabilities:
        return False
    store = authority_persistence(authority)
    try:
        receipt = store.bindings.receipt(owner)
        row = store.sessions.get(receipt.entry['session_id'])
    except (AuthorityRefusal, KeyError, TypeError):
        return False
    return (receipt.profile_id == authority.profile_id
            and (same_owner(receipt.principal_id, actor.subject) or 'session:operator' in actor.capabilities)
            and row is not None and row.ended_at is not None
            and not store.admissions.list(session_id=owner))
