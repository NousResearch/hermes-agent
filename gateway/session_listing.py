"""Read-only native picker history; enumeration never adopts a transcript."""
from hermes_state_local_migration import (
    client_history_row, delegated_history, validate_delegated_history, validate_legacy_row,
)
from hermes_state_mutation_binding import authorize_history
from hermes_state_runtime import RuntimeStoreError

from gateway.owner_identity import same_owner
from gateway.session_route_recovery import _messaging_platform, mountable

# Local labels (a cron run is a canonical local session whose last viewer is its own fire).
_LOCAL_SOURCES = ('cli', 'tui', 'gui', 'desktop', 'local', 'cron', 'subagent')


def _excluded_sources():
    """Main's picker listed every source but the internal ones; the API transport has its own restore.
    Plugin platforms are ``Platform`` pseudo-members enum iteration never yields, so the query
    excludes rather than enumerates, and each messaging row is judged by the cold mount's own check."""
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES
    return [*INTERNAL_LISTING_SOURCES, 'api_server']


def _show_subagents(db):
    from pathlib import Path
    from os import PathLike
    from hermes_cli.session_listing import show_subagent_sessions
    path = getattr(db, 'db_path', None)
    if not isinstance(path, (str, PathLike)):
        return False
    path = str(path)
    if not path or path == ':memory:' or path.startswith('file:'):
        return False
    return show_subagent_sessions(Path(path).parent)


def _picker_row(row, tip, sid):
    return {'id': row['id'], 'session_id': sid, 'resolved_id': sid, 'title': row.get('title') or '',
            'preview': row.get('preview') or tip.get('preview') or '',
            'source': tip['source'], 'started_at': row.get('started_at'),
            'message_count': row.get('message_count', 0), 'model': tip.get('model'), 'running': False}


def stored_picker_sessions(authority, actor, *, limit, offset=0, include_hidden=False):
    """Offer legacy local rows only to the same native transport that may resume them.

    Canonical sessions are already in the live feed. Old local rows have no policy
    receipt until explicit resume; listing them must not claim or restore anything.
    """
    if type(offset) is not int or not 0 <= offset <= 9007199254740991 or type(limit) is not int or not 1 <= limit <= 200:
        raise RuntimeStoreError('invalid_params')
    metadata = {'offset': offset, 'limit': limit, 'has_more': False}
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if (getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor
            or not {'session:read', 'session:create'} <= actor.capabilities):
        return {'sessions': [], 'scope': 'stored', **metadata}
    db = authority.db
    own_profile = db._own_profile_name() or 'default'
    include_subagents = _show_subagents(db)
    sessions, db_offset, skipped, seen = [], 0, 0, set()
    while True:
        rows = db.list_sessions_rich(exclude_sources=_excluded_sources(), limit=200, offset=db_offset,
            order_by_last_active=True, compact_rows=True, include_hidden=include_hidden,
            include_subagents=include_subagents)
        # Rich projection replaces each fetched row; it does not compact the window.
        db_offset += len(rows)
        for row in rows:
            candidate = _stored_picker_row(authority, actor, row, own_profile, include_subagents)
            if candidate is None or candidate['resolved_id'] in seen:
                continue
            seen.add(candidate['resolved_id'])
            if skipped < offset:
                skipped += 1
            elif len(sessions) < limit:
                sessions.append(candidate)
            else:
                return {'sessions': sessions, 'scope': 'stored', **metadata, 'has_more': True}
        if len(rows) < 200:
            return {'sessions': sessions, 'scope': 'stored', **metadata}


def _stored_picker_row(authority, actor, row, own_profile, include_subagents):
    """Resolve one eligible history row without adopting or restoring its transcript."""
    db = authority.db
    local = row['source'] in _LOCAL_SOURCES or client_history_row(row)
    if not local and _messaging_platform(row['source']) is None:
        return None
    sid = db.get_compression_tip(row['id']) or row['id']
    tip = db.get_session(sid)
    if tip is None:
        return None
    delegated = delegated_history(tip)
    if delegated:
        if not include_subagents:
            return None
        try:
            with db._read_ctx() as conn:
                validate_delegated_history(conn, tip, subject=actor.subject,
                    profile_id=actor.profile_id, profile_name=own_profile)
        except RuntimeStoreError:
            return None
    if not local:
        # Messaging history is projected onto the lineage root the cold mount opens.
        root = row.get('_lineage_root_id') or row['id']
        if row.get('profile_name') in (None, '', own_profile) and mountable(authority, root):
            return _picker_row({**row, 'id': root}, tip, root)
        return None
    if delegated and tip.get('chat_id'):
        # Existing adoption was validated above; enumeration must not restore adapters.
        return _picker_row(row, tip, sid)
    closed = _closed_local_session(authority, actor, row, tip)
    if closed is not None:
        return closed
    try:
        validate_legacy_row(db, tip, subject=actor.subject, profile_id=actor.profile_id)
        with db._read_ctx() as conn:
            if not authorize_history(conn, actor, sid):
                return None
    except RuntimeStoreError as exc:
        if exc.reason not in {'not_found', 'permission_denied'}:
            raise
        return None
    return _picker_row(row, tip, sid)


def _closed_local_session(authority, actor, row, tip):
    """A canonical local session (created here, or adopted from main) its last viewer closed: main's
    picker listed it as resumable. None for anything else, including a session another principal owns."""
    from gateway.session_contract import SessionRef
    from gateway.session_local_close import closed_and_unviewed
    owner = tip.get('chat_id')
    live = authority.sessions.get(owner) if owner else None
    if live is None:
        if not owner or not _stored_closed(authority, actor, owner):
            return None
    else:
        try:
            handle = authority._handle(SessionRef(authority.profile_id, owner))
            authority.authorize(actor, handle.ref, 'session:read')
        except RuntimeStoreError:
            return None
        if not closed_and_unviewed(authority, owner, handle):
            return None
    return {'id': owner, 'session_id': owner, 'resolved_id': owner, 'title': tip.get('title') or row.get('title') or '',
            'preview': row.get('preview') or tip.get('preview') or '',
            'source': tip['source'], 'started_at': row.get('started_at'),
            'message_count': row.get('message_count', 0), 'model': tip.get('model'), 'running': False}


def _stored_closed(authority, actor, owner):
    """Boot leaves an ended session unrestored and a listing never restores one (a worker thread): judge
    its stored receipt as ``authorize`` and ``closed_and_unviewed`` would after a restore. No viewer."""
    from hermes_state_local import local_receipt
    from hermes_state_runtime import list_session_admissions
    if actor.profile_id != authority.profile_id or 'session:read' not in actor.capabilities:
        return False
    try:
        receipt = local_receipt(authority.db, owner)
        row = authority.db.get_session(receipt['entry']['session_id'])
    except (RuntimeStoreError, KeyError, TypeError):
        return False
    return (receipt.get('profile_id') == authority.profile_id
            and (same_owner(receipt.get('principal_id'), actor.subject) or 'session:operator' in actor.capabilities)
            and row is not None and row['ended_at'] is not None
            and not list_session_admissions(authority.db, session_id=owner))
