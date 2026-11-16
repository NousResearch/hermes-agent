"""Read-only native picker history; enumeration never adopts a transcript."""
import base64
import binascii
import hashlib
import json

from hermes_state_local_migration import client_history_row, validate_legacy_row
from hermes_state_mutation_binding import authorize_history
from hermes_state_runtime import RuntimeStoreError

from gateway.owner_identity import same_owner
from gateway.session_route_recovery import _messaging_platform, mountable

# Local labels (a cron run is a canonical local session whose last viewer is its own fire).
_LOCAL_SOURCES = ('cli', 'tui', 'gui', 'desktop', 'local', 'cron')
_CURSOR_TAG = 'l1'
_CURSOR_MAX_CHARS = 1024
_QUERY_MAX_CHARS = 200
_BATCH = 200


def check_page_params(params):
    """``query`` and ``cursor`` filter and page the stored listing only; anything else is ``invalid_params``."""
    query, cursor = params.get('query'), params.get('cursor')
    if query is None and cursor is None:
        return
    if params.get('scope') != 'stored' or 'title' in params:
        raise RuntimeStoreError('invalid_params', field='cursor' if query is None else 'query')
    if query is not None and not (isinstance(query, str) and query.strip() and len(query) <= _QUERY_MAX_CHARS):
        raise RuntimeStoreError('invalid_params', field='query')
    if cursor is not None and not isinstance(cursor, str):
        raise RuntimeStoreError('invalid_params', field='cursor')


def _binding_digest(binding):
    """The filters a cursor pages, at a fixed size: a cursor must fit ``_CURSOR_MAX_CHARS`` whatever their length."""
    return hashlib.sha256(json.dumps(list(binding)).encode()).hexdigest()[:32]


def _encode_cursor(offset, binding):
    """*offset*: the storage offset of the first row no page has examined; *binding*: the filters it pages."""
    held = [_CURSOR_TAG, offset, _binding_digest(binding)]
    return base64.urlsafe_b64encode(json.dumps(held).encode()).decode().rstrip('=')


def _decode_cursor(cursor, binding):
    try:
        if len(cursor) > _CURSOR_MAX_CHARS:
            raise ValueError('cursor too long')
        tag, offset, held = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
    except (TypeError, ValueError, binascii.Error):
        raise RuntimeStoreError('invalid_params', field='cursor') from None
    if (tag != _CURSOR_TAG or type(offset) is not int or not 0 <= offset < 2**63
            or held != _binding_digest(binding)):
        raise RuntimeStoreError('invalid_params', field='cursor')
    return offset


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


def _native_reader(authority, actor):
    """The native transport old local rows are offered to: the one that may resume (adopt) them."""
    return (getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) == actor
            and {'session:read', 'session:create'} <= actor.capabilities)


def legacy_readable(authority, actor, sid):
    """Whether this picker lists old local row *sid* (unadopted: no receipt, no route) to *actor*, so a
    read of it may show it without adopting it (``gateway/session_stored_reads.py``)."""
    db = authority.db
    row = db.get_session(sid)
    if actor.profile_id != authority.profile_id or row is None or not _native_reader(authority, actor):
        return False
    try:
        validate_legacy_row(db, row, subject=actor.subject)
        with db._read_ctx() as conn:
            return authorize_history(conn, actor, sid)
    except RuntimeStoreError as exc:
        if exc.reason not in {'not_found', 'permission_denied'}:
            raise
        return False


def stored_picker_sessions(authority, actor, *, limit, include_hidden=False, workspace=None, picker=False,
                           query=None, cursor=None):
    """Offer legacy local rows only to the same native transport that may resume them.

    Canonical sessions are already in the live feed. Old local rows have no policy
    receipt until explicit resume; listing them must not claim or restore anything.
    *workspace*: only the folder's rows (``hermes -c``'s rule). *picker*: each row carries the picker
    fields, from the columns the listing already read (``gateway/session_picker_fields.py``).
    *query*: only titles or ids containing it (``list_sessions_rich(search_query=)``). *cursor*: continue after
    the page that returned it; ``next_cursor`` is None once the storage listing is exhausted.
    """
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    binding = (workspace, query, include_hidden)
    offset = _decode_cursor(cursor, binding) if cursor is not None else 0
    if not _native_reader(authority, actor):
        return {'sessions': [], 'scope': 'stored', 'next_cursor': None}
    db = authority.db
    own_profile = db._own_profile_name() or 'default'
    sessions, seen, sources, exhausted = [], set(), [], False  # sources: (first row, current segment, listed row)
    while len(sessions) < limit:
        rows = db.list_sessions_rich(exclude_sources=_excluded_sources(), limit=_BATCH, offset=offset,
            order_by_last_active=True, compact_rows=True, include_hidden=include_hidden, workspace_key=workspace,
            search_query=query)
        end = offset + len(rows)
        for row in rows:
            offset += 1
            local = row['source'] in _LOCAL_SOURCES or client_history_row(row)
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
                    sources.append((root, sid, row))
                if len(sessions) == limit:
                    break
                continue
            closed = _closed_local_session(authority, actor, row, tip)
            if closed is not None:
                if closed['id'] not in seen:
                    seen.add(closed['id'])
                    sessions.append(closed)
                    sources.append((closed['id'], sid, row))
                if len(sessions) == limit:
                    break
                continue
            if not legacy_readable(authority, actor, sid):
                continue
            sessions.append(_picker_row(row, tip, sid))
            sources.append((row.get('_lineage_root_id') or row['id'], sid, row))
            if len(sessions) == limit:
                break
        if len(rows) < _BATCH:
            exhausted = offset == end
            break
    if picker and sessions:
        from gateway.session_picker_fields import fill_picker_fields
        fill_picker_fields(db, [(row, *source) for row, source in zip(sessions, sources)])
    return {'sessions': sessions, 'scope': 'stored',
            'next_cursor': None if exhausted else _encode_cursor(offset, binding)}


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
            'source': tip['source'], 'started_at': row.get('started_at'),
            'message_count': row.get('message_count', 0), 'model': tip.get('model'), 'running': False}


def _stored_closed(authority, actor, owner):
    """Boot leaves an ended session unrestored and a listing never restores one (a worker thread): judge
    its stored receipt as ``authorize`` and ``closed_and_unviewed`` would after a restore. No viewer.
    The stored receipt is judged by the checks a restore runs on it (``restorable_receipt``); the
    restore's live route check (another session on its route) stays the restore's."""
    from gateway.session_local_recovery import restorable_receipt
    from hermes_state_runtime import list_session_admissions
    if actor.profile_id != authority.profile_id or 'session:read' not in actor.capabilities:
        return False
    try:
        receipt, entry = restorable_receipt(authority, owner)
        row = authority.db.get_session(entry.session_id)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return ((same_owner(receipt['principal_id'], actor.subject) or 'session:operator' in actor.capabilities)
            and row is not None and row['ended_at'] is not None
            and not list_session_admissions(authority.db, session_id=owner))
