"""Read-only native picker history; enumeration never adopts a transcript."""
import base64
import binascii
import hashlib
import json

from persistence.core.errors import AuthorityRefusal
from session_rules.legacy_local import client_history_row, validate_legacy_row

from gateway.owner_identity import same_owner
from gateway.session_persistence import authority_persistence, owner_rule
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
        raise AuthorityRefusal('invalid_params', field='cursor' if query is None else 'query')
    if query is not None and not (isinstance(query, str) and query.strip() and len(query) <= _QUERY_MAX_CHARS):
        raise AuthorityRefusal('invalid_params', field='query')
    if cursor is not None and not isinstance(cursor, str):
        raise AuthorityRefusal('invalid_params', field='cursor')


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
        raise AuthorityRefusal('invalid_params', field='cursor') from None
    if (tag != _CURSOR_TAG or type(offset) is not int or not 0 <= offset < 2**63
            or held != _binding_digest(binding)):
        raise AuthorityRefusal('invalid_params', field='cursor')
    return offset


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


def _own_profile(store):
    from persistence_assembly import store_owner_name
    return store_owner_name(store.scope.profile_dir) or 'default'


def _native_reader(authority, actor):
    """The native transport old local rows are offered to: the one that may resume (adopt) them."""
    return (getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) == actor
            and {'session:read', 'session:create'} <= actor.capabilities)


def legacy_readable(authority, actor, sid):
    """Whether this picker lists old local row *sid* (unadopted: no receipt, no route) to *actor*, so a
    read of it may show it without adopting it (``gateway/session_stored_reads.py``)."""
    from gateway.session_local_migration import legacy_row
    store = authority_persistence(authority)
    row = store.sessions.get(sid)
    if actor.profile_id != authority.profile_id or row is None or not _native_reader(authority, actor):
        return False
    rule = owner_rule()
    try:
        validate_legacy_row(_own_profile(store), legacy_row(row), subject=actor.subject, rule=rule)
        return store.bindings.authorize_history(session_id=sid, profile_id=actor.profile_id, subject=actor.subject,
                                                may_create='session:create' in actor.capabilities, owner=rule)
    except AuthorityRefusal as exc:
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
        raise AuthorityRefusal('profile_mismatch')
    binding = (workspace, query, include_hidden)
    offset = _decode_cursor(cursor, binding) if cursor is not None else 0
    if not _native_reader(authority, actor):
        return {'sessions': [], 'scope': 'stored', 'next_cursor': None}
    from gateway.session_local_migration import legacy_row
    from persistence.conversation.sessions import SessionOrder, SessionQuery, SessionWindow
    store = authority_persistence(authority)
    own_profile = _own_profile(store)
    rule = owner_rule()
    # Content-less rows are not conversations the picker can open: a launch that
    # never receives a prompt, a reaped orphan and a resumed-over row each leave
    # a zero-message row behind, and listing them fills the picker with blank
    # titles that fail to open. They stay in the store — the listing just skips
    # them, as the sibling human-facing listings already do. A finished cron run
    # is the exception: it resumes from its job, not its transcript (the stored
    # feed's TUI-8 contract offers it), and the scheduler books ``cron_complete``
    # only on a real assistant reply, so the reason — not the source — is what
    # keeps it listed; a suppressed cron row stays hidden with the husks.
    listing = SessionQuery(exclude_sources=_excluded_sources(), order=SessionOrder.LAST_ACTIVE,
                           include_hidden=include_hidden, workspace_key=workspace, search_text=query,
                           min_message_count=1, keep_empty_end_reasons=('cron_complete',))
    sessions, seen, sources, exhausted = [], set(), [], False  # sources: (first row, current segment, listed row)
    while len(sessions) < limit:
        rows = store.sessions.list_sessions(listing, SessionWindow(limit=_BATCH, offset=offset), compact_rows=True)
        end = offset + len(rows)
        for entry in rows:
            offset += 1
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
                    sources.append((root, sid, entry))
                if len(sessions) == limit:
                    break
                continue
            closed = _closed_local_session(authority, actor, row, tip)
            if closed is not None:
                if closed['id'] not in seen:
                    seen.add(closed['id'])
                    sessions.append(closed)
                    sources.append((closed['id'], sid, entry))
                if len(sessions) == limit:
                    break
                continue
            if not legacy_readable(authority, actor, sid):
                continue
            sessions.append(_picker_row(row.session_id, row, tip, sid))
            sources.append((entry.lineage_root_id or row.session_id, sid, entry))
            if len(sessions) == limit:
                break
        if len(rows) < _BATCH:
            exhausted = offset == end
            break
    if picker and sessions:
        from gateway.session_picker_fields import fill_picker_fields
        fill_picker_fields(store, [(row, *source) for row, source in zip(sessions, sources)])
    return {'sessions': sessions, 'scope': 'stored',
            'next_cursor': None if exhausted else _encode_cursor(offset, binding)}


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
    its stored receipt as ``authorize`` and ``closed_and_unviewed`` would after a restore. No viewer.
    The stored receipt is judged by the checks a restore runs on it (``restorable_receipt``); the
    restore's live route check (another session on its route) stays the restore's."""
    from gateway.session_local_recovery import restorable_receipt
    if actor.profile_id != authority.profile_id or 'session:read' not in actor.capabilities:
        return False
    store = authority_persistence(authority)
    try:
        receipt, entry = restorable_receipt(authority, owner)
        row = store.sessions.get(entry.session_id)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return ((same_owner(receipt['principal_id'], actor.subject) or 'session:operator' in actor.capabilities)
            and row is not None and row.ended_at is not None
            and not store.admissions.list(session_id=owner))
