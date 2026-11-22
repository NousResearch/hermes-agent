"""Title-keyed local session selection, resolved read-only by the authority.

``hermes chat --resume <title>`` / ``-c <title> [--create-if-missing]`` used to open
SessionDB in the client; the zero-writer client now sends the name and the owner
applies the classic CLI rule (``hermes_cli.main._resolve_session_by_name_or_id``).
"""
from __future__ import annotations

from persistence.core.errors import AuthorityRefusal


def resolve_titled_session(authority, actor, name, *, missing_ok=False):
    """Exact session id first, then title (latest ``"<title> #N"`` continuation wins over
    the bare title), projected forward to the compression tip. Only a local session the
    actor may read is returned, or one a messaging lane took over from a local surface
    (``handed_off_ref``); a foreign or other non-local match is ``not_found`` so another
    principal's titles are never confirmed."""
    if not isinstance(name, str) or not name.strip():
        raise AuthorityRefusal('invalid_params')
    from gateway.session_persistence import authority_persistence
    store = authority_persistence(authority)
    row = store.sessions.get(name)
    # Title-as-identity (not a literal id): skip deliberately archived rows so a
    # retired hidden Bot Chat is not treated as still occupying the name (#139779).
    # Recoverable auto-archives stay resolvable for unarchive_recoverable_session.
    sid = row.session_id if row else store.sessions.resolve_by_title(name)
    tip = store.sessions.get(store.lineage.tip(sid)) if sid else None
    if tip is None:
        if missing_ok:
            return None
        raise AuthorityRefusal('not_found')
    from gateway.session_route_recovery import handed_off_ref
    moved = handed_off_ref(authority, tip.session_id)
    try:
        if moved is not None:
            authority.authorize(actor, moved, 'session:read')
            return moved
        from gateway.session_local_migration import resolve_local_target
        ref = resolve_local_target(authority, actor, tip.session_id)
    except AuthorityRefusal as exc:
        if exc.reason == 'permission_denied':
            raise AuthorityRefusal('not_found') from None
        raise
    return ref


def resolve_latest_session(authority, actor, family, workspace=None):
    """The most recently active session of a surface family (``cli`` = classic CLI incl. its finite
    one-shot runs, ``tui``), scoped to ``workspace`` (git root / cwd) first and then global: the
    classic ``hermes -c`` / ``--resume latest`` rule (``hermes_cli.main._resolve_last_session``),
    applied by the owner so the client opens no database. Only a local session the actor may read
    counts; another principal's sessions are skipped, never confirmed."""
    from agent.session_source import CLI_FAMILY_SOURCES
    families = {'cli': sorted(CLI_FAMILY_SOURCES), 'tui': ['tui']}
    if family not in families or (workspace is not None and (not isinstance(workspace, str) or not workspace)):
        raise AuthorityRefusal('invalid_params')
    from gateway.session_local_migration import resolve_local_target
    from gateway.session_persistence import authority_persistence
    from persistence.conversation.sessions import SessionWindow
    store = authority_persistence(authority)
    # The owner's own rows are stored ``local``, with the family in ``created_source``.
    sources = [*families[family], 'local']
    for scope in ([workspace] if workspace else []) + [None]:
        # A launch stub (the CLI crashed or a resume failed before the first turn) is the newest row
        # and carries the surface from its insert. It has no messages and no admission, so it is never
        # "latest"; a session whose first prompt is admitted but not yet written still is.
        for row in store.sessions.search_sessions(sources=sources, local_created_sources=[family], window=SessionWindow(limit=50),
                                                  workspace_key=scope, skip_launch_stubs=True):
            tip = store.lineage.tip(row.session.session_id)
            try:
                return resolve_local_target(authority, actor, tip)
            except AuthorityRefusal as exc:
                if exc.reason not in {'permission_denied', 'not_found', 'profile_mismatch'}:
                    raise
    raise AuthorityRefusal('not_found')


def validate_title(title):
    from persistence.conversation.sessions import sanitize_title
    try:
        cleaned = sanitize_title(title) if isinstance(title, str) else None
    except ValueError:
        cleaned = None
    if cleaned is None:
        raise AuthorityRefusal('invalid_params')
    return cleaned


def title_new_session(authority, ref, title):
    """User-provenance title on a just-created local session (auto-titling never overwrites it).
    A concurrent creator that won the unique title is a conflict, not a silent orphan."""
    from gateway.session_persistence import authority_persistence
    try:
        authority_persistence(authority).sessions.set_title(ref.session_id, title)
    except ValueError:
        raise AuthorityRefusal('admission_conflict') from None
