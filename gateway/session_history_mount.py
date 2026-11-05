"""Read-only mount of stored history no receipt, route or binding restores (#143547 RT-30b/48/49).

Pre-authority cron runs, subagent transcripts, API rows without a binding, and rows a third-party
WebUI or the foreign-session importer wrote: before #106742 a viewer opened any of them straight
from ``state.db``. Nothing can run them (no frozen policy, no lane, and adopting a WebUI row as
writable would legitimise its second writer), so the owner mounts them for reading only. The mount
is keyed by no lane's route, so nothing route-keyed acts through it, and every capability but
``session:read`` is refused by name. Archive, pin and delete still work as for any read-only mount
(gateway/session_mutations.py), but only a leaf row deletes. Old local CLI/TUI/Desktop rows are not mounted
here: resume adopts them as writable conversations (gateway/session_local_migration.py), as it does a phone
app's self-labelled chat (``mobile``, ``ios``) for its native owner; anyone else still reads that one here.
"""

from persistence.core.errors import AuthorityRefusal

# Never a lane key, so no lane publishing a conversation takes this mount over.
_ROUTE_PREFIX = 'history:'


def _reader(authority, actor, sid):
    """The importer an import binding names, else the profile's owner: its native peer (the evidence
    legacy adoption takes) or an operator. Knowing an old id is not ownership."""
    from gateway.session_persistence import authority_persistence, owner_rule
    from session_rules.legacy_local import binding_names
    binding = authority_persistence(authority).bindings.history_binding(sid)
    if binding is not None:
        return binding_names(binding, actor.profile_id, actor.subject, owner_rule())
    return ('session:operator' in actor.capabilities
            or getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) == actor)


def mount_history(authority, actor, sid):
    """Mount *sid* read-only for its owner; False when another loader owns it or the actor may not."""
    from gateway.session_persistence import authority_persistence
    from persistence.authority.vocabulary import LEGACY_LOCAL_SOURCES
    from persistence_assembly import store_owner_name
    store = authority_persistence(authority)
    row = store.sessions.get(sid)
    if (row.source in LEGACY_LOCAL_SOURCES | {'local'}
            or row.profile_name not in (None, '', store_owner_name(store.scope.profile_dir) or 'default')
            or not _reader(authority, actor, sid)):
        return False
    try:
        if store.bindings.api_binding(sid) is not None:
            return False  # restore_api_session's
    except ValueError:
        return False  # a stored binding that does not decode is still restore_api_session's
    authority._require_admission_open()
    from gateway.session_authority import LiveSession
    authority.sessions.setdefault(sid, LiveSession(None, _ROUTE_PREFIX + sid))
    return True


def is_history_mount(live):
    return live.source is None and live.route.startswith(_ROUTE_PREFIX)


def require_history_leaf(authority, sid):
    """A delete takes its whole compression chain, every chain row's delegate subtree, and detaches
    their other children, and a history mount shows the owner only the chain, so it deletes only a
    chain nothing else hangs off (as #140692 by @pnpwin did for subagents)."""
    if not is_history_mount(authority.sessions[sid]):
        return
    from gateway.session_persistence import authority_persistence, owner_rule
    store = authority_persistence(authority)
    chain = set(store.lineage.lineage(sid))
    footprint = store.runtime.delete_footprint(sid, owner=owner_rule())
    if not chain.issuperset([*footprint.targets, *footprint.children]):
        raise AuthorityRefusal('session_has_children', field='session_id',
                               detail='this stored conversation has child sessions a delete would take with it')


def read_only():
    return AuthorityRefusal('session_read_only', field='session_id',
                            detail='this stored conversation can be read but not continued; start a new chat')


def require_history_access(authority, actor, sid, capability):
    if not _reader(authority, actor, sid):
        raise AuthorityRefusal('not_found')
    if capability != 'session:read':
        raise read_only()
