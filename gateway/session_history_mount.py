"""Read-only mount of stored history no receipt, route or binding restores (#143547 RT-30b/48/49).

Pre-authority cron runs, subagent transcripts, API rows without a binding, and rows a third-party
WebUI or the foreign-session importer wrote: before #106742 a viewer opened any of them straight
from ``state.db``. Nothing can run them (no frozen policy, no lane, and adopting a WebUI row as
writable would legitimise its second writer), so the owner mounts them for reading only. The mount
is ``superseded``, so nothing route-keyed acts through it, and every capability but
``session:read`` is refused by name. Archive, pin and delete still work as for any read-only mount
(gateway/session_mutations.py), but only a leaf row deletes. Old local CLI/TUI/Desktop rows are not mounted
here: resume adopts them as writable conversations (gateway/session_local_migration.py).
"""
import json

from hermes_state_runtime import RuntimeStoreError

# Never a lane key, so no lane publishing a conversation takes this mount over.
_ROUTE_PREFIX = 'history:'


def _reader(authority, actor, sid):
    """The importer an import binding names, else the profile's owner: its native peer (the evidence
    legacy adoption takes) or an operator. Knowing an old id is not ownership."""
    from hermes_state_mutation_binding import history_binding
    with authority.db._read_ctx() as conn:
        binding = history_binding(conn, sid)
    if binding is not None:
        return binding == json.dumps([actor.profile_id, actor.subject], separators=(',', ':'))
    return ('session:operator' in actor.capabilities
            or getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) == actor)


def mount_history(authority, actor, sid):
    """Mount *sid* read-only for its owner; False when another loader owns it or the actor may not."""
    from hermes_state_local import API_BINDING_PREFIX
    from hermes_state_local_migration import LEGACY_LOCAL_SOURCES
    row = authority.db.get_session(sid)
    if (row['source'] in LEGACY_LOCAL_SOURCES | {'local'}
            or row.get('profile_name') not in (None, '', authority.db._own_profile_name() or 'default')
            or not _reader(authority, actor, sid)):
        return False
    with authority.db._read_ctx() as conn:
        if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (API_BINDING_PREFIX + sid,)).fetchone():
            return False  # restore_api_session's
    authority._require_admission_open()
    from gateway.session_authority import LiveSession
    authority.sessions.setdefault(sid, LiveSession(None, _ROUTE_PREFIX + sid, superseded=True, history_mount=True))
    return True


def is_history_mount(live):
    return live.source is None and live.route.startswith(_ROUTE_PREFIX)


def require_history_leaf(authority, sid):
    """A delete takes its whole compression chain, every chain row's delegate subtree, and detaches
    their other children, and a history mount shows the owner only the chain, so it deletes only a
    chain nothing else hangs off (as #140692 by @pnpwin did for subagents)."""
    if not is_history_mount(authority.sessions[sid]):
        return
    from hermes_state_mutation_guards import delete_targets
    chain = set(authority.db.get_compression_lineage(sid))
    with authority.db._read_ctx() as conn:
        targets = delete_targets(conn, sid)
        children = conn.execute(f"SELECT id FROM sessions WHERE parent_session_id IN ({','.join('?' * len(targets))})",
                                targets).fetchall()
        if not chain.issuperset([*targets, *(row[0] for row in children)]):
            raise RuntimeStoreError('session_has_children', field='session_id',
                                    detail='this stored conversation has child sessions a delete would take with it')


def read_only():
    return RuntimeStoreError('session_read_only', field='session_id',
                             detail='this stored conversation can be read but not continued; start a new chat')


def require_history_access(authority, actor, sid, capability):
    if not _reader(authority, actor, sid):
        raise RuntimeStoreError('not_found')
    if capability != 'session:read':
        raise read_only()
