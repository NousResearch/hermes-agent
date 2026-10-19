"""Mount a cold messaging-platform session into the authority for a local viewer (TUI-8).

A messaging session (Telegram, Weixin, Discord, ...) is published into the authority only when its
lane's first post-restart inbound runs ``_publish_route``, so ``session.resume`` of its row answered
``not_found`` until someone wrote to the lane again, on every gateway restart. Before the canonical
authority (#106742) a viewer opened any such row straight from ``state.db``; this restores that for
every cold row of this profile that carries a ``session_key`` and a messaging-platform source:

* the lane's routing entry still names the row: its persisted ``origin`` is the highest-fidelity
  source, exactly what that first inbound would have published;
* the route is gone or moved on (``/new``, reset, a pruned entry): the row's own ``origin_json``,
  else its columns, rebuild the source.

Either way the source is revived through ``_restored_source``, so delivery stays pinned to the
persisted receiving bot. A row the lane moved past mounts READ-ONLY (``LiveSession.superseded``): its
only route is the lane's current key, which names the newer conversation, so a Stop, an agent lookup
or a turn through it would act on that conversation. One gateway owns every session and never
crosses them, so writes are refused by name (``session_superseded``) rather than given a private
route that no adapter would deliver to (#140183 review F2). A compression continuation is never
mounted under its physical id: the lane's turns publish to the lineage root (``logical_owner``), so
``session.resume`` maps it to that root, as normal admission does. Rows without a ``session_key`` (subagents) have nothing to rebuild a lane
from. LOCAL rows (cron, CLI, TUI) restore from their private receipt and API rows from their binding
(``restore_api_session``); neither is ever mounted here. The mount semantics are the first inbound's
``_publish_route``: a row that ended for a recoverable reason (``agent_close``, ``ws_orphan_reap``,
...) is mounted live too, as the lane's next message would reopen it.
"""
import json
from datetime import datetime

from gateway.config import Platform


def _messaging_platform(label):
    """The messaging ``Platform`` a row's ``source`` label names; None for the API transport (its
    own restore) and for labels no platform answers to (the CLI/TUI/cron local labels)."""
    try:
        platform = Platform(label)
    except ValueError:
        return None
    return None if platform in (Platform.LOCAL, Platform.API_SERVER) else platform


def _routed_entry(authority, sid, route):
    """The routing entry that still names *sid* under *route*, else None."""
    store = authority.runner.session_store
    with store._lock:
        entry = store._entry_locked(route)
    if (entry is None or entry.origin is None or entry.session_key != route
            or authority.logical_owner(entry.session_id) != sid):
        return None
    return entry


def _row_source(row, platform, profile):
    """The row's persisted origin, else its columns (``channel_directory`` reads rows the same way)."""
    from gateway.session import SessionSource
    try:
        return SessionSource.from_dict(json.loads(row['origin_json']))
    except (TypeError, ValueError, KeyError):
        pass
    if row.get('chat_id') is None:
        return None
    source = SessionSource(
        platform=platform, chat_id=str(row['chat_id']), user_id=row.get('user_id'),
        chat_type=row.get('chat_type') or 'dm', thread_id=row.get('thread_id'),
        chat_name=row.get('display_name'))
    if profile is not None:
        source.profile = profile
    return source


def _names_another_conversation(authority, sid, route):
    """The lane's current routing entry names a conversation other than *sid* (``/new``, reset)."""
    store = authority.runner.session_store
    with store._lock:
        entry = store._entry_locked(route)
    return entry is not None and authority.logical_owner(entry.session_id) != sid


def _row_entry(row, platform, route, profile):
    """A routing entry rebuilt from the row alone (the route no longer vouches for it)."""
    from gateway.session import SessionEntry
    source = _row_source(row, platform, profile)
    if source is None or source.platform != platform:
        return None
    started = datetime.fromtimestamp(row['started_at']) if row.get('started_at') else datetime.now()
    return SessionEntry(session_key=route, session_id=row['id'], created_at=started, updated_at=started,
                        origin=source, platform=platform, chat_type=source.chat_type,
                        transport_profile=row.get('transport_profile'))


def restore_routed_session(authority, sid):
    """Publish *sid* as its lane would after a restart; None when this row is not one a route or a
    messaging-platform row can stand for."""
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionRef
    row = authority.db.get_session(sid)
    route = row.get('session_key') if row is not None else None
    platform = _messaging_platform(row['source']) if route else None
    if platform is None or authority.logical_owner(sid) != sid:
        return None  # a compression continuation: its lineage root is the mount (session.resume maps it)
    store = authority.runner.session_store
    registry = getattr(authority.runner, 'session_authorities', None)
    profile = store._named_profile_for_key(route)
    if profile != (registry.profile_name(authority) if registry is not None else None):
        return None
    routed = _routed_entry(authority, sid, route)
    entry = routed or _row_entry(row, platform, route, profile)
    if entry is None:
        return None
    superseded = routed is None and _names_another_conversation(authority, sid, route)
    authority._require_admission_open()
    # Re-pin the persisted receiving bot so delivery after a restart goes through it or nowhere,
    # never the default bot by heuristic (gateway/AGENTS.md § Identity survives the process).
    source = authority.runner._restored_source(entry)
    authority.sessions.setdefault(sid, LiveSession(source, route, superseded=superseded, history_mount=True))
    return SessionRef(authority.profile_id, sid)


def messaging_owner_ref(authority, ref):
    """The mount a messaging row resumes on: its compression lineage root, where the lane's turns
    publish their events (``logical_owner``). Any other row keeps *ref*."""
    row = authority.db.get_session(ref.session_id)
    if row is None or not row.get('session_key') or _messaging_platform(row['source']) is None:
        return ref
    from gateway.session_contract import SessionRef
    owner = authority.logical_owner(ref.session_id)
    return ref if owner == ref.session_id else SessionRef(ref.profile_id, owner)
