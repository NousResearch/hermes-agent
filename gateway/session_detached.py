"""A messaging conversation its chat moved past stays usable from the apps (Teknium's ruling).

When a messaging chat names another conversation (``/new``, ``/resume``, ``/branch``, a handoff,
its next inbound naming another one), a Desktop / Webapp / mobile view still open on the previous
conversation keeps working: its turns, Stop and controls act on that conversation alone, and its
replies reach its viewers only, never the chat, which now gets only its new conversation's replies.

``detach`` moves the live session off the chat's route onto a key that names the conversation
(``agent:<ns>:detached:<session id>``) and pins that key on a copy of its source
(``SessionSource.detached_route``). From there the turn runs on the ordinary messaging path:

* ``build_session_key`` answers the pinned key, so the runner's per-route state (running slot,
  agent cache, busy guard, the agent a Stop reaches) is the conversation's own, never the chat's;
* ``SessionStore.get_or_create_session`` resolves the key to the conversation itself
  (``detached_entry``), and never creates or persists a route for it;
* delivery goes through the profile's ``LocalSessionAdapter`` (gateway/session_local.py), as a local
  session's does: its sends publish nothing, viewers read the canonical event stream.

The key carries no platform slot, so a reader that parses a chat out of a session key
(``gateway.run._parse_session_key``) finds none and delivers nowhere. Its turns bind the session
variables a Desktop session's turn binds (``session_env_source``), so a reminder, a kanban subscription
or a ``send_message`` it starts names the app, never the chat. The chat naming the conversation again
(``/resume``) moves it back onto the chat's route (``SessionAuthority.name_lane``).

Every route change reaches ``name_lane`` from one place: the session store announces each conversation a
route comes to name (``SessionStore.route_named``: ``/new``, ``/resume``, ``/branch``, ``/handoff``,
the context-exhaustion auto-reset, a reset policy, a topic binding heal), wired by ``watch_routes``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

_SLOT = 'detached'


def detached_route_of(source) -> str | None:
    """*source*'s pinned detached key; None for any other source (duck-typed test sources too)."""
    route = getattr(source, 'detached_route', None)
    return route if isinstance(route, str) else None


def detached_key(route: str, session_id: str) -> str:
    """The detached key of *session_id*, in the profile namespace of its chat's *route*."""
    parts = route.split(':')
    namespace = ':'.join(parts[:2]) if len(parts) > 1 and parts[0] == 'agent' else 'agent:main'
    return f'{namespace}:{_SLOT}:{session_id}'


def detached_session_id(key: str) -> str | None:
    parts = key.split(':')
    return parts[3] if len(parts) == 4 and parts[0] == 'agent' and parts[2] == _SLOT else None


def detach(authority, live, session_id) -> None:
    """Move *live* (a messaging conversation its chat no longer names) off the chat's route. A
    history mount (no source) has no route to leave, and a detached one has left it already."""
    if live.source is None or detached_route_of(live.source):
        return
    from gateway.session_identity import replace_source
    from gateway.session_local_recovery import local_adapter
    adapter = local_adapter(authority)  # its delivery; a messaging-only gateway has none yet
    live.route = detached_key(live.route, session_id)
    # No reply anchor: its replies reach the apps, and a background result's event is this very source.
    live.source = replace_source(live.source, detached_route=live.route, message_id=None)
    adapter.register_source(live.source)  # what admits its background results (``authorize_source``)


def lane_source(source):
    """*source* as its chat's own, a detached copy's pin dropped: what a conversation that takes the
    chat's route over starts from (the route moved first, so the one it replaced is detached by now)."""
    if detached_route_of(source) is None:
        return source
    from gateway.session_identity import replace_source
    return replace_source(source, detached_route=None)


def forget_detached(authority, key) -> None:
    """*key* names no live conversation any more (it came back onto its chat, or retired): its
    cached agent commits and goes, as at any boundary (a later idle eviction would commit it while the
    conversation runs elsewhere), and the store forgets the ids it owned (``detached_entry``)."""
    runner = authority.runner
    evict = getattr(runner, '_evict_cached_agent_at_boundary', None)
    if evict is not None:
        evict(key)
    drop_owner_hints(runner.session_store, key)


def drop_owner_hints(store, key) -> None:
    hints = getattr(store, '_session_owner_hints', None) or {}
    for session_id in [sid for sid, owner in list(hints.items()) if owner == key]:
        hints.pop(session_id, None)


def running_key(runner, route, session_id) -> str:
    """The key a turn of *session_id* runs under: its detached key while a turn that started there
    still runs (the chat took the conversation back meanwhile), so a Stop from the chat or the app
    reaches it; else *route*."""
    key = detached_key(route, session_id)
    return key if key in (getattr(runner, '_running_agents', None) or {}) else route


def chat_running_key(runner, route, entry) -> str:
    """``running_key`` for the conversation the chat's *route* names (*entry*, its routing entry)."""
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    return route if authority is None else running_key(runner, route, authority.logical_owner(entry.session_id))


def session_env_source(runner, source):
    """The source a turn's session variables (``HERMES_SESSION_PLATFORM``, ``_CHAT_ID``, ...) name.
    A conversation its chat moved past binds the local view a Desktop session's turn binds
    (``local_source``): whatever it schedules for "here" reaches the apps, never the chat."""
    key = detached_route_of(source)
    if key is None:
        return source
    from gateway.session_authorities import active_authority
    from gateway.session_local_recovery import local_source
    return local_source(active_authority(runner), detached_session_id(key), source.user_id)


def _route_authority(runner, store, route):
    """The authority owning *route*: its key names the profile, whatever scope the writer runs in."""
    from gateway.session_authorities import authority_for_home
    from hermes_constants import hermes_home_key
    registry = getattr(runner, 'session_authorities', None)
    if store._named_profile_for_key(route) is None:  # the launch profile's own key
        if registry is None:
            return getattr(runner, 'session_authority', None)
        return next((a for a in registry.serving() if hermes_home_key(a.profile_id) == registry.launch_key), None)
    home = store._profile_home_for_key(route)
    return authority_for_home(runner, home) if home is not None else None


def watch_routes(runner, store) -> None:
    """Wire *store*'s route announcements to the owning authority's ``name_lane``. A store write off
    the owner loop (a worker thread) hands the move to the loop, ahead of the writer's own resumption,
    so nothing on the loop sees the new route before the conversation it replaced has left it."""
    loop = asyncio.get_running_loop()

    def route_named(route, session_id, origin):
        authority = _route_authority(runner, store, route)
        if authority is None or origin is None:
            return

        def name():
            # Writers announce outside the store lock, so announcements can land out of order: one
            # the store has since moved past (a worker's /resume overtaken by a /new) would hand the
            # chat back a conversation it left; the newer write announces itself. Compared by logical
            # owner: a compression repoint since (unannounced) is still the same conversation.
            current = store.lookup_by_session_key(route)
            owner = authority.logical_owner(session_id)
            if current is None or authority.logical_owner(current.session_id) != owner:
                return
            authority.name_lane(route, owner, origin)
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            name()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(name)
    store.route_named = route_named


def recover_detached_sessions(authority, pending):
    """Boot: schedule the queued input of conversations their chats moved past (an app prompt
    accepted during a restart's after-turn wait). The routing index names only each chat's current
    conversation, so native recovery never binds these, and no chat message will wake them."""
    from gateway.session_contract import SessionRef
    from gateway.session_route_recovery import restore_routed_session
    scheduled = []
    from hermes_state_runtime import list_session_admissions
    for session_id in sorted(pending):
        if session_id in authority.sessions or not any(  # a lost (unknown) turn waits for /discard
                row['status'] == 'queued' for row in list_session_admissions(authority.db, session_id=session_id)):
            continue
        if restore_routed_session(authority, session_id) is None:
            continue
        if authority.sessions[session_id].detached:
            authority._schedule(SessionRef(authority.profile_id, session_id))
            scheduled.append(session_id)
    return scheduled


def detached_live(authority, route):
    """``(ref, live)`` of the conversation still live on detached key *route*, else None."""
    from gateway.session_contract import SessionRef
    session_id = detached_session_id(route)
    live = authority.sessions.get(session_id) if session_id else None
    if live is None or live.route != route:
        return None
    return SessionRef(authority.profile_id, session_id), live


def own_detached_key(key, session_id) -> bool:
    """*key* is *session_id*'s own detached key (the route it was on while its chat was elsewhere)."""
    return isinstance(key, str) and detached_session_id(key) == session_id


async def follow_detached_pin(runner, evt):
    """*evt* (a background result) on its pinned conversation's detached key, else *evt* itself.

    The chat's turn on a conversation started it, and the chat has moved on since, but the
    conversation is still open: the app used it after the chat's ``/new`` (a row the ``/new`` left
    ended stays a boundary, and its result is dropped once). The result lands in that conversation,
    for its app viewers, never in the chat's current one, and is never refused there on every retry
    (#141562). A conversation not mounted (a restart) mounts off the chat first."""
    key = str(evt.get('session_key') or '')
    pinned = str(evt.get('parent_session_id') or '').strip()
    from gateway.session_authorities import all_authorities
    if not key or not all_authorities(runner):
        return evt
    if (detached := detached_session_id(key)) is not None:
        return _follow_detached_key(runner, evt, key, detached)
    if not pinned:
        return evt
    store = runner.session_store
    authority = _route_authority(runner, store, key)
    entry = store.lookup_by_session_key(key) if authority is not None else None
    if authority is None or entry is None:
        return evt
    verdict, _owner, target = await runner._classify_completion_owner(pinned, key)
    session_id = authority.logical_owner(target) if verdict == 'deliver' else None
    if session_id is None or session_id == authority.logical_owner(entry.session_id):
        return evt
    row = authority.db.get_session(target)
    if row is None or row.get('ended_at'):
        return evt
    live = _mounted(authority, session_id)
    if live is None or live.route != detached_key(key, session_id):
        return evt
    return {**evt, 'session_key': live.route}


def _follow_detached_key(runner, evt, key, session_id):
    """A result of a turn that ran on detached *key*: on the chat's route once the chat names the
    conversation again (its ``/resume``), as that conversation's own result there."""
    authority = _route_authority(runner, runner.session_store, key)
    live = _mounted(authority, session_id) if authority is not None else None
    if live is None or live.route == key:
        return evt
    return {**evt, 'session_key': live.route}


def _mounted(authority, session_id):
    """*session_id*'s live session, mounted as its route would after a restart (off the chat when the
    chat names another conversation); None when it cannot be mounted now."""
    if session_id not in authority.sessions:
        from gateway.session_authorities import owner_scope
        from gateway.session_route_recovery import restore_routed_session
        from hermes_state_runtime import RuntimeStoreError
        try:
            with owner_scope(authority):
                restore_routed_session(authority, session_id)
        except RuntimeStoreError:  # draining: the drain retries it
            return None
    return authority.sessions.get(session_id)


def detached_live_source(runner, key):
    """The source of the conversation still live on detached key *key*, else None. The key's
    namespace names the owning profile: this runs before any profile scope is bound
    (``_completion_event_scope`` derives the scope from it)."""
    authority = _route_authority(runner, runner.session_store, key) if detached_session_id(key) else None
    found = detached_live(authority, key) if authority is not None else None
    return found[1].source if found is not None else None


def route_entry(authority, route):
    """The routing entry *route* names: the store's, or a detached key's live conversation (a
    background result it started finds it there, never on the chat's route)."""
    store = authority.runner.session_store
    entry = store.lookup_by_session_key(route)
    if entry is None and (found := detached_live(authority, route)) is not None:
        return detached_entry(store, found[1].source)
    return entry


def detached_entry(store, source):
    """The routing entry a detached *source* resolves to: its own conversation, at the compression
    tip a turn writes to. Never stored: the chat's route stays the only one this conversation has."""
    from gateway.session import SessionEntry
    key = source.detached_route
    session_id = str(detached_session_id(key))
    db = store._db_for_key(key)
    row = db.get_session(session_id) if db is not None else None
    tip = (db.get_compression_tip(session_id) if db is not None else None) or session_id
    # No route names it, so its by-id reads and writes (history, transcript, compression) find their
    # profile's store through the key; ``forget_detached`` drops these.
    hints = store._lazy('_session_owner_hints', dict)
    hints[session_id] = hints[tip] = key
    now = datetime.now()
    # Never a new conversation (that would fire session:start and first-contact notes).
    started = datetime.fromtimestamp(row['started_at']) if row and row.get('started_at') else now - timedelta(seconds=1)
    return SessionEntry(session_key=key, session_id=tip, created_at=started, updated_at=now, origin=source,
                        platform=source.platform, chat_type=source.chat_type)
