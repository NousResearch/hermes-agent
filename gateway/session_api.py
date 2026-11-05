"""Server-only binding of API transcript identities to the existing TurnRunner."""
import asyncio
import hashlib
import logging

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource, _is_path_unsafe
from gateway.session_contract import SessionRef
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal, StoreUnavailable


def _handle(store, persistence):
    """*persistence*, else *store* when it is a storage interface handle. A caller that passes only its
    ``state.db`` has no handle to read through: ``StoreUnavailable``."""
    if persistence is not None:
        return persistence
    if hasattr(store, 'bindings'):
        return store
    raise StoreUnavailable('no storage interface handle for the API session lookup')


def declared_api_session(store, key, *, persistence=None):
    """The session the declared conversation key *key* names, else None. *store* is the profile's
    storage interface handle (*persistence*, when given, is used instead)."""
    store = _handle(store, persistence)
    found = store.bindings.declared_session(key)
    if found:
        return found
    # Earlier releases kept no declared-key registry: the declared key itself sat in
    # ``sessions.session_key`` (#139835). The pre-binding lookup finds that conversation, and
    # ``bind_api_steps`` adopts it under the same key, so later requests hit the registry.
    legacy = store.routing.latest_session_for_peer(source='api_server', session_key=key)
    return str(legacy.session.session_id) if legacy else None


def implicit_api_session(store, session_id, request_id=None, *, persistence=None):
    """The session a header-less Chat Completions turn lands on. Its id is derived from the
    conversation's opening, and a deleted id stays retired for good, so a NEW conversation that
    opens like a deleted one moves to the next generation of that opening, the way a deleted Bot
    Chat is replaced: the first ``api-<digest(id, n)>`` that is not retired. Read from durable rows
    only, so every later turn of that conversation (same opening) lands on the same session. An
    exact retry stays on the generation holding its request id's terminal identity; explicit ids
    never come here. *store* is the profile's storage interface handle (*persistence*, when given, is
    used instead)."""
    from persistence.authority.runtime import ApiGenerations, PinnedIdentity
    pinned = PinnedIdentity('api', request_id) if request_id else None
    return _handle(store, persistence).runtime.first_generation(ApiGenerations(session_id), pinned=pinned).session_id


def api_generation_id(session_id, generation):
    """Generation *generation* of an API session id ``implicit_api_session`` walks: the id itself,
    then ``api-<digest(id, n)>``."""
    if not generation:
        return session_id
    return 'api-' + hashlib.sha256(f'{session_id}\0{generation}'.encode()).hexdigest()[:16]


def continued_api_session(runner, session_id, request_id=None):
    """The session a new turn on a retained chain (``previous_response_id``, a named conversation)
    lands on: its stored id, or that id's next generation once the user deleted it (the history
    comes from the response store). An exact retry stays on the generation holding its identity."""
    from gateway.session_authorities import active_authority
    authority = active_authority(runner) if session_id else None
    if authority is None:
        return session_id
    return implicit_api_session(authority_persistence(authority), session_id, request_id)


def room_session_id(identity, generation=0):
    """``room_<digest>`` of a hosted room member's identity (home install, room, member, target
    profile); a later generation folds its number into the digest."""
    seed = '\0'.join(identity) + (f'\0{generation}' if generation else '')
    return 'room_' + hashlib.sha256(seed.encode()).hexdigest()[:32]


def hosted_member_session(store, identity, *, persistence=None):
    """``(generation, session_id)`` of the hidden session a hosted room member's dispatches land on:
    a deleted one stays retired, so the room's next task moves to the next generation. A room
    retry is replayed by the run's Idempotency-Key, never by this id, so nothing pins it. *store* is
    the profile's storage interface handle (*persistence*, when given, is used instead)."""
    from persistence.authority.runtime import RoomGenerations
    pick = _handle(store, persistence).runtime.first_generation(RoomGenerations(tuple(identity)))
    return pick.generation, pick.session_id


def _served_profile(authority):
    """Name of the profile *authority* serves; None for the launch profile (and for bare runners
    without a registry), which is what ``SessionSource.profile`` means on the wire."""
    registry = getattr(authority.runner, 'session_authorities', None)
    return registry.profile_name(authority) if registry is not None else None


def _pin_api_identity(runner, source, *, transport_profile=None):
    """Pin the shared-listener identity on an API source: transport = the launch profile whose
    listener received ``/p/<name>/``, runtime = ``source.profile``. Returns the transport profile to
    persist; None outside multiplexing, where the launch adapter is the only one by construction."""
    from gateway.session_identity import restore_identity
    if transport_profile is None:
        transport_profile = getattr(runner, '_primary_profile_name', None) or 'default'
    identity = restore_identity(source, runner=runner, transport_profile=transport_profile)
    return identity.transport_profile if identity is not None else None


def run_steps(steps):
    """Drive a write-step generator (see ``run_steps_off_loop``) inline: tests and sync callers."""
    result = error = None
    while True:
        try:
            write = steps.throw(error) if error is not None else steps.send(result)
        except StopIteration as done:
            return done.value
        try:
            result, error = write(), None
        except Exception as exc:  # health: allow BLE001 -- not swallowed: thrown back into the steps, which re-raise it
            result, error = None, exc


async def run_steps_off_loop(authority, steps, *, orphaned=None):
    """Drive a write-step generator on the owner loop, each yielded SQLite write in a worker thread
    in admission order (``tracked_write(ordered=True)``): a held writer must not freeze every session.
    The whole drive is one tracked, shielded task, so a cancelled caller cannot separate a commit
    from the steps after it; ``orphaned(result)`` then runs when the drive completes."""
    from gateway.session_runtime_workers import track_mutation, tracked_write

    async def drive():
        result = error = None
        while True:
            try:
                write = steps.throw(error) if error is not None else steps.send(result)
            except StopIteration as done:
                return done.value
            try:
                result, error = await tracked_write(authority, write, ordered=True), None
            except Exception as exc:  # health: allow BLE001 -- not swallowed: thrown back into the steps, which re-raise it
                result, error = None, exc
    async def serialized():
        # One API admission drive at a time per authority: its lookups, binding and admission stay
        # one step relative to every other API door, as they were when admission ran on the loop.
        lock = getattr(authority, '_api_admissions', None)
        if lock is None:
            lock = authority._api_admissions = asyncio.Lock()
        async with lock:
            return await drive()
    task = track_mutation(authority, serialized())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if orphaned is not None:
            task.add_done_callback(lambda done: done.cancelled() or done.exception() or orphaned(done.result()))
        raise


def bind_api_session(authority, session_id, *, hosted_dispatch=None, declared_key=None, caller_key=None,
                     walked_from=None):
    """Only the authenticated API edge may reserve an API source; never public RPC."""
    return run_steps(bind_api_steps(authority, session_id, hosted_dispatch=hosted_dispatch,
                                    declared_key=declared_key, caller_key=caller_key, walked_from=walked_from))


def bind_api_steps(authority, session_id, *, hosted_dispatch=None, declared_key=None, caller_key=None,
                   walked_from=None):
    """``bind_api_session`` as write steps (``run_steps``): the binding transaction is yielded.

    ``caller_key`` is this request's validated ``X-Hermes-Session-Key`` even when an explicit
    session id selected the conversation (``declared_key`` is set only when the key selected it).
    It is used only to adopt a row an earlier release stamped with that same key, and is then
    registered as that session's declared key (and stamped into its binding receipt).

    ``walked_from`` is the stored session a Responses/Runs chain continued from when the chain moved
    past it to *session_id* (``continued_api_session``): the fences it passed stay for good. If one
    aged out since that lookup, the bind refuses with the retryable ``chain_moved``."""
    authority._require_admission_open()
    if not isinstance(session_id, str) or not session_id or _is_path_unsafe(session_id):
        raise AuthorityRefusal('invalid_params')
    persistence = authority_persistence(authority)
    storage_source, title, room_identity, room_generation = 'api_server', None, None, 0
    if hosted_dispatch is not None:
        from gateway.hosted_room_peer import HostedMemberDispatch
        dispatch = HostedMemberDispatch.from_mapping(hosted_dispatch)
        room_identity = [dispatch.home_install_id, dispatch.room_id, dispatch.member_id, dispatch.target_profile]
        room_generation, expected = hosted_member_session(persistence, room_identity)
        if expected != session_id:
            raise AuthorityRefusal('admission_conflict')
        storage_source, title = 'bot_room', f'Group: {dispatch.room_id}'
    from gateway.session_history_mount import is_history_mount
    if session_id in authority.sessions and not is_history_mount(authority.sessions[session_id]):
        if (authority.sessions[session_id].source.platform != Platform.API_SERVER
                or persistence.sessions.get(session_id).source != storage_source):
            raise AuthorityRefusal('permission_denied')
    # An API session is the shared-listener row of the transport matrix: the launch profile's
    # ``api_server`` adapter receives ``/p/<name>/`` and answers it, while the turn RUNS under the
    # authority's own profile. Both ride the persisted source — ``source.profile`` places the runtime
    # (unset = launch profile, so a secondary's session would otherwise execute under the default's
    # scope) and ``transport_profile`` names the delivering bot after a restart.
    source = SessionSource(platform=Platform.API_SERVER, chat_id=session_id,
                           user_id='api', chat_type='dm', profile=_served_profile(authority))
    transport_profile = _pin_api_identity(authority.runner, source)
    route = authority.runner.session_store._generate_session_key(source)
    from gateway.session_lifecycle import _now
    now = _now()
    entry = SessionEntry(route, session_id, now, now, origin=source, platform=Platform.API_SERVER,
                         transport_profile=transport_profile)
    receipt = {'profile_id': authority.profile_id, 'session_id': session_id,
               'route': route, 'entry': entry.to_dict()}
    if declared_key:
        receipt['declared_key'] = declared_key
    if room_identity is not None:
        receipt.update(storage_source=storage_source, room_identity=room_identity)
        if room_generation:
            receipt['room_generation'] = room_generation

    from persistence.authority.bindings import NewApiBinding
    binding = NewApiBinding(session_id=session_id, route=route, storage_source=storage_source, title=title,
                            declared_key=declared_key or None, caller_key=caller_key or None,
                            origin=source.to_dict(), entry=entry.to_dict(), receipt=receipt,
                            started_at=now.timestamp())

    def write():
        # One write: a retired id, a chain the walk passed that moved since (``walked_from``) and a
        # conflicting binding, declared key, row or route refuse; a row an earlier release stamped with
        # the caller's key is adopted (#139835) and a route a heal replacement holds is taken back. The
        # admission gate is decided in this thread right before the write.
        persistence.bindings.bind_api(epoch=authority.epoch, binding=binding, gate=authority.gate_refusal(),
                                      walked_from=walked_from)
    yield write
    return restore_api_session(authority, session_id)


def _heal_replacement(persistence, caller, holder):
    """The route is derived from the caller's id, so only the stale-route heal (#54878) names another
    session on it, one with no binding of its own: the caller takes the route back, used or not, and
    the replacement keeps its transcript under its own id. A holder with its own binding refuses."""
    try:
        if persistence.bindings.api_binding(holder) is not None:
            return False
    except ValueError:
        return False  # a binding that does not decode is still the holder's own
    logging.getLogger(__name__).warning('API route of %s taken back from heal replacement %s', caller, holder)
    return True


def reopen_api_session(authority, ref):
    """The drain's admitted turn reopens a finalized API row, as ``reopen_local_session`` does a
    local one: the binding keeps the route on it (``SessionStore._api_bound``). So does a turn of a
    conversation its chat moved past (gateway/session_detached.py), whose row the chat's /new ended."""
    target = authority.physical_target(ref)
    persistence = authority_persistence(authority)
    row = persistence.sessions.get(target)
    if row is not None and row.end_reason is not None:
        persistence.sessions.reopen(target)


def api_storage_source(store, session_id, fallback):
    """*fallback*, or for ``api_server`` the storage source *session_id*'s binding names; *store* is the
    storage interface handle of the session's profile."""
    if fallback != 'api_server':
        return fallback
    binding = store.bindings.api_binding(session_id)
    return binding.storage_source if binding is not None else fallback


def restore_api_session(authority, session_id):
    from gateway.session_authority import LiveSession
    persistence = authority_persistence(authority)
    receipt = persistence.bindings.api_binding(session_id)
    if receipt is None:
        raise AuthorityRefusal('not_found')
    if receipt.profile_id != authority.profile_id:
        raise AuthorityRefusal('profile_mismatch')
    entry = SessionEntry.from_dict(receipt.entry)
    source = entry.origin
    _pin_api_identity(authority.runner, source, transport_profile=entry.transport_profile)
    row = persistence.sessions.get(session_id)
    if (receipt.session_id != session_id or entry.session_id != session_id
            or source.platform != Platform.API_SERVER or source.chat_id != session_id
            or row is None or row.source != receipt.storage_source
            or (row.session_key, row.chat_id, row.user_id) != (entry.session_key, session_id, source.user_id)
            or entry.session_key != authority.runner.session_store._generate_session_key(source)):
        raise AuthorityRefusal('admission_conflict')
    if receipt.storage_source == 'bot_room':
        identity = receipt.room_identity
        expected = room_session_id(identity, receipt.room_generation)
        if expected != session_id or row.title != 'Group: ' + identity[1] or not row.hidden:
            raise AuthorityRefusal('admission_conflict')
    store = authority.runner.session_store
    target = authority.physical_target(SessionRef(authority.profile_id, session_id))
    entry.session_id = target
    with store._lock:
        store._ensure_loaded_locked()
        current = store._entries.get(entry.session_key)
        if current is not None and current.session_id not in persistence.lineage.lineage(session_id):
            if not _heal_replacement(persistence, session_id, current.session_id):
                raise AuthorityRefusal('admission_conflict')
        store._entries[entry.session_key] = current if current is not None and current.session_id == target else entry
    from gateway.session_history_mount import is_history_mount
    live = authority.sessions.get(session_id)
    if live is None:
        authority.sessions[session_id] = LiveSession(source, entry.session_key)
    elif is_history_mount(live):
        # A read-only view opened while the row had no binding becomes the bound conversation in place,
        # so its viewers keep their subscriptions and event stream.
        live.source, live.route = source, entry.session_key
    return SessionRef(authority.profile_id, session_id)
