"""Server-only binding of API transcript identities to the existing TurnRunner."""
import asyncio
from functools import partial
import hashlib
import json
import logging

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource, _is_path_unsafe
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, _epoch, _json

from hermes_state_local import API_BINDING_PREFIX as _BINDING_PREFIX, API_DECLARED_PREFIX as _DECLARED_PREFIX


def declared_api_session(db, key):
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                           (_DECLARED_PREFIX + key,)).fetchone()
    if row:
        return row[0]
    # Earlier releases kept no declared-key registry: the declared key itself sat in
    # ``sessions.session_key`` (#139835). The pre-binding lookup finds that conversation, and
    # ``bind_api_steps`` adopts it under the same key, so later requests hit the registry.
    legacy = db.find_latest_gateway_session_for_peer(source='api_server', session_key=key)
    return legacy['id'] if legacy else None


def implicit_api_session(db, session_id, request_id=None):
    """The session a header-less Chat Completions turn lands on. Its id is derived from the
    conversation's opening, and a deleted id stays retired for good, so a NEW conversation that
    opens like a deleted one moves to the next generation of that opening, the way a deleted Bot
    Chat is replaced: the first ``api-<digest(id, n)>`` that is not retired. Read from durable rows
    only, so every later turn of that conversation (same opening) lands on the same session. An
    exact retry stays on the generation holding its request id's terminal identity; explicit ids
    never come here."""
    from hermes_state_mutation_retirement import first_generation
    from hermes_state_terminal import identity_key
    pinned = None if not request_id else lambda conn, sid: conn.execute(
        'SELECT 1 FROM state_meta WHERE key=?', (identity_key('api', sid, request_id),)).fetchone() is not None
    return first_generation(db, partial(api_generation_id, session_id), pinned=pinned)[1]


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
    return implicit_api_session(authority.db, session_id, request_id) if authority is not None else session_id


def room_session_id(identity, generation=0):
    """``room_<digest>`` of a hosted room member's identity (home install, room, member, target
    profile); a later generation folds its number into the digest."""
    seed = '\0'.join(identity) + (f'\0{generation}' if generation else '')
    return 'room_' + hashlib.sha256(seed.encode()).hexdigest()[:32]


def hosted_member_session(db, identity):
    """``(generation, session_id)`` of the hidden session a hosted room member's dispatches land on:
    a deleted one stays retired, so the room's next task moves to the next generation. A room
    retry is replayed by the run's Idempotency-Key, never by this id, so nothing pins it."""
    from hermes_state_mutation_retirement import first_generation
    return first_generation(db, lambda n: room_session_id(identity, n))


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
        raise RuntimeStoreError('invalid_params')
    storage_source, title, room_identity, room_generation = 'api_server', None, None, 0
    if hosted_dispatch is not None:
        from gateway.hosted_room_peer import HostedMemberDispatch
        dispatch = HostedMemberDispatch.from_mapping(hosted_dispatch)
        room_identity = [dispatch.home_install_id, dispatch.room_id, dispatch.member_id, dispatch.target_profile]
        room_generation, expected = hosted_member_session(authority.db, room_identity)
        if expected != session_id:
            raise RuntimeStoreError('admission_conflict')
        storage_source, title = 'bot_room', f'Group: {dispatch.room_id}'
    from gateway.session_history_mount import is_history_mount
    if session_id in authority.sessions and not is_history_mount(authority.sessions[session_id]):
        if (authority.sessions[session_id].source.platform != Platform.API_SERVER
                or authority.db.get_session(session_id)['source'] != storage_source):
            raise RuntimeStoreError('permission_denied')
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

    def write(conn):
        _epoch(conn, authority.epoch)
        authority._require_admission_open()  # a drain that began while this waited on the writer wins
        from hermes_state_mutation_retirement import RETIRED_PREFIX, keep_passed_fences
        if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + session_id,)).fetchone():
            raise RuntimeStoreError('not_found')
        # The walk ran before this write: a sweep marker it passed may have been reclaimed since,
        # and binding the successor then would split the chain (the next walk stops at the root).
        if walked_from and not keep_passed_fences(conn, partial(api_generation_id, walked_from), session_id):
            raise RuntimeStoreError('chain_moved')
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (_BINDING_PREFIX + session_id,)).fetchone()
        if saved is not None:
            binding = json.loads(saved[0])
            if binding.get('storage_source', 'api_server') != storage_source:
                raise RuntimeStoreError('permission_denied')
            if declared_key and binding.get('declared_key') != declared_key:
                raise RuntimeStoreError('admission_conflict')
            return
        if declared_key:
            existing_declared = conn.execute('SELECT value FROM state_meta WHERE key=?',
                (_DECLARED_PREFIX + declared_key,)).fetchone()
            if existing_declared and existing_declared[0] != session_id:
                raise RuntimeStoreError('admission_conflict')
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO NOTHING',
                         (_DECLARED_PREFIX + declared_key, session_id))
        row = conn.execute('SELECT source,session_key,title FROM sessions WHERE id=?', (session_id,)).fetchone()
        if row is not None and row['source'] != storage_source:
            raise RuntimeStoreError('permission_denied')
        if storage_source == 'bot_room':
            if row is not None and row['title'] != title:
                raise RuntimeStoreError('admission_conflict')
            if conn.execute('SELECT 1 FROM sessions WHERE title=? AND id!=?', (title, session_id)).fetchone():
                raise RuntimeStoreError('admission_conflict')
        # Earlier API releases stored the caller-declared key directly on the session row
        # (#139835). A caller presenting that exact key adopts the row onto the canonical route;
        # the declared-key registry above already refused a key owned by another session, and any
        # other stored key (another conversation's, a foreign route) still conflicts.
        legacy_key = declared_key or caller_key
        if row is not None and row['session_key'] not in (None, '', route):
            if not legacy_key or row['session_key'] != legacy_key:
                raise RuntimeStoreError('admission_conflict')
            # The rewrite below replaces the stamped key with the route: the registry keeps it,
            # so a later request selecting the conversation by that key still finds this row.
            owner = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                 (_DECLARED_PREFIX + legacy_key,)).fetchone()
            if owner is not None and owner[0] != session_id:
                raise RuntimeStoreError('admission_conflict')
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO NOTHING',
                         (_DECLARED_PREFIX + legacy_key, session_id))
            receipt['declared_key'] = legacy_key
        existing = conn.execute("SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=?",
                                (route,)).fetchone()
        held = json.loads(existing[0])['session_id'] if existing is not None else session_id
        if held != session_id and not _heal_replacement(conn, session_id, held):
            raise RuntimeStoreError('admission_conflict')
        conn.execute('''INSERT INTO sessions(id,source,title,hidden,started_at) VALUES(?,?,?,?,?)
                        ON CONFLICT(id) DO NOTHING''', (session_id, storage_source, title,
                                                       int(storage_source == 'bot_room'), now.timestamp()))
        conn.execute('UPDATE sessions SET session_key=?,chat_id=?,user_id=?,chat_type=?,origin_json=? WHERE id=?',
                     (route, session_id, source.user_id, 'dm', _json(source.to_dict()), session_id))
        conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?) "
                     'ON CONFLICT(scope,session_key) DO UPDATE SET entry_json=excluded.entry_json',
                     (route, _json(entry.to_dict()), now.timestamp()))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (_BINDING_PREFIX + session_id, _json(receipt)))
    yield partial(authority.db._execute_write, write)
    return restore_api_session(authority, session_id)


def _heal_replacement(conn, caller, holder):
    """The route is derived from the caller's id, so only the stale-route heal (#54878) names another
    session on it, one with no binding of its own: the caller takes the route back, used or not, and
    the replacement keeps its transcript under its own id. A holder with its own binding refuses."""
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (_BINDING_PREFIX + holder,)).fetchone():
        return False
    logging.getLogger(__name__).warning('API route of %s taken back from heal replacement %s', caller, holder)
    return True


def reopen_api_session(authority, ref):
    """The drain's admitted turn reopens a finalized API row, as ``reopen_local_session`` does a
    local one: the binding keeps the route on it (``SessionStore._api_bound``)."""
    target = authority.physical_target(ref)
    row = authority.db.get_session(target)
    if row is not None and row.get('end_reason') is not None:
        authority.db.reopen_session(target)


def api_storage_source(db, session_id, fallback):
    if fallback != 'api_server':
        return fallback
    with db._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (_BINDING_PREFIX + session_id,)).fetchone()
    return json.loads(saved[0]).get('storage_source', fallback) if saved else fallback


def restore_api_session(authority, session_id):
    from gateway.session_authority import LiveSession
    with authority.db._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (_BINDING_PREFIX + session_id,)).fetchone()
    if saved is None:
        raise RuntimeStoreError('not_found')
    receipt = json.loads(saved[0])
    if receipt['profile_id'] != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    entry = SessionEntry.from_dict(receipt['entry'])
    source = entry.origin
    _pin_api_identity(authority.runner, source, transport_profile=entry.transport_profile)
    row = authority.db.get_session(session_id)
    if (receipt['session_id'] != session_id or entry.session_id != session_id
            or source.platform != Platform.API_SERVER or source.chat_id != session_id
            or row is None or row['source'] != receipt.get('storage_source', 'api_server')
            or (row['session_key'], row['chat_id'], row['user_id']) != (entry.session_key, session_id, source.user_id)
            or entry.session_key != authority.runner.session_store._generate_session_key(source)):
        raise RuntimeStoreError('admission_conflict')
    if receipt.get('storage_source') == 'bot_room':
        identity = receipt['room_identity']
        expected = room_session_id(identity, receipt.get('room_generation', 0))
        if expected != session_id or row['title'] != 'Group: ' + identity[1] or not row['hidden']:
            raise RuntimeStoreError('admission_conflict')
    store = authority.runner.session_store
    target = authority.physical_target(SessionRef(authority.profile_id, session_id))
    entry.session_id = target
    with store._lock:
        store._ensure_loaded_locked()
        current = store._entries.get(entry.session_key)
        if current is not None and current.session_id not in authority.db.get_compression_lineage(session_id):
            with authority.db._read_ctx() as conn:
                if not _heal_replacement(conn, session_id, current.session_id):
                    raise RuntimeStoreError('admission_conflict')
        store._entries[entry.session_key] = current if current is not None and current.session_id == target else entry
    from gateway.session_history_mount import is_history_mount
    live = authority.sessions.get(session_id)
    if live is None:
        authority.sessions[session_id] = LiveSession(source, entry.session_key)
    elif is_history_mount(live):
        # A read-only view opened while the row had no binding becomes the bound conversation in place,
        # so its viewers keep their subscriptions and event stream.
        live.source, live.route, live.superseded, live.history_mount = source, entry.session_key, False, False
    return SessionRef(authority.profile_id, session_id)
