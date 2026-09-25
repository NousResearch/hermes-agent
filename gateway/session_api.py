"""Server-only binding of API transcript identities to the existing TurnRunner."""
import json
import hashlib

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource, _is_path_unsafe
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, _epoch, _json

_BINDING_PREFIX = 'gateway.api.binding.v1.'
_DECLARED_PREFIX = 'gateway.api.conversation.v1.'


def declared_api_session(db, key):
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                           (_DECLARED_PREFIX + key,)).fetchone()
    return row[0] if row else None


def hosted_session_id(dispatch):
    identity = (dispatch.home_install_id, dispatch.room_id, dispatch.member_id, dispatch.target_profile)
    return 'room_' + hashlib.sha256('\0'.join(identity).encode()).hexdigest()[:32]


def prospective_room_session(authority, dispatch):
    """Derive the canonical identity and reject conflicts without binding it."""
    sid = hosted_session_id(dispatch)
    title = 'Group: ' + dispatch.room_id
    source = SessionSource(platform=Platform.API_SERVER, chat_id=sid, user_id='api',
                           chat_type='dm', profile=_served_profile(authority))
    _pin_api_identity(authority.runner, source)
    route = authority.runner.session_store._generate_session_key(source)
    with authority.db._read_ctx() as conn:
        _epoch(conn, authority.epoch)
        row = conn.execute('SELECT source,title,session_key FROM sessions WHERE id=?', (sid,)).fetchone()
        if row is not None and (row['source'] != 'bot_room' or row['title'] != title
                                or row['session_key'] not in (None, '', route)):
            raise RuntimeStoreError('admission_conflict')
        if conn.execute('SELECT 1 FROM sessions WHERE title=? AND id!=?', (title, sid)).fetchone():
            raise RuntimeStoreError('admission_conflict')
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (_BINDING_PREFIX + sid,)).fetchone()
        if saved is not None:
            binding = json.loads(saved[0])
            identity = [dispatch.home_install_id, dispatch.room_id, dispatch.member_id, dispatch.target_profile]
            if (binding.get('profile_id') != authority.profile_id or binding.get('session_id') != sid
                    or binding.get('storage_source') != 'bot_room' or binding.get('room_identity') != identity
                    or binding.get('route') != route):
                raise RuntimeStoreError('admission_conflict')
        routing = conn.execute("SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=?", (route,)).fetchone()
        if routing is not None and json.loads(routing[0])['session_id'] != sid:
            raise RuntimeStoreError('admission_conflict')
    live = authority.sessions.get(sid)
    if live is not None and live.source != source:
        raise RuntimeStoreError('admission_conflict')
    return sid


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


def bind_api_session(authority, session_id, *, hosted_dispatch=None, declared_key=None):
    """Only the authenticated API edge may reserve an API source; never public RPC."""
    authority._require_admission_open()
    if not isinstance(session_id, str) or not session_id or _is_path_unsafe(session_id):
        raise RuntimeStoreError('invalid_params')
    storage_source, title, room_identity = 'api_server', None, None
    if hosted_dispatch is not None:
        from gateway.hosted_room_peer import HostedMemberDispatch
        dispatch = HostedMemberDispatch.from_mapping(hosted_dispatch)
        room_identity = [dispatch.home_install_id, dispatch.room_id, dispatch.member_id, dispatch.target_profile]
        expected = hosted_session_id(dispatch)
        if expected != session_id:
            raise RuntimeStoreError('admission_conflict')
        storage_source, title = 'bot_room', f'Group: {dispatch.room_id}'
    if session_id in authority.sessions:
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

    def write(conn):
        from gateway.session_selected_route import own_session_creation
        with own_session_creation(authority, session_id, source, storage_source, conn):
            _epoch(conn, authority.epoch)
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
            if row is not None and row['session_key'] not in (None, '', route):
                raise RuntimeStoreError('admission_conflict')
            existing = conn.execute("SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=?",
                                    (route,)).fetchone()
            if existing is not None and json.loads(existing[0])['session_id'] != session_id:
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
    authority.db._execute_write(write)
    return restore_api_session(authority, session_id)


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
        expected = 'room_' + hashlib.sha256('\0'.join(identity).encode()).hexdigest()[:32]
        if expected != session_id or row['title'] != 'Group: ' + identity[1] or not row['hidden']:
            raise RuntimeStoreError('admission_conflict')
    store = authority.runner.session_store
    from gateway.session_selected_route import session_store_guard
    with session_store_guard(authority.runner):
        store._ensure_loaded_locked()
        current = store._entries.get(entry.session_key)
        if current is not None and current.session_id != session_id:
            raise RuntimeStoreError('admission_conflict')
        store._entries[entry.session_key] = entry
    authority.sessions.setdefault(session_id, LiveSession(source, entry.session_key))
    return SessionRef(authority.profile_id, session_id)
