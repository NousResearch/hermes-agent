"""Trusted API preparation and observation of the canonical durable FIFO."""
import asyncio
from contextvars import ContextVar
from contextlib import contextmanager
import hmac
import json
import re
import uuid

from gateway.config import Platform
from gateway.session_api import bind_api_session, restore_api_session
from gateway.session_results import admission_result
from hermes_state_runtime import RuntimeStoreError, admit_session_input, _epoch, _json

api_execution: ContextVar[dict | None] = ContextVar('api_execution', default=None)
_SETTINGS_PREFIX = 'gateway.api.settings.v1.'
_SETTING_KEYS = ('ephemeral_system_prompt', 'requested_model', 'requested_provider',
                 'model_options', 'route', 'session_model', 'confirmed_runtime_lock',
                 'requested_runtime', 'route_source', 'room_dispatch', 'room_execution_policy',
                 'session_history_delivery')
_OWNER_SCOPE_RE = re.compile(r'[0-9a-f]{64}')


def api_settings(authority, ref):
    with authority.db._read_ctx() as conn:
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (_SETTINGS_PREFIX + ref.session_id,)).fetchone()
    return json.loads(saved[0]) if saved else {}


def check_api_turn(authority, ref, payload):
    live = authority.sessions[ref.session_id]
    if live.source.platform == Platform.API_SERVER:
        restore_api_session(authority, ref.session_id)
    adapter = authority.runner._adapter_for_source(live.source)
    if adapter is None or getattr(adapter, 'gateway_runner', None) is not authority.runner:
        raise RuntimeStoreError('runtime_draining')
    if 'api_turn_v1' in payload:
        data = payload['api_turn_v1']
        if (set(data) - {'history', 'settings', 'turn_author', 'media', 'run_owner_scope'}
                or not {'history', 'settings'} <= set(data)
                or (data['history'] is not None and not isinstance(data['history'], list))
                or ('run_owner_scope' in data and not _valid_owner_scope(data['run_owner_scope']))):
            raise RuntimeStoreError('invalid_params')
        if set(data['settings']) - set(_SETTING_KEYS):
            raise RuntimeStoreError('invalid_params')
    settings = payload.get('api_turn_v1', {}).get('settings') or api_settings(authority, ref)
    check_api_settings(adapter, settings)
    return adapter


def _valid_owner_scope(value):
    return isinstance(value, str) and _OWNER_SCOPE_RE.fullmatch(value) is not None


def check_api_settings(adapter, settings):
    dispatch = settings.get('room_dispatch')
    if dispatch is not None:
        from gateway.hosted_room_peer import HostedMemberDispatch, GatewayRoomCatalog
        from gateway.platforms.api_server_room_grants import _local_room_catalog
        from gateway import hosted_rooms
        bound = HostedMemberDispatch.from_mapping(dispatch)
        if bound.target_install_id != hosted_rooms.local_authority_gateway_id():
            raise RuntimeStoreError('permission_denied')
        _, catalog = _local_room_catalog(adapter, bound.target_profile, bound.target_install_id)
        current = GatewayRoomCatalog.from_mapping(catalog)
        if (current.catalog_digest != bound.capability_digest
                or current.execution_policy.as_mapping() != settings.get('room_execution_policy')):
            raise RuntimeStoreError('permission_denied')
    return adapter


@contextmanager
def api_policy_scope():
    current = api_execution.get()
    policy = current['settings'].get('room_execution_policy') if current else None
    token = None
    if policy is not None:
        from gateway.hosted_room_execution_policy import RoomExecutionPolicy, bind_room_execution_policy
        token = bind_room_execution_policy(RoomExecutionPolicy.from_mapping(policy))
    try:
        yield
    finally:
        if token is not None:
            from gateway.hosted_room_execution_policy import reset_room_execution_policy
            reset_room_execution_policy(token)


def admit_api_turn(adapter, **kwargs):
    # ``/p/<profile>/`` middleware scoped this request; the routed home's authority admits it.
    from gateway.session_authorities import active_authority
    authority = active_authority(adapter.gateway_runner)
    if authority is None or adapter._ensure_session_db() is not authority.db:
        raise RuntimeStoreError('profile_mismatch')
    sid = kwargs.get('session_id') or uuid.uuid4().hex
    declared_key = kwargs.get('gateway_session_key') if kwargs.get('bind_declared_conversation') else None
    if declared_key:
        from gateway.session_api import declared_api_session
        sid = declared_api_session(authority.db, declared_key) or sid
    authority._require_admission_open()
    settings = {key: kwargs.get(key) for key in _SETTING_KEYS}
    # Route credentials remain in the server's configuration, never admission JSON.
    route = settings.get('route')
    if route and route.get('api_key'):
        alias = settings.get('requested_model')
        configured = adapter._model_routes.get(alias)
        if configured != route:
            raise RuntimeStoreError('permission_denied')
        settings['route'] = {k: v for k, v in route.items() if k != 'api_key'}
    payload = json.loads(_json({'text': kwargs['user_message'], 'api_turn_v1': {
        'history': None if kwargs.get('history_from_session') else kwargs['conversation_history'], 'settings': settings}}))
    run_owner_scope = kwargs.get('run_owner_scope')
    if run_owner_scope is not None:
        if not _valid_owner_scope(run_owner_scope):
            raise RuntimeStoreError('invalid_params')
        # This opaque namespace is persisted in the same row/transaction as
        # admission. It is never a bearer credential or execution input.
        payload['api_turn_v1']['run_owner_scope'] = run_owner_scope
    if isinstance(kwargs['user_message'], list):
        from gateway.session_api_media import commit_api_images
        payload['api_turn_v1']['media'] = commit_api_images(kwargs['user_message'])
    if kwargs.get('turn_author') is not None:
        from agent.turn_author import parse_turn_author
        author = parse_turn_author(kwargs['turn_author'])
        if author is None:
            raise RuntimeStoreError('invalid_params')
        payload['api_turn_v1']['turn_author'] = author
    request_id = kwargs.get('request_id') or kwargs.get('active_run_id') or uuid.uuid4().hex
    from hermes_state_terminal import retry_terminal_admission
    row = retry_terminal_admission(authority.db, epoch=authority.epoch, principal_id='api',
        session_id=sid, request_id=request_id, payload=payload)
    if row is not None:
        check_api_settings(adapter, settings)
        from gateway.session_contract import SessionRef
        return authority, SessionRef(authority.profile_id, sid), row
    ref = bind_api_session(authority, sid, hosted_dispatch=kwargs.get("room_dispatch"), declared_key=declared_key)
    check_api_turn(authority, ref, payload)
    row = admit_session_input(authority.db, epoch=authority.epoch, principal_id='api',
                              session_id=sid, request_id=request_id, payload=payload)
    return authority, ref, row


def owns_api_run(adapter, run_id, owner_scope):
    """Match a caller scope against one canonical API admission, failing closed."""
    if not _valid_owner_scope(owner_scope):
        return False
    from gateway.platforms.api_server_authority_runs import run_admission
    try:
        owned = run_admission(adapter, run_id)
    except RuntimeStoreError:
        # Duplicate admissions for one run are an unanswered ownership question.
        return False
    if owned is None:
        return False
    try:
        stored = owned[1]['payload']['api_turn_v1']['run_owner_scope']
    except (KeyError, TypeError):
        return False
    return _valid_owner_scope(stored) and hmac.compare_digest(stored, owner_scope)


def recover_api_turns(adapter):
    """Recover committed work only after the real API adapter is published."""
    from gateway.session_authorities import all_authorities
    for authority in all_authorities(adapter.gateway_runner):
        _recover_api_turns(adapter, authority)


def _recover_api_turns(adapter, authority):
    from hermes_state_runtime import list_session_admissions
    import logging
    with authority.db._read_ctx() as conn:
        targets = [row[0] for row in conn.execute(
            "SELECT DISTINCT target_session_id FROM session_admissions WHERE principal_id='api' AND status='queued'")]
    for sid in targets:
        try:
            ref = restore_api_session(authority, sid)
            pending = list_session_admissions(authority.db, session_id=sid)
            if any(row['status'] == 'unknown' for row in pending):
                continue
            for row in pending:
                if row['status'] == 'queued':
                    check_api_turn(authority, ref, row['payload'])
            authority._schedule(ref)
        except RuntimeStoreError as exc:
            logging.getLogger(__name__).warning('API session %s paused: %s', sid, exc.reason)


async def run_api_turn(adapter, **kwargs):
    admitted = admit_api_turn(adapter, **kwargs)
    return await observe_api_turn(admitted, **kwargs)


async def observe_api_turn(admitted, **kwargs):
    authority, ref, row = admitted
    if row['status'] == 'unknown':
        raise RuntimeStoreError('unknown_execution')
    if row['status'] == 'terminal':
        result = admission_result(authority.db, row['admission_id'])
        if result is None:
            raise RuntimeStoreError('unknown_execution')
        callback = kwargs.get('stream_delta_callback')
        if callback:
            callback(result['result'].get('final_response') or '')
        return result['result'], result['usage']
    waiter = authority.waiters.setdefault(row['admission_id'], asyncio.get_running_loop().create_future())
    observers = getattr(authority, 'api_observers', None)
    if observers is None:
        observers = authority.api_observers = {}
    observer = {key: kwargs[key] for key in ('stream_delta_callback', 'tool_start_callback', 'tool_complete_callback')
                if kwargs.get(key) is not None}
    registered = observers.setdefault(row['admission_id'], [])
    registered.append(observer)
    try:
        authority._publish_pending(ref)
        authority._schedule(ref)
        await asyncio.shield(waiter)
    finally:
        # Shielding keeps the canonical turn alive past a cancelled request; only this
        # request's observer leaves, and the entry itself goes once the last one is gone.
        registered.remove(observer)
        if not registered:
            observers.pop(row['admission_id'], None)
    saved = admission_result(authority.db, row['admission_id'])
    if saved is None:
        from hermes_state_runtime import get_session_admission
        current = get_session_admission(authority.db, admission_id=row['admission_id'])
        if current['outcome'] == 'cancelled':
            return {'final_response': '', 'interrupted': True, 'completed': False}, {}
        raise RuntimeStoreError('unknown_execution')
    return saved['result'], saved['usage']


_CONTROL_EVENTS = frozenset({'approval.request', 'approval.settled', 'clarify.request', 'clarify.settled'})


@contextmanager
def observe_api_controls(admitted, sink):
    """Project the same approval/clarify prompts WS viewers receive to ``sink(type, payload)``
    for the admission's lifetime; ``sink`` runs on the publishing thread under the stream lock."""
    authority, ref, row = admitted
    events = authority.sessions[ref.session_id].event_stream

    def observer(frame):
        params = frame['params']
        if params.get('admission_id') == row['admission_id'] and params.get('type') in _CONTROL_EVENTS:
            sink(params['type'], params.get('payload') or {})
    with events.lock:
        events.observers.add(observer)
    try:
        yield
    finally:
        with events.lock:
            events.observers.discard(observer)


def prepare_api_execution(authority, ref, payload):
    adapter = check_api_turn(authority, ref, payload)
    data = payload.get('api_turn_v1')
    settings = data['settings'] if data else api_settings(authority, ref)
    if data:
        def write(conn):
            _epoch(conn, authority.epoch)
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                         'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         (_SETTINGS_PREFIX + ref.session_id, _json(settings)))
        authority.db._execute_write(write)
    content = payload['text']
    if data and isinstance(content, list):
        from gateway.session_api_media import restore_api_images
        content = restore_api_images(content, data.get('media') or [])
    return {'adapter': adapter, 'settings': settings, 'history': data['history'] if data else None,
            'content': content, 'turn_author': data.get('turn_author') if data else None}


def _api_observers(authority, session_id):
    execution = authority.sessions[session_id].event_stream.execution
    admission_id = execution.get('admission_id') if execution else None
    return tuple(getattr(authority, 'api_observers', {}).get(admission_id, ()))


def _notify_observers(authority, session_id, key, *args):
    """Observer callbacks are request-owned sinks; one that raises (closed socket, torn-down
    loop) must not abort canonical execution or starve the other observers."""
    import logging
    for observer in _api_observers(authority, session_id):
        callback = observer.get(key)
        if callback:
            try:
                callback(*args)
            except Exception:
                logging.getLogger(__name__).warning('API observer %s failed for %s', key, session_id, exc_info=True)


def publish_api_event(authority, session_id, event_type, payload):
    if event_type == 'message.delta':
        _notify_observers(authority, session_id, 'stream_delta_callback', payload['text'])


def publish_api_tool_event(authority, session_id, generation, event_type, call_id, tool_name, args, result=None):
    """Real tool arguments/results for the admission's API observers (Responses streaming,
    runs SSE); never part of the shared viewer event stream."""
    live = authority.sessions[session_id]
    with live.event_stream.lock:
        try:
            authority.check_approval_generation(session_id, generation)
        except RuntimeStoreError:
            return
        if event_type == 'tool.start':
            _notify_observers(authority, session_id, 'tool_start_callback', call_id, tool_name, args or {})
        elif event_type == 'tool.complete':
            _notify_observers(authority, session_id, 'tool_complete_callback', call_id, tool_name, args or {}, result)


def prepare_api_runtime(model, runtime_kwargs):
    current = api_execution.get()
    if current is None:
        return model, runtime_kwargs
    options = current['settings']
    route = options.get('route')
    configured = current['adapter']._model_routes.get(options.get('requested_model'))
    if configured and {k: v for k, v in configured.items() if k != 'api_key'} == route:
        route = configured
    model, _, _, _ = current['adapter']._select_agent_runtime(runtime_kwargs, model,
        requested_model=options.get('requested_model'), requested_provider=options.get('requested_provider'),
        route=route, session_model=options.get('session_model'),
        confirmed_runtime_lock=bool(options.get('confirmed_runtime_lock')),
        gateway_session_key=None, session_id=None)
    return model, runtime_kwargs
