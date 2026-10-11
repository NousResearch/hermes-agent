"""Trusted API preparation and observation of the canonical durable FIFO."""
import asyncio
from contextvars import ContextVar
from contextlib import contextmanager
from functools import partial
import hmac
import json
import re
import uuid

from gateway.config import Platform
from gateway.session_api import bind_api_steps, restore_api_session, run_steps, run_steps_off_loop
from gateway.session_results import admission_result
from hermes_state_runtime import RuntimeStoreError, admit_session_input, _epoch, _json

api_execution: ContextVar[dict | None] = ContextVar('api_execution', default=None)
_SETTINGS_PREFIX = 'gateway.api.settings.v1.'
# Request-owned sinks an API observer may register for its admission's execution.
_OBSERVER_KEYS = ('stream_delta_callback', 'tool_start_callback', 'tool_complete_callback',
                  'reasoning_callback', 'status_callback', 'interim_assistant_callback')
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
        if (set(data) - {'history', 'settings', 'turn_author', 'media', 'run_owner_scope', 'browser_control'}
                or not {'history', 'settings'} <= set(data)
                or (data['history'] is not None and not isinstance(data['history'], list))
                or ('run_owner_scope' in data and not _valid_owner_scope(data['run_owner_scope']))
                or ('browser_control' in data and not _valid_browser_control(data['browser_control']))):
            raise RuntimeStoreError('invalid_params')
        if set(data['settings']) - set(_SETTING_KEYS):
            raise RuntimeStoreError('invalid_params')
    settings = payload.get('api_turn_v1', {}).get('settings') or api_settings(authority, ref)
    check_api_settings(adapter, settings)
    return adapter


def _valid_owner_scope(value):
    return isinstance(value, str) and _OWNER_SCOPE_RE.fullmatch(value) is not None


def _valid_browser_control(value):
    return (isinstance(value, dict) and set(value) == {'principal', 'transport_family'}
            and isinstance(value['principal'], str) and value['principal'].startswith('principal:')
            and value['transport_family'] in ('local-api', 'remote-api'))


def _request_browser_control():
    """The authenticated request's browser-controller identity, as the profile-prefix
    middleware derived it from the served profile and its API key; never client JSON."""
    from gateway.platforms.api_server import (
        _api_request_browser_control_principal, _api_request_browser_control_transport_family)
    bound = {'principal': _api_request_browser_control_principal.get(),
             'transport_family': _api_request_browser_control_transport_family.get()}
    return bound if _valid_browser_control(bound) else None


def _rebind_browser_control(adapter, data):
    """Re-derive the admitted principal from the profile's current API key before it is bound
    for execution (also after recovery). A rotated key no longer authorizes the old principal,
    so that turn runs unbound, like a request the middleware stamped nothing for."""
    bound = (data or {}).get('browser_control')
    if bound is None:
        return None
    from gateway.platforms.api_server import _api_request_profile
    profile = bound['principal'][len('principal:'):].rpartition(':')[0]
    token = _api_request_profile.set(None if profile == 'default' else profile)
    try:
        expected = adapter._derive_browser_control_principal(profile)
    finally:
        _api_request_profile.reset(token)
    return bound if hmac.compare_digest(bound['principal'], expected) else None


def api_session_vars():
    """Session-context fields an executing API admission binds beyond the shared source:
    #98619 history-delivery provenance and the rebound browser-controller identity. ``{}``
    when no API admission is executing."""
    current = api_execution.get()
    if current is None:
        return {}
    bound = current.get('browser_control') or {}
    return {'session_history_delivery': current['settings'].get('session_history_delivery') or '',
            'browser_control_principal': bound.get('principal', ''),
            'browser_control_transport_family': bound.get('transport_family', '')}


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
    """Admit inline (sync callers and tests); the HTTP doors use ``admit_api_turn_async``."""
    return run_steps(admit_api_steps(adapter, **kwargs))


async def admit_api_turn_async(adapter, steps=None, **kwargs):
    """``admit_api_turn`` (or *steps*, a generator ending in ``admit_api_steps``) with every SQLite
    write off the owner loop, in admission order."""
    from gateway.session_authorities import active_authority
    authority = active_authority(adapter.gateway_runner)
    if authority is None:
        raise RuntimeStoreError('profile_mismatch')
    return await run_steps_off_loop(authority, steps or admit_api_steps(adapter, **kwargs), orphaned=schedule_orphan)


def schedule_orphan(admitted):
    """A caller cancelled while its admission committed never observes it: its drain runs anyway."""
    authority, ref, row = admitted[:3]
    if row['status'] == 'queued':
        authority._schedule(ref)


def admit_api_steps(adapter, **kwargs):
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
    # Every API door (chat header, runs session_id, /api/sessions/{id}/chat, Responses chain) may
    # name a compression continuation; the binding, FIFO and retry identity live on its root.
    sid = authority.logical_owner(sid)
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
    browser_control = _request_browser_control()
    if browser_control is not None:
        # Server-derived like the owner scope: the admission keeps the identity the legacy
        # path bound for its executor, so execution and recovery can rebind it.
        payload['api_turn_v1']['browser_control'] = browser_control
    # Refuse a foreign-surface target before committing any media for it (a refused
    # admission otherwise retains its image bytes under native-inputs/ forever).
    existing = authority.db.get_session(sid)
    if existing is not None and existing['source'] not in ('api_server', 'bot_room'):
        raise RuntimeStoreError('permission_denied')
    if kwargs.get('turn_author') is not None:
        from agent.turn_author import parse_turn_author
        author = parse_turn_author(kwargs['turn_author'])
        if author is None:
            raise RuntimeStoreError('invalid_params')
        payload['api_turn_v1']['turn_author'] = author
    request_id = kwargs.get('request_id') or kwargs.get('active_run_id') or uuid.uuid4().hex
    if not isinstance(kwargs['user_message'], list):
        return (yield from _admit_api_payload(authority, adapter, sid, request_id, payload, settings, declared_key, kwargs))
    from gateway.session_api_media import commit_api_images
    from gateway.session_ingress_media import release_unheld_media
    payload['api_turn_v1']['media'] = commit_api_images(kwargs['user_message'])
    # Retained bytes belong to an accepted admission. A refused request (or an exact retry of a
    # retired one, whose references were erased) owns nothing, so its bytes are collected unless
    # another admission holds them.
    release = partial(release_unheld_media, authority.db, payload['api_turn_v1']['media'])
    try:
        admitted = yield from _admit_api_payload(authority, adapter, sid, request_id, payload, settings, declared_key, kwargs)
    except Exception:
        yield release
        raise
    yield release
    return admitted


def _admit_api_payload(authority, adapter, sid, request_id, payload, settings, declared_key, kwargs):
    from hermes_state_terminal import retry_terminal_admission
    row = retry_terminal_admission(authority.db, epoch=authority.epoch, principal_id='api',
        session_id=sid, request_id=request_id, payload=payload)
    if row is not None:
        check_api_settings(adapter, settings)
        from gateway.session_contract import SessionRef
        return authority, SessionRef(authority.profile_id, sid), row
    ref = yield from bind_api_steps(authority, sid, hosted_dispatch=kwargs.get("room_dispatch"), declared_key=declared_key)
    check_api_turn(authority, ref, payload)
    row = yield partial(admit_session_input, authority.db, epoch=authority.epoch, principal_id='api',
                        session_id=sid, request_id=request_id, payload=payload,
                        _authorize_write=authority._admission_gate(partial(_one_target, sid, request_id)))
    return authority, ref, row


def _one_target(session_id, request_id, conn):
    """One API request id (an Idempotency-Key) names one admission. The ledger keys identity by
    target too, so the same key aimed at another session would otherwise run the work again."""
    if conn.execute("SELECT 1 FROM session_admissions WHERE principal_id='api' AND request_id=? "
                    'AND target_session_id!=? LIMIT 1', (request_id, session_id)).fetchone():
        raise RuntimeStoreError('admission_conflict')


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
    from gateway.session_authorities import all_authorities, owner_scope
    for authority in all_authorities(adapter.gateway_runner):
        # Startup is unscoped (the launch profile's home): each profile's media root, config and
        # API bindings must pair with its own state.db, never the launch profile's.
        with owner_scope(authority):
            _recover_api_turns(adapter, authority)


def _recover_api_turns(adapter, authority):
    from hermes_state_runtime import list_session_admissions
    from gateway.session_ingress_media import collect_unheld_api_images
    import logging
    collect_unheld_api_images(authority.db)
    from gateway.session_api_media import compact_settled_api_payloads
    compact_settled_api_payloads(authority.db)
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


async def run_api_turn(adapter, *, approval_notify_callback=None, approval_session_key=None, **kwargs):
    admitted = await admit_api_turn_async(adapter, **kwargs)
    if approval_notify_callback is None or not approval_session_key:
        return await observe_api_turn(admitted, **kwargs)
    # A streaming surface advertises its own run id (``chatcmpl-*`` / ``run_*``): bind it to this
    # exact admission so ``/v1/runs/{id}/approval`` answers the shared, generation-fenced prompt,
    # and hand that prompt to the stream's notifier in the event shape it already emits.
    aliases = adapter._run_admission_aliases
    aliases[approval_session_key] = admitted[2]['admission_id']

    def sink(event_type, prompt):
        if event_type == 'approval.request':
            choices = prompt.get('choices') or ()
            approval_notify_callback({
                **{k: v for k, v in prompt.items() if k not in ('kind', 'prompt_id', 'choices')},
                'request_id': prompt['prompt_id'], 'allow_session': 'session' in choices,
                'allow_permanent': 'always' in choices})
    try:
        with observe_api_controls(admitted, sink):
            return await observe_api_turn(admitted, **kwargs)
    finally:
        if aliases.get(approval_session_key) == admitted[2]['admission_id']:
            aliases.pop(approval_session_key)


async def observe_api_turn(admitted, **kwargs):
    authority, ref, row = admitted
    from hermes_state_runtime import get_session_admission
    # The admission-time snapshot is stale once a drain already running ahead claims and settles
    # (or a stop cancels) the row; a waiter registered after that is never resolved. The current
    # row decides, read with no suspension before the waiter is registered.
    status = (get_session_admission(authority.db, admission_id=row['admission_id']) or row)['status']
    if status == 'unknown':
        raise RuntimeStoreError('unknown_execution')
    if status == 'terminal':
        result, usage = _settled_api_result(authority, row['admission_id'])
        callback = kwargs.get('stream_delta_callback')
        if callback:
            callback(result.get('final_response') or '')
        return result, usage
    waiter = authority.waiters.setdefault(row['admission_id'], asyncio.get_running_loop().create_future())
    observers = getattr(authority, 'api_observers', None)
    if observers is None:
        observers = authority.api_observers = {}
    observer = {key: kwargs[key] for key in _OBSERVER_KEYS if kwargs.get(key) is not None}
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
    return _settled_api_result(authority, row['admission_id'])


def _settled_api_result(authority, admission_id):
    saved = admission_result(authority.db, admission_id)
    if saved is None:
        from hermes_state_runtime import get_session_admission
        current = get_session_admission(authority.db, admission_id=admission_id)
        if current['outcome'] == 'cancelled':
            return {'final_response': '', 'interrupted': True, 'completed': False}, {}
        raise RuntimeStoreError('unknown_execution')
    return saved['result'], saved['usage']


_CONTROL_EVENTS = frozenset({'approval.request', 'approval.settled', 'clarify.request', 'clarify.settled'})


@contextmanager
def observe_api_controls(admitted, sink):
    """Project the same approval/clarify prompts WS viewers receive to ``sink(type, payload)``
    for the admission's lifetime; ``sink`` runs under the stream lock.

    An exact retry joins an admission whose prompt may already be pending: it was published
    before this observer existed, so it is replayed from the shared pending controls (the
    source ``GET /v1/runs/{id}`` reports). Subscribing and snapshotting in ONE stream-lock hold
    is the consistent cut: prompts are registered and answered under that lock, so each one is
    either in the snapshot or arrives live (never both), and one already answered is absent."""
    from hermes_state_runtime import get_session_admission
    authority, ref, row = admitted
    live = authority.sessions[ref.session_id]
    events = live.event_stream

    def observer(frame):
        params = frame['params']
        if params.get('admission_id') == row['admission_id'] and params.get('type') in _CONTROL_EVENTS:
            sink(params['type'], params.get('payload') or {})
    with events.lock:
        events.observers.add(observer)
        current = get_session_admission(authority.db, admission_id=row['admission_id']) or row
        if current['status'] == 'started':
            for prompt in live.controls.snapshot(ref.session_id, current['generation']):
                sink(prompt['kind'] + '.request', prompt)
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
            'content': content, 'turn_author': data.get('turn_author') if data else None,
            'browser_control': _rebind_browser_control(adapter, data)}


def _api_observers(authority, session_id):
    execution = authority.sessions[session_id].event_stream.execution
    admission_id = execution.get('admission_id') if execution else None
    return tuple(getattr(authority, 'api_observers', {}).get(admission_id, ()))


def _notify_observers(authority, session_id, key, *args, **kwargs):
    """Observer callbacks are request-owned sinks; one that raises (closed socket, torn-down
    loop) must not abort canonical execution or starve the other observers. Returns whether
    any observer accepted the event."""
    import logging
    accepted = False
    for observer in _api_observers(authority, session_id):
        callback = observer.get(key)
        if callback:
            try:
                callback(*args, **kwargs)
                accepted = True
            except Exception:
                logging.getLogger(__name__).warning('API observer %s failed for %s', key, session_id, exc_info=True)
    return accepted


def wire_api_observers(agent, owner, want_interim):
    """Hand this admission's API observers (Chat/Responses SSE, runs) the agent's reasoning,
    status and commentary callbacks, the ones the direct API path gave the agent itself. Each
    event is fenced to the exact running claim, so a settled worker's late callback is inert;
    the turn's own status/commentary sinks keep running after the observers."""
    authority, session_id, generation = owner

    def publish(key, *args, **kwargs):
        with authority.sessions[session_id].event_stream.lock:
            try:
                authority.check_approval_generation(session_id, generation)
            except RuntimeStoreError:
                return False
            return _notify_observers(authority, session_id, key, *args, **kwargs)

    status, interim = agent.status_callback, agent.interim_assistant_callback

    def status_callback(kind, message=None):
        publish('status_callback', kind, message)
        if status is not None:
            status(kind, message)

    def interim_assistant_callback(text, *, already_streamed=False):
        publish('interim_assistant_callback', text, already_streamed=already_streamed)
        if interim is not None:
            interim(text, already_streamed=already_streamed)
    agent.reasoning_callback = lambda text: publish('reasoning_callback', text)
    agent.status_callback = status_callback
    if want_interim:
        agent.interim_assistant_callback = interim_assistant_callback


def publish_api_event(authority, session_id, event_type, payload):
    if event_type == 'message.delta':
        return _notify_observers(authority, session_id, 'stream_delta_callback', payload['text'])
    return False


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
