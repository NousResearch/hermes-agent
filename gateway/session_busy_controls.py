"""Session-scoped busy preferences and controls of the existing execution only."""
from functools import partial
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


def handlers(connection):
    return {
        'config.get': partial(busy_config, connection),
        'config.set': partial(busy_config, connection, write=True),
        'session.steer': partial(correct, connection, verb='steer'),
        'session.redirect': partial(correct, connection, verb='redirect'),
    }


def authorize(connection, ref, params, capability):
    connection.authority.authorize(connection.actor, ref, capability)
    if 'profile' in params:
        from hermes_cli.profiles import profile_matches_home
        profile = params['profile']
        if not isinstance(profile, str):
            raise RuntimeStoreError('invalid_params')
        if profile and not profile_matches_home(profile, Path(connection.authority.profile_id)):
            raise RuntimeStoreError('profile_mismatch')


async def busy_config(connection, ref, params, *, write=False):
    allowed = {'session_id', 'profile', 'key'} | ({'value'} if write else set())
    if params.get('key') == 'verbose' and not set(params) - allowed and ref.session_id:
        return verbose_config(connection, ref, params, write)
    if params.get('key') == 'show_reasoning' and not set(params) - allowed and ref.session_id:
        return show_reasoning_config(connection, ref, params, write)
    if params.get('key') == 'service_tier' and write and not set(params) - allowed and ref.session_id:
        return service_tier_config(connection, ref, params)
    if params.get('key') == 'yolo' and not set(params) - allowed and ref.session_id:
        return await yolo_config(connection, ref, params, write)
    if (set(params) - allowed or not ref.session_id or params.get('key') != 'busy'
            or (write and params.get('value') not in ('interrupt', 'steer', 'queue'))):
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control' if write else 'session:read')
    authority = connection.authority
    live = authority.sessions[ref.session_id]
    if write:
        # UI negotiation is not a settings write and must not rebuild frozen
        # launch policy, toolsets, history, or the retained AIAgent.
        live.busy_input_mode = params['value']
    value = getattr(live, 'busy_input_mode', None)
    if value is None:
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(authority.runner, live.source)
        # Main's TUI defaulted to ``queue`` (``ui-tui/src/app/useConfigSync.ts`` TUI_BUSY_DEFAULT): in a
        # full-screen composer the next prompt is authored while the agent streams, and an unintended
        # interrupt loses work. Only an explicit ``display.busy_input_mode`` overrides it; every other
        # surface keeps the framework default ``interrupt``.
        fallback = 'queue' if policy is not None and getattr(policy, 'source', None) == 'tui' else 'interrupt'
        display = (policy.config().get('display') or {}) if policy else {}
        value = (display.get('busy_input_mode') or fallback if policy else
                 authority.runner._effective_busy_input_mode(live.source))
        if value not in ('interrupt', 'steer', 'queue'):
            value = fallback
    return {'key': 'busy', 'value': value, 'scope': 'session'}


_VERBOSE_CYCLE = ('off', 'new', 'all', 'verbose')


def verbose_config(connection, ref, params, write):
    """Session-scoped /verbose: the tool-progress mode the next tool events of
    this session render with (read by TurnRunner), like busy never a settings write."""
    value = params.get('value')
    if write and value not in (*_VERBOSE_CYCLE, 'cycle'):
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control' if write else 'session:read')
    live = connection.authority.sessions[ref.session_id]
    current = getattr(live, 'tool_progress_mode', None) or 'all'
    if write:
        live.tool_progress_mode = current = (
            _VERBOSE_CYCLE[(_VERBOSE_CYCLE.index(current) + 1) % 4] if value == 'cycle' else value)
    return {'key': 'verbose', 'value': current, 'scope': 'session'}


def show_reasoning_config(connection, ref, params, write):
    """Session-scoped reasoning display (the classic CLI's ``/reasoning show|hide`` and verbose
    reasoning): whether a local-viewer turn of this session streams ``reasoning.delta``. Off until
    set, because a reasoning sink changes what the provider is asked for. Never a settings write."""
    value = params.get('value')
    word = str(value).strip().lower()
    if write and word not in _YOLO_WORDS:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control' if write else 'session:read')
    live = connection.authority.sessions[ref.session_id]
    if write:
        live.show_reasoning = _YOLO_WORDS[word]
    return {'key': 'show_reasoning', 'value': '1' if getattr(live, 'show_reasoning', None) is True else '0',
            'scope': 'session'}


def service_tier_config(connection, ref, params):
    """Session-scoped ``/fast`` (the classic CLI's tier pick, ``normal`` clears to normal): the
    session's service-tier override the next turn's agent reads (``current_service_tier``), never a
    settings write. ``--global`` persistence stays the client's own config.yaml write."""
    from agent.fast_mode import parse_exact_service_tier
    try:
        tier = parse_exact_service_tier(params.get('value')) if isinstance(params.get('value'), str) else None
    except ValueError:
        tier = None
    if tier is None:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control')
    live = connection.authority.sessions[ref.session_id]
    connection.authority.runner._set_session_service_tier_override(live.route, tier or None)
    return {'key': 'service_tier', 'value': tier or 'normal', 'scope': 'session'}


_YOLO_WORDS = {'1': True, 'on': True, 'true': True, '0': False, 'off': False, 'false': False}


async def yolo_config(connection, ref, params, write):
    """Session-scoped /yolo (TUI slash + Shift+Tab, Desktop toggle): this session's approval bypass,
    never HERMES_YOLO_MODE or approvals.mode. A ``--yolo`` launch is seeded first, so revoking it
    here is not re-enabled by the next turn's launch seeding. The route's persisted copy
    (``SessionEntry.yolo``, re-armed every turn by ``_restore_session_yolo``) is written before the
    live flag, like the messaging ``/yolo``, so an OFF here is not revived by the next turn."""
    value = params.get('value')
    if write and value is not None and str(value).strip().lower() not in _YOLO_WORDS:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control' if write else 'session:read')
    from gateway.session_policy import policy_for_source
    from tools.approval import apply_launch_yolo, disable_session_yolo, enable_session_yolo, is_session_yolo_enabled
    live = connection.authority.sessions[ref.session_id]
    runner = connection.authority.runner
    policy = policy_for_source(runner, live.source)
    if policy is not None and policy.yolo:
        apply_launch_yolo(live.route)
    store = runner.async_session_store if getattr(runner, 'session_store', None) is not None else None
    entry = await store.lookup_by_session_key(live.route) if store is not None else None
    # After a restart only the persisted copy is set until the next turn re-arms it: still ON.
    enabled = is_session_yolo_enabled(live.route) or getattr(entry, 'yolo', False) is True
    if write:
        enabled = not enabled if value is None else _YOLO_WORDS[str(value).strip().lower()]
        if store is not None:
            await store.set_session_yolo(live.route, enabled)
        (enable_session_yolo if enabled else disable_session_yolo)(live.route)
    return {'key': 'yolo', 'value': '1' if enabled else '0', 'scope': 'session'}


def adopts_owner_agent(authority, ref):
    """The claimed turn runs an in-process owner agent, which adopt_agent hands latched steers to;
    a cron job or a managed worker has no steer seam, so a latch there would be acknowledged and lost."""
    from gateway.session_managed_worker import managed_policy
    from gateway.session_policy import policy_for_source
    try:
        policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
        return not (policy is not None and policy.source == 'cron') and managed_policy(authority, ref) is None
    except RuntimeStoreError:
        return False


async def correct(connection, ref, params, *, verb):
    if (set(params) - {'session_id', 'profile', 'text', 'execution_generation'}
            or not isinstance(params.get('text'), str) or not params['text'].strip()
            or type(params.get('execution_generation')) is not int):
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control')
    authority = connection.authority
    generation = params['execution_generation']
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        authority.check_approval_generation(ref.session_id, generation)
        # Managed workers have a separate control channel, not a cached owner
        # agent. Refuse explicitly instead of acknowledging undeliverable text.
        if ref.session_id in getattr(authority, '_managed_workers', {}):
            raise RuntimeStoreError('unsupported_control')
        # Accepted for this exact claim before its turn built an agent: steer_agent latches it for
        # the agent this generation adopts, never a later one.
        agent, latched = authority.steer_agent(ref, generation, params['text'] if verb == 'steer' else None)
        if latched:
            return {'status': 'queued', 'text': params['text'], 'execution_generation': generation,
                    'authority_epoch': authority.epoch}
        method = getattr(agent, verb, None)
        if not callable(method):
            raise RuntimeStoreError('execution_not_ready')
        accepted = method(params['text'])
        return {'status': ({'steer': 'queued', 'redirect': 'redirected'}[verb] if accepted else 'rejected'),
                'text': params['text'], 'execution_generation': generation,
                'authority_epoch': authority.epoch}
