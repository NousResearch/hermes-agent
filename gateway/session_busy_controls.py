"""Session-scoped busy preferences and controls of the existing execution only."""
import asyncio
import contextlib
from functools import partial
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


def handlers(connection):
    return {
        'config.get': partial(busy_config, connection),
        'config.set': partial(busy_config, connection, write=True),
        'session.steer': partial(correct, connection, verb='steer'),
        'session.redirect': partial(correct, connection, verb='redirect'),
        # The queue strip's source of truth in shared-gateway mode. Standalone
        # tui_gateway serves the same names against its own session dict.
        'session.pending': partial(pending_view, connection),
        'session.steer_clear': partial(clear_steer, connection),
    }


def authorize(connection, ref, params, capability):
    connection.authority.authorize(connection.actor, ref, capability)
    # The contract's ``profile`` is nullable; null means this owner, as an omitted key does.
    if params.get('profile') is not None:
        from hermes_cli.profiles import profile_matches_home
        profile = params['profile']
        if not isinstance(profile, str):
            raise RuntimeStoreError('invalid_params')
        if profile and not profile_matches_home(profile, Path(connection.authority.profile_id)):
            raise RuntimeStoreError('profile_mismatch')


async def busy_config(connection, ref, params, *, write=False):
    if write and params.get('key') == 'model':
        # The legacy picker verb. On the owner a model pick is ``session.mutate`` operation=model
        # (revision-fenced, retry-idempotent, selection-guarded); a config write would bypass all
        # three. Refused before authorization or any read: nothing is written or deferred.
        raise RuntimeStoreError('use_session_mutation_model')
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
    reasoning): whether a local-viewer turn of this session streams ``reasoning.delta``. Unset, it
    reports the launch default the turn uses. Never a settings write."""
    value = params.get('value')
    word = str(value).strip().lower()
    if write and word not in _YOLO_WORDS:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control' if write else 'session:read')
    live = connection.authority.sessions[ref.session_id]
    if write:
        live.show_reasoning = _YOLO_WORDS[word]
    shown = getattr(live, 'show_reasoning', None)
    if shown is None:
        from gateway.local_viewer_events import launch_shows_reasoning
        from gateway.session_policy import policy_for_source
        shown = launch_shows_reasoning(policy_for_source(connection.authority.runner, live.source),
                                       connection.authority)
    return {'key': 'show_reasoning', 'value': '1' if shown is True else '0', 'scope': 'session'}


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
    here is not re-enabled by the next turn's launch seeding. The write goes through the shared
    ``tools.approval_yolo.toggle_session_yolo`` contract on the route's persisted copy
    (``SessionEntry.yolo``, restored by ``restore_gateway_yolo``), like the messaging
    ``/yolo``: persisted before the live flip, so an OFF here is not revived by the next turn."""
    value = params.get('value')
    if write and value is not None and str(value).strip().lower() not in _YOLO_WORDS:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control' if write else 'session:read')
    from gateway.session_policy import policy_for_source
    from tools.approval import is_session_yolo_enabled
    from tools.approval_yolo import restore_gateway_yolo, toggle_session_yolo
    live = connection.authority.sessions[ref.session_id]
    runner = connection.authority.runner
    policy = policy_for_source(runner, live.source)
    store = getattr(runner, 'session_store', None)
    entry = await runner.async_session_store.lookup_by_session_key(live.route) if store is not None else None
    restore_gateway_yolo(live.route, getattr(entry, 'yolo', None), launch=bool(policy is not None and policy.yolo))
    # After a restart only the persisted copy is set until the next turn re-arms it: still ON.
    persisted = getattr(entry, 'yolo', False) is True
    enabled = is_session_yolo_enabled(live.route) or persisted
    if write:
        persist = (lambda on: store.set_session_yolo(live.route, on)) if store is not None else None
        # Off the loop: the persist is a routing-store write (thread-safe), as in the messaging /yolo.
        enabled = await asyncio.to_thread(
            toggle_session_yolo, live.route, None if value is None else _YOLO_WORDS[str(value).strip().lower()],
            persisted=persisted, persist=persist)
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
    # TUI-26: text typed during a running turn in Ink's interrupt/steer busy modes arrives here, not
    # at prompt.submit. A bare stop phrase in a LOCAL voice chat ends the chat and reaches no agent,
    # whatever generation is running.
    voice_stop = getattr(connection, '_typed_voice_stop', None)
    if voice_stop is not None and await voice_stop(ref, params):
        return {'voice_stopped': True}
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
                    'authority_epoch': authority.epoch, 'pending': 'steer'}
        method = getattr(agent, verb, None)
        if not callable(method):
            raise RuntimeStoreError('execution_not_ready')
        accepted = method(params['text'])
        status = {'steer': 'queued', 'redirect': 'redirected'}[verb] if accepted else 'rejected'
        result = {'status': status, 'text': params['text'], 'execution_generation': generation,
                  'authority_epoch': authority.epoch}
        # ``pending`` says where this text is waiting. A tool-time redirect is a
        # steer (the agent will not kill the tool); only a model redirect waits
        # in the redirect slot. The TUI uses it to pick the strip vs a user row.
        if accepted:
            result['pending'] = _accepted_surface(agent, verb, params['text'], latched)
        return result


def _text_slot(agent, name):
    value = getattr(agent, name, None) if agent is not None else None
    return value if isinstance(value, str) and value.strip() else None


def _live_agent(authority, ref):
    """The agent this generation adopted, else the route's resident agent.

    A test or pre-turn authority may not have a cache lookup yet; that is no
    agent, not an error. ``SessionAuthority.agent`` is the production read once
    the lookup exists."""
    _generation, adopted = authority.adopted.get(ref.session_id, (None, None))
    if adopted is not None:
        return adopted
    runner = authority.runner
    lookup = getattr(runner, '_resident_agent_for', None) or getattr(runner, '_cached_agent_for', None)
    if lookup is None:
        return None
    return authority.agent(ref)


def _stream_lock(authority, ref):
    live = authority.sessions.get(ref.session_id)
    return live.event_stream.lock if live is not None else contextlib.nullcontext()


def _accepted_surface(agent, verb, text, latched):
    """``steer`` when the text is waiting to inject; ``redirect`` when it will
    rebuild the live model request as a user message."""
    if latched or verb == 'steer':
        return 'steer'
    redirect = _text_slot(agent, '_pending_redirect')
    if redirect and text.strip() in redirect:
        return 'redirect'
    return 'steer'


def correction_snapshot(authority, ref):
    """Pending steer (agent slot, else the pre-adopt latch), pending model
    redirect, and queued-admission texts. The running prompt is not part of the
    queue — it is already the live turn."""
    with _stream_lock(authority, ref):
        agent = _live_agent(authority, ref)
        _generation, texts = authority.pending_steers.get(ref.session_id, (None, []))
    latched = "\n".join(text.strip() for text in texts if isinstance(text, str) and text.strip())
    steer = _text_slot(agent, '_pending_steer') or latched or None
    redirect = _text_slot(agent, '_pending_redirect')
    queue = []
    for row in authority._pending_rows(ref):
        if row.get('status') != 'queued':
            continue
        payload = row.get('payload') or {}
        text = payload.get('text') if isinstance(payload, dict) else None
        if isinstance(text, str) and text:
            queue.append(text)
    return steer, redirect, queue


async def pending_view(connection, ref, params):
    if set(params) - {'session_id', 'profile'} or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:read')
    steer, redirect, queue = correction_snapshot(connection.authority, ref)
    result = {'steer': steer, 'queue': queue}
    if redirect:
        result['redirect'] = redirect
    return result


def _clear_agent_steer(agent):
    if agent is None:
        return False
    clearer = getattr(agent, 'clear_pending_steer', None)
    if callable(clearer):
        return bool(clearer())
    lock = getattr(agent, '_pending_steer_lock', None)
    if lock is not None:
        with lock:
            had = getattr(agent, '_pending_steer', None) is not None
            agent._pending_steer = None
            return had
    had = getattr(agent, '_pending_steer', None) is not None
    agent._pending_steer = None
    return had


async def clear_steer(connection, ref, params):
    """Drop a pending steer that has not been injected: the pre-adopt latch and
    the agent slot. A model redirect is not a steer and stays put."""
    if set(params) - {'session_id', 'profile'} or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:control')
    authority = connection.authority
    with _stream_lock(authority, ref):
        _generation, texts = authority.pending_steers.pop(ref.session_id, (None, ()))
        agent = _live_agent(authority, ref)
    cleared = bool(texts) or _clear_agent_steer(agent)
    return {'status': 'cleared' if cleared else 'noop', 'cleared': cleared}
