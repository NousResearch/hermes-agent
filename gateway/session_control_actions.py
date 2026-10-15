"""Desktop's goal / loop / heartbeat / subgoal card buttons (``session.control``) on the owner.

``session.control.read`` was served, so the cards rendered as controllable, while every button
answered -32601. Each action is the typed slash command the button stands for, run through
``execute_command`` (the gateway's own /goal /loop /heartbeat /subgoal handlers, with their
capability, local-source, admission and busy rules), answered with the owner's control snapshot.
"""
from hermes_state_runtime import RuntimeStoreError

_ACTION_COMMANDS = {
    'goal.pause': ('goal', 'pause'), 'goal.resume': ('goal', 'resume'),
    'goal.clear': ('goal', 'clear'), 'goal.unwait': ('goal', 'unwait'),
    'loop.pause': ('loop', 'pause'), 'loop.resume': ('loop', 'resume'), 'loop.stop': ('loop', 'stop'),
    'heartbeat.pause': ('heartbeat', 'pause'), 'heartbeat.resume': ('heartbeat', 'resume'),
    'heartbeat.clear': ('heartbeat', 'clear'), 'subgoal.clear': ('subgoal', 'clear'),
}
_SET_ACTIONS = frozenset({'goal.set', 'goal.wait_time', 'loop.set', 'heartbeat.set'})
_ENVELOPE = ('type', 'output', 'notice', 'message', 'display')


def _command(action, args):
    """``(name, arg)`` of the slash command for *action*, or ``('subgoal', None)`` for an add, whose
    free text must not be parsed as a /subgoal verb ("clear the cache" is a criterion, not a clear)."""
    if not isinstance(args, dict) or set(args) - {'text', 'index'}:
        raise RuntimeStoreError('invalid_params')
    if action in _ACTION_COMMANDS:
        return _ACTION_COMMANDS[action]
    if action == 'subgoal.remove':
        index = args.get('index')
        if type(index) is not int or index < 1:
            raise RuntimeStoreError('invalid_params')
        return 'subgoal', f'remove {index}'
    if action == 'subgoal.add':
        if not isinstance(args.get('text'), str) or not args['text'].strip():
            raise RuntimeStoreError('invalid_params')
        return 'subgoal', None
    raise RuntimeStoreError('invalid_params')


def _set_args(action, args):
    """Validate typed setter fields before a manager or its persisted state is opened."""
    if not isinstance(args, dict):
        raise RuntimeStoreError('invalid_params')
    if action == 'goal.set':
        title, maximum, contract = args.get('title'), args.get('max_turns'), args.get('contract')
        if (set(args) - {'title', 'max_turns', 'contract'} or not isinstance(title, str) or not title.strip()
                or (maximum is not None and (type(maximum) is not int or maximum < 1))
                or (contract is not None and (not isinstance(contract, dict)
                    or set(contract) - {'outcome', 'verification', 'constraints', 'boundaries', 'stop_when'}
                    or not all(isinstance(value, str) for value in contract.values())))):
            raise RuntimeStoreError('invalid_params')
        return {'title': title.strip(), 'max_turns': maximum, 'contract': contract}
    if action == 'goal.wait_time':
        seconds, reason = args.get('seconds'), args.get('reason', '')
        if set(args) - {'seconds', 'reason'} or type(seconds) is not int or seconds < 1 or not isinstance(reason, str):
            raise RuntimeStoreError('invalid_params')
        return {'seconds': seconds, 'reason': reason.strip()}
    if action == 'loop.set':
        prompt, interval, times, until = args.get('prompt'), args.get('interval_seconds'), args.get('times', 0), args.get('until', '')
        if (set(args) - {'prompt', 'interval_seconds', 'times', 'until'} or not isinstance(prompt, str) or not prompt.strip()
                or (interval is not None and type(interval) is not int) or type(times) is not int or times < 0
                or not isinstance(until, str)):
            raise RuntimeStoreError('invalid_params')
        return {'prompt': prompt.strip(), 'interval_seconds': interval, 'times': times, 'until': until.strip()}
    prompt, interval = args.get('prompt'), args.get('interval_seconds')
    if (set(args) - {'prompt', 'interval_seconds'} or not isinstance(prompt, str) or not prompt.strip()
            or type(interval) is not int or interval < 60):
        raise RuntimeStoreError('invalid_params')
    return {'prompt': prompt.strip(), 'interval_seconds': interval}


def _loop_route(source):
    platform = getattr(source, 'platform', '')
    route = {'platform': platform.value if hasattr(platform, 'value') else str(platform or '')}
    route.update({key: str(getattr(source, key, '') or '')
                  for key in ('chat_id', 'chat_type', 'thread_id', 'user_id', 'user_name', 'profile')})
    return {key: value for key, value in route.items() if value}


async def _set_control(connection, ref, params, action, args):
    """Run a setter at the owner under the same authorization and workspace scope as slash commands."""
    from pathlib import Path
    from gateway.run import _profile_runtime_scope
    from gateway.session_ancillary import physical_target
    from gateway.session_busy_controls import authorize
    from gateway.session_commands import _session_workspace
    from gateway.session_local import authorize_local_source
    from gateway.session_viewer_automation import invoke

    authority = connection.authority
    authorize(connection, ref, params, 'session:control')
    live = authority.sessions[ref.session_id]
    if authorize_local_source(authority.runner, live.source) is not True:
        raise RuntimeStoreError('permission_denied')
    authority._require_admission_open()
    runner, target = authority.runner, physical_target(authority, ref)
    with _profile_runtime_scope(Path(authority.profile_id)), _session_workspace(runner, live.source):
        if action == 'loop.set':
            from hermes_cli.loops import LoopManager

            await runner._warm_goals_session_db('loop manager')
            LoopManager(session_id=target).set(**args, route=_loop_route(live.source))
        elif action == 'heartbeat.set':
            reply = invoke(runner, 'heartbeat', target, 'set', [args['prompt']], {'interval_seconds': args['interval_seconds']})
            quick_key = runner._session_key_for_source(live.source)
            runner._register_heartbeat_watch(quick_key, live.source, target)
        elif action == 'goal.set':
            reply = invoke(runner, 'goal', target, 'set', [args['title']],
                           {key: value for key, value in {'max_turns': args['max_turns'], 'contract': args['contract']}.items()
                            if value is not None})
        else:
            reply = invoke(runner, 'goal', target, 'wait_for_seconds', [args['seconds']], {'reason': args['reason']})
        if action != 'loop.set' and reply.get('error'):
            raise RuntimeStoreError('invalid_params')
    return {'type': 'exec', 'output': '✓ Control set.'}


async def _add_subgoal(connection, ref, params, text):
    """``/subgoal <text>``'s add branch (``slash_commands_goals``) on the session's goal manager."""
    from pathlib import Path
    from agent.i18n import t
    from gateway.platforms.base import MessageEvent
    from gateway.run import _profile_runtime_scope
    from gateway.session_busy_controls import authorize
    from gateway.session_local import authorize_local_source
    from gateway.slash_commands_goals import _mgr_call
    authority = connection.authority
    authorize(connection, ref, params, 'session:control')
    live = authority.sessions[ref.session_id]
    if authorize_local_source(authority.runner, live.source) is not True:
        raise RuntimeStoreError('permission_denied')
    authority._require_admission_open()
    with _profile_runtime_scope(Path(authority.profile_id)):
        mgr, _entry = await authority.runner._get_goal_manager_for_event(
            MessageEvent(text='/subgoal', source=live.source))
        if mgr is None:
            return {'type': 'exec', 'output': t('gateway.goal.unavailable')}
        if not mgr.has_goal():
            return {'type': 'exec', 'output': t('gateway.subgoal.no_goal')}
        added, err = _mgr_call('/subgoal', mgr.add_subgoal, text)
        index = len(mgr.state.subgoals) if mgr.state else 0
    return {'type': 'exec', 'output': err or t('gateway.subgoal.added', index=index, text=added)}


async def control(connection, ref, params):
    if set(params) - {'session_id', 'profile', 'action', 'args'} or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    action = params.get('action')
    if not isinstance(action, str):
        raise RuntimeStoreError('invalid_params')
    args = {} if params.get('args') is None else params['args']
    action = action.strip()
    setter = action in _SET_ACTIONS
    if setter:
        args = _set_args(action, args)
    else:
        name, arg = _command(action, args)
    if ref.session_id not in connection.authority.sessions:
        raise RuntimeStoreError('not_found')
    scope = {key: params[key] for key in ('session_id', 'profile') if key in params}
    if setter:
        result = await _set_control(connection, ref, scope, action, args)
    elif arg is None:
        result = await _add_subgoal(connection, ref, scope, args['text'].strip())
    else:
        from gateway.session_commands import execute_command
        result = await execute_command(connection, ref, {**scope, 'name': name, 'arg': arg}, dispatch=True)
    from gateway.session_ancillary import read
    snapshot = await read(connection, ref, scope, kind='control')
    # Other viewers (another Desktop window, Ink) repaint their cards from this event, not the reply.
    connection.authority.sessions[ref.session_id].event_stream.publish(
        ref.session_id, snapshot, event_type='session.control.update')
    return {**snapshot, 'dispatch': {key: result.get(key) for key in _ENVELOPE}}
